"""
hmm_engine.py – Gaussian Hidden Markov Model for market regime detection.

Trains on daily returns, realised volatility, and (optionally) spread features
to classify the market into discrete regimes (Bull / Bear / Chop).

Key outputs consumed downstream:
  - Current regime label (str)
  - Posterior probabilities for each regime (np.ndarray)
  - Full regime history (pd.DataFrame)

The risk_manager uses the posterior probabilities for position sizing and
the pairs_engine uses the regime label to filter entry signals.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from config import (
    ANNUAL_TRADING_DAYS,
    HMM_COV_TYPE,
    HMM_N_ITER,
    HMM_N_REGIMES,
    HMM_RANDOM_SEED,
    HMM_TRAINING_WINDOW,
    REGIME_LABELS,
)

logger = logging.getLogger(__name__)

# Suppress hmmlearn convergence warnings during grid search / short windows
warnings.filterwarnings("ignore", category=DeprecationWarning, module="hmmlearn")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RegimeState:
    """Snapshot of the current regime detection output."""
    label: str                          # e.g. "BULL", "BEAR", "CHOP"
    state_id: int                       # Raw HMM state index (0, 1, 2)
    confidence: float                   # Posterior probability of current state
    posteriors: Dict[str, float]        # {label: probability} for all states
    is_favorable_for_pairs: bool        # True if regime supports mean-reversion


@dataclass
class RegimeHistory:
    """Full time-series output from the HMM."""
    states: pd.Series                   # Integer state IDs indexed by date
    labels: pd.Series                   # String labels indexed by date
    posterior_probs: pd.DataFrame       # Columns = regime labels, indexed by date
    model: GaussianHMM                  # Fitted model (for serialisation / reuse)
    state_means: Dict[str, float]       # Mean return per regime (for labelling)
    state_vols: Dict[str, float]        # Mean vol per regime


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(
    returns: pd.DataFrame,
    volatility: pd.DataFrame,
    tickers: Optional[List[str]] = None,
    vol_window: int = 21,
) -> pd.DataFrame:
    """
    Build the feature matrix for the HMM.

    Features are chosen specifically for pairs-trading regime detection.
    The key question: is the market TRENDING (bad for pairs) or
    MEAN-REVERTING (good for pairs)?

    Features per observation (row = trading day):
      1. Cross-sectional mean return   (market direction)
      2. Return autocorrelation (10d)  (trending vs mean-reverting)
      3. Cross-sectional return dispersion (divergence/convergence)
      4. Smoothed return momentum (21d) (trend strength and direction)

    Feature 2 is the critical pairs-trading discriminator:
      - Positive autocorrelation = momentum/trending → pairs lose money
      - Negative autocorrelation = mean-reverting → pairs make money
      - Near zero = random walk → pairs are neutral

    Volatility is deliberately excluded. Mining stocks have structurally
    high vol (60-70% annualised). Previous versions used raw or normalised
    vol as a feature, causing the HMM to mislabel normal mining-stock
    volatility as BEAR. For pairs trading, trend structure matters more
    than vol level.
    """
    if tickers is not None:
        available = [t for t in tickers if t in returns.columns]
        returns = returns[available]
        volatility = volatility[available]

    features = pd.DataFrame(index=returns.index)

    # 1. Market return (cross-sectional mean)
    features["market_return"] = returns.mean(axis=1)

    # 2. Return autocorrelation (10-day rolling)
    # Positive = trending (bad for pairs), negative = mean-reverting (good)
    market_ret = features["market_return"]
    features["market_vol"] = market_ret.rolling(10, min_periods=5).apply(
        lambda x: x.autocorr(lag=1), raw=False,
    )

    # 3. Return dispersion (cross-sectional std of daily returns)
    features["return_dispersion"] = returns.std(axis=1)

    # 4. Smoothed return momentum (21-day rolling mean return)
    features["return_momentum"] = market_ret.rolling(21, min_periods=10).mean()

    # Drop NaN rows
    features = features.replace([np.inf, -np.inf], np.nan).dropna()

    logger.info("Built feature matrix: %d rows x %d features", len(features), len(features.columns))
    return features


# ---------------------------------------------------------------------------
# HMM training
# ---------------------------------------------------------------------------

def fit_hmm(
    features: pd.DataFrame,
    n_regimes: int = HMM_N_REGIMES,
    n_iter: int = HMM_N_ITER,
    covariance_type: str = HMM_COV_TYPE,
    random_state: int = HMM_RANDOM_SEED,
) -> Tuple[GaussianHMM, np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit a Gaussian HMM to the feature matrix.

    Features are standardised before fitting so that columns on different
    scales (e.g. returns ~0.001 vs vol ~0.3) contribute equally. Without
    this, the HMM collapses to degenerate posteriors (0/1).

    Returns:
      - model: fitted GaussianHMM
      - hidden_states: array of predicted state labels (int) per row
      - posteriors: (n_samples, n_states) matrix of posterior probabilities
      - scaler: fitted StandardScaler (needed for label_states to invert)
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        verbose=False,
    )

    model.fit(X)

    hidden_states = model.predict(X)
    posteriors = model.predict_proba(X)

    logger.info(
        "HMM fitted: %d regimes, converged=%s, score=%.2f",
        n_regimes, model.monitor_.converged, model.score(X),
    )

    return model, hidden_states, posteriors, scaler


# ---------------------------------------------------------------------------
# State labelling
# ---------------------------------------------------------------------------

def label_states(
    model: GaussianHMM,
    features: pd.DataFrame,
    hidden_states: np.ndarray,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[Dict[int, str], Dict[str, float], Dict[str, float]]:
    """
    Assign semantic labels (BULL / BEAR / CHOP) to HMM states based on
    a composite score of return direction and volatility.

    Labelling heuristic (for 3 states):
      Uses the momentum feature (smoothed 21-day return) to assign labels:
      - Highest momentum → BULL (strong uptrend)
      - Lowest momentum → BEAR (strong downtrend)
      - Middle → CHOP (range-bound, ideal for pairs)

      Volatility is excluded from labelling because mining stocks have
      structurally high vol that would otherwise bias toward BEAR.
    """
    n_states = model.n_components
    return_col = list(features.columns).index("market_return")
    momentum_col = list(features.columns).index("return_momentum")

    # Model means are in scaled space — invert to original scale for labelling
    if scaler is not None:
        original_means = scaler.inverse_transform(model.means_)
    else:
        original_means = model.means_

    # Extract per-state mean return and mean autocorrelation (stored as market_vol)
    state_mean_returns = {}
    state_mean_vols = {}
    for s in range(n_states):
        state_mean_returns[s] = original_means[s, return_col]
        # Use momentum for labelling — it directly measures trend direction
        state_mean_vols[s] = original_means[s, momentum_col]

    # Sort by momentum: lowest = BEAR, highest = BULL, middle = CHOP
    sorted_by_momentum = sorted(range(n_states), key=lambda s: original_means[s, momentum_col])

    label_map = {}
    if n_states == 3:
        label_map[sorted_by_momentum[0]] = "BEAR"
        label_map[sorted_by_momentum[1]] = "CHOP"
        label_map[sorted_by_momentum[2]] = "BULL"
    elif n_states == 2:
        label_map[sorted_by_momentum[0]] = "BEAR"
        label_map[sorted_by_momentum[1]] = "BULL"
    else:
        for s in range(n_states):
            label_map[s] = REGIME_LABELS.get(sorted_by_momentum[s], f"STATE_{s}")

    # Build readable dicts keyed by label
    means_by_label = {label_map[s]: state_mean_returns[s] for s in range(n_states)}
    vols_by_label = {label_map[s]: state_mean_vols[s] for s in range(n_states)}

    logger.info("Regime labelling: %s", {v: f"ret={state_mean_returns[k]:.5f} vol={state_mean_vols[k]:.4f}" for k, v in label_map.items()})

    return label_map, means_by_label, vols_by_label


# ---------------------------------------------------------------------------
# Full regime detection pipeline
# ---------------------------------------------------------------------------

def detect_regimes(
    returns: pd.DataFrame,
    volatility: pd.DataFrame,
    tickers: Optional[List[str]] = None,
    n_regimes: int = HMM_N_REGIMES,
    training_window: Optional[int] = HMM_TRAINING_WINDOW,
) -> RegimeHistory:
    """
    Run the complete regime detection pipeline:
      1. Build features from returns and volatility
      2. Optionally trim to trailing `training_window` days
      3. Fit HMM
      4. Label states semantically
      5. Return full history + current state

    This is the main entry point for the dashboard and risk manager.
    """
    features = build_features(returns, volatility, tickers=tickers)

    # Optionally use only the trailing window for training
    if training_window is not None and len(features) > training_window:
        train_features = features.iloc[-training_window:]
    else:
        train_features = features

    model, hidden_states, posteriors, scaler = fit_hmm(train_features, n_regimes=n_regimes)
    label_map, state_means, state_vols = label_states(model, train_features, hidden_states, scaler=scaler)

    # Build time-series outputs
    states = pd.Series(hidden_states, index=train_features.index, name="state")
    labels = states.map(label_map)
    labels.name = "regime"

    posterior_df = pd.DataFrame(
        posteriors,
        index=train_features.index,
        columns=[label_map[s] for s in range(n_regimes)],
    )

    return RegimeHistory(
        states=states,
        labels=labels,
        posterior_probs=posterior_df,
        model=model,
        state_means=state_means,
        state_vols=state_vols,
    )


# ---------------------------------------------------------------------------
# Current regime snapshot
# ---------------------------------------------------------------------------

def get_current_regime(history: RegimeHistory) -> RegimeState:
    """
    Extract the latest regime state from a RegimeHistory.

    This is the object passed to pairs_engine.generate_signal() and
    risk_manager for position sizing decisions.
    """
    latest_label = history.labels.iloc[-1]
    latest_state = history.states.iloc[-1]
    latest_posteriors = history.posterior_probs.iloc[-1]

    confidence = latest_posteriors[latest_label]

    posteriors_dict = latest_posteriors.to_dict()

    # CHOP regime is favorable for pairs trading (mean-reversion works)
    # BULL/BEAR are unfavorable (trending markets break pairs)
    favorable = latest_label in ("CHOP", "MEAN_REVERTING")

    return RegimeState(
        label=latest_label,
        state_id=int(latest_state),
        confidence=float(confidence),
        posteriors=posteriors_dict,
        is_favorable_for_pairs=favorable,
    )


# ---------------------------------------------------------------------------
# Regime transition matrix (for dashboard display)
# ---------------------------------------------------------------------------

def get_transition_matrix(history: RegimeHistory) -> pd.DataFrame:
    """
    Extract the fitted transition probability matrix from the HMM.

    Returns a DataFrame where entry (i, j) = P(next_state = j | current_state = i).
    Rows and columns are labelled with regime names.
    """
    model = history.model
    n_states = model.n_components

    # Reconstruct label map from posterior_probs columns
    regime_names = list(history.posterior_probs.columns)

    trans_matrix = pd.DataFrame(
        model.transmat_,
        index=regime_names,
        columns=regime_names,
    )

    return trans_matrix


# ---------------------------------------------------------------------------
# Regime stability metric
# ---------------------------------------------------------------------------

def regime_stability(history: RegimeHistory, lookback: int = 21) -> float:
    """
    Compute a stability score for the current regime over the last `lookback` days.

    Returns a float in [0, 1]:
      - 1.0 = regime has been the same every day for `lookback` days
      - 0.0 = regime changes every day

    The risk_manager uses this to scale confidence: a stable regime deserves
    more conviction than a flickering one.
    """
    recent = history.labels.iloc[-lookback:]
    if len(recent) == 0:
        return 0.0

    current = recent.iloc[-1]
    agreement = (recent == current).sum() / len(recent)
    return float(agreement)


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from data_loader import load_universe

    logger.info("Loading data...")
    data = load_universe(start="2022-01-01")

    logger.info("Detecting regimes...")
    history = detect_regimes(data["returns"], data["volatility"])

    current = get_current_regime(history)
    stability = regime_stability(history)
    trans = get_transition_matrix(history)

    print(f"\n{'='*60}")
    print(f"CURRENT REGIME: {current.label}")
    print(f"Confidence:     {current.confidence:.1%}")
    print(f"Stability (21d): {stability:.1%}")
    print(f"Favorable for pairs: {current.is_favorable_for_pairs}")
    print(f"\nPosterior probabilities:")
    for label, prob in current.posteriors.items():
        print(f"  {label:6s}: {prob:.1%}")

    print(f"\nRegime means (daily return):")
    for label, mean in history.state_means.items():
        print(f"  {label:6s}: {mean:+.5f}")

    print(f"\nTransition matrix:")
    print(trans.round(3).to_string())

    print(f"\nRegime history (last 10 days):")
    print(history.labels.tail(10).to_string())

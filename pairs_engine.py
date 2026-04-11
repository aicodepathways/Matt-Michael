"""
pairs_engine.py – Statistical Arbitrage / Pairs Trading engine.

Pipeline:
  1. Cointegration scanning   (Engle-Granger two-step)
  2. Rolling hedge ratios      (OLS regression)
  3. Dynamic Z-score tracking
  4. Signal generation         (filtered by HMM regime when available)

All outputs are DataFrames/dicts suitable for the Streamlit dashboard.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import coint

from config import (
    DEFAULT_COINT_PVALUE,
    DEFAULT_LOOKBACK,
    DEFAULT_ZSCORE_ENTRY,
    DEFAULT_ZSCORE_EXIT,
    DEFAULT_ZSCORE_STOP,
    MAX_HALF_LIFE,
    MIN_HALF_LIFE,
)
from data_loader import get_ticker_pairs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for structured output
# ---------------------------------------------------------------------------

@dataclass
class PairProfile:
    """Result of cointegration testing for a single pair."""
    ticker_a: str
    ticker_b: str
    coint_pvalue: float
    adf_stat: float
    hedge_ratio: float
    half_life: float
    is_valid: bool
    rejection_reason: str = ""


@dataclass
class PairSignal:
    """Live trading signal for an active pair."""
    ticker_a: str
    ticker_b: str
    zscore: float
    hedge_ratio: float
    half_life: float
    signal: str              # "LONG_A_SHORT_B", "SHORT_A_LONG_B", "EXIT", "STOP", "FLAT"
    regime: Optional[str] = None
    regime_confidence: Optional[float] = None
    coint_pvalue: float = 0.0
    spread_mean: float = 0.0
    spread_std: float = 0.0


# ---------------------------------------------------------------------------
# Cointegration testing
# ---------------------------------------------------------------------------

def engle_granger_test(
    y: pd.Series,
    x: pd.Series,
) -> Tuple[float, float, float]:
    """
    Run Engle-Granger cointegration test.

    Returns (coint_pvalue, adf_statistic, hedge_ratio).
    The hedge ratio is the OLS beta from regressing y on x.
    """
    combined = pd.concat([y, x], axis=1).dropna()
    if len(combined) < 60:
        return (1.0, 0.0, 0.0)

    y_clean = combined.iloc[:, 0]
    x_clean = combined.iloc[:, 1]

    coint_stat, pvalue, crit_values = coint(y_clean, x_clean, trend="c")

    x_const = add_constant(x_clean)
    model = OLS(y_clean, x_const).fit()
    hedge_ratio = model.params.iloc[1]

    return (pvalue, coint_stat, hedge_ratio)


def compute_spread(
    y: pd.Series,
    x: pd.Series,
    hedge_ratio: float,
) -> pd.Series:
    """Compute the cointegration spread: y - hedge_ratio * x."""
    return y - hedge_ratio * x


def estimate_half_life(spread: pd.Series) -> float:
    """
    Estimate the mean-reversion half-life of the spread via
    an AR(1) regression: delta_spread = phi * spread_lag + eps.

    half_life = -log(2) / log(1 + phi)
    If phi >= 0 the spread is not mean-reverting → return inf.
    """
    spread = spread.dropna()
    if len(spread) < 30:
        return float("inf")

    lag = spread.shift(1)
    delta = spread.diff()
    combined = pd.concat([delta, lag], axis=1).dropna()
    combined.columns = ["delta", "lag"]

    x = add_constant(combined["lag"])
    model = OLS(combined["delta"], x).fit()
    phi = model.params["lag"]

    if phi >= 0:
        return float("inf")

    half_life = -np.log(2) / np.log(1 + phi)
    return max(half_life, 0.0)


# ---------------------------------------------------------------------------
# Cointegration scanner
# ---------------------------------------------------------------------------

def scan_cointegration(
    prices: pd.DataFrame,
    candidate_pairs: Optional[List[Tuple[str, str]]] = None,
    pvalue_threshold: float = DEFAULT_COINT_PVALUE,
    min_half_life: float = MIN_HALF_LIFE,
    max_half_life: float = MAX_HALF_LIFE,
    correlation_prefilter: float = 0.3,
) -> List[PairProfile]:
    """
    Scan all candidate pairs for cointegration.

    Performance optimization: pre-filters candidates by absolute correlation
    (default 0.3). Highly uncorrelated pairs are rarely cointegrated, so this
    skips the expensive Engle-Granger test on pairs unlikely to pass.
    0.3 is conservative enough to preserve weakly-correlated cointegrated pairs.

    Returns a list of PairProfile objects sorted by p-value (best first).
    """
    if candidate_pairs is None:
        candidate_pairs = get_ticker_pairs(available_tickers=list(prices.columns))

    available = set(prices.columns)
    candidate_pairs = [
        (a, b) for a, b in candidate_pairs
        if a in available and b in available
    ]
    total_candidates = len(candidate_pairs)

    # Pre-filter: compute correlation matrix once, skip pairs below threshold
    logger.info("Pre-filtering %d pairs by correlation >= %.2f...", total_candidates, correlation_prefilter)
    log_returns = np.log(prices / prices.shift(1)).dropna(how="all")
    corr_matrix = log_returns.corr()

    filtered_pairs = []
    for a, b in candidate_pairs:
        try:
            c = corr_matrix.loc[a, b]
            if pd.notna(c) and abs(c) >= correlation_prefilter:
                filtered_pairs.append((a, b))
        except KeyError:
            continue

    logger.info(
        "Correlation pre-filter: %d of %d pairs retained (%.0f%% skipped)",
        len(filtered_pairs), total_candidates,
        100 * (1 - len(filtered_pairs) / max(1, total_candidates)),
    )

    logger.info("Running Engle-Granger test on %d filtered pairs...", len(filtered_pairs))
    results: List[PairProfile] = []

    for ticker_a, ticker_b in filtered_pairs:
        y = prices[ticker_a]
        x = prices[ticker_b]

        pvalue, adf_stat, hedge_ratio = engle_granger_test(y, x)

        if pvalue <= pvalue_threshold:
            spread = compute_spread(y, x, hedge_ratio)
            half_life = estimate_half_life(spread)
        else:
            half_life = float("inf")

        is_valid = True
        rejection_reason = ""

        if pvalue > pvalue_threshold:
            is_valid = False
            rejection_reason = f"p-value {pvalue:.4f} > {pvalue_threshold}"
        elif half_life < min_half_life:
            is_valid = False
            rejection_reason = f"half-life {half_life:.1f}d < {min_half_life}d (too fast, likely noise)"
        elif half_life > max_half_life:
            is_valid = False
            rejection_reason = f"half-life {half_life:.1f}d > {max_half_life}d (too slow)"

        results.append(PairProfile(
            ticker_a=ticker_a,
            ticker_b=ticker_b,
            coint_pvalue=pvalue,
            adf_stat=adf_stat,
            hedge_ratio=hedge_ratio,
            half_life=half_life,
            is_valid=is_valid,
            rejection_reason=rejection_reason,
        ))

    results.sort(key=lambda p: (not p.is_valid, p.coint_pvalue))
    valid_count = sum(1 for p in results if p.is_valid)
    logger.info("Found %d valid cointegrated pairs out of %d tested.", valid_count, len(results))

    return results


# ---------------------------------------------------------------------------
# Rolling hedge ratio & Z-score
# ---------------------------------------------------------------------------

def rolling_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    window: int = DEFAULT_LOOKBACK,
) -> pd.Series:
    """
    Compute a rolling OLS hedge ratio (beta) over `window` days.
    """
    ratios = pd.Series(index=y.index, dtype=float)

    for i in range(window, len(y)):
        y_win = y.iloc[i - window : i]
        x_win = x.iloc[i - window : i]
        combined = pd.concat([y_win, x_win], axis=1).dropna()
        if len(combined) < 30:
            ratios.iloc[i] = np.nan
            continue
        x_const = add_constant(combined.iloc[:, 1])
        model = OLS(combined.iloc[:, 0], x_const).fit()
        ratios.iloc[i] = model.params.iloc[1]

    return ratios


def rolling_zscore(
    spread: pd.Series,
    window: int = DEFAULT_LOOKBACK,
) -> pd.Series:
    """Compute a rolling Z-score of the spread."""
    mean = spread.rolling(window, min_periods=30).mean()
    std = spread.rolling(window, min_periods=30).std()
    return (spread - mean) / std.replace(0, np.nan)


def compute_pair_analytics(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    window: int = DEFAULT_LOOKBACK,
) -> Dict[str, pd.Series]:
    """
    Compute all rolling analytics for a single pair.

    Returns dict with keys: hedge_ratio, spread, zscore, spread_mean, spread_std.
    """
    y = prices[ticker_a]
    x = prices[ticker_b]

    hr = rolling_hedge_ratio(y, x, window=window)
    spread = y - hr * x
    zscore = rolling_zscore(spread, window=window)
    spread_mean = spread.rolling(window, min_periods=30).mean()
    spread_std = spread.rolling(window, min_periods=30).std()

    return {
        "hedge_ratio": hr,
        "spread": spread,
        "zscore": zscore,
        "spread_mean": spread_mean,
        "spread_std": spread_std,
    }


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_signal(
    zscore: float,
    entry_threshold: float = DEFAULT_ZSCORE_ENTRY,
    exit_threshold: float = DEFAULT_ZSCORE_EXIT,
    stop_threshold: float = DEFAULT_ZSCORE_STOP,
    regime: Optional[str] = None,
    regime_confidence: Optional[float] = None,
    min_regime_confidence: float = 0.6,
) -> str:
    """
    Determine the trading signal for a pair based on its current Z-score
    and (optionally) the HMM market regime.

    Signal logic:
      - |z| > stop_threshold  → "STOP" (emergency exit / do not enter)
      - z < -entry_threshold  → "LONG_A_SHORT_B"  (spread is cheap)
      - z >  entry_threshold  → "SHORT_A_LONG_B"  (spread is rich)
      - |z| < exit_threshold  → "EXIT" (close existing position)
      - otherwise             → "FLAT" (no action)

    Regime interaction:
      The regime does NOT suppress signals at this layer. Entry signals
      always pass through if the Z-score warrants it. The risk_manager
      handles regime-based position scaling via regime_confidence_scalar(),
      which smoothly reduces sizing in trending regimes rather than
      applying a binary on/off filter. This prevents the system from
      going completely dark during persistent trending periods.
    """
    abs_z = abs(zscore)

    if abs_z > stop_threshold:
        return "STOP"

    if abs_z < exit_threshold:
        return "EXIT"

    if zscore < -entry_threshold:
        return "LONG_A_SHORT_B"
    elif zscore > entry_threshold:
        return "SHORT_A_LONG_B"

    return "FLAT"


def generate_pair_signals(
    prices: pd.DataFrame,
    valid_pairs: List[PairProfile],
    window: int = DEFAULT_LOOKBACK,
    regime: Optional[str] = None,
    regime_confidence: Optional[float] = None,
    entry_threshold: float = DEFAULT_ZSCORE_ENTRY,
    exit_threshold: float = DEFAULT_ZSCORE_EXIT,
    stop_threshold: float = DEFAULT_ZSCORE_STOP,
) -> List[PairSignal]:
    """
    Generate live signals for all valid cointegrated pairs.
    """
    signals: List[PairSignal] = []

    for pair in valid_pairs:
        if not pair.is_valid:
            continue

        try:
            analytics = compute_pair_analytics(
                prices, pair.ticker_a, pair.ticker_b, window=window,
            )
        except Exception as e:
            logger.error(
                "Failed to compute analytics for %s/%s: %s",
                pair.ticker_a, pair.ticker_b, e,
            )
            continue

        current_z = analytics["zscore"].dropna().iloc[-1] if not analytics["zscore"].dropna().empty else 0.0
        current_hr = analytics["hedge_ratio"].dropna().iloc[-1] if not analytics["hedge_ratio"].dropna().empty else pair.hedge_ratio
        current_mean = analytics["spread_mean"].dropna().iloc[-1] if not analytics["spread_mean"].dropna().empty else 0.0
        current_std = analytics["spread_std"].dropna().iloc[-1] if not analytics["spread_std"].dropna().empty else 1.0

        signal = generate_signal(
            zscore=current_z,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            stop_threshold=stop_threshold,
            regime=regime,
            regime_confidence=regime_confidence,
        )

        signals.append(PairSignal(
            ticker_a=pair.ticker_a,
            ticker_b=pair.ticker_b,
            zscore=current_z,
            hedge_ratio=current_hr,
            half_life=pair.half_life,
            signal=signal,
            regime=regime,
            regime_confidence=regime_confidence,
            coint_pvalue=pair.coint_pvalue,
            spread_mean=current_mean,
            spread_std=current_std,
        ))

    priority = {"STOP": 0, "LONG_A_SHORT_B": 1, "SHORT_A_LONG_B": 1, "EXIT": 2, "FLAT": 3}
    signals.sort(key=lambda s: (priority.get(s.signal, 99), -abs(s.zscore)))

    active = [s for s in signals if s.signal not in ("FLAT",)]
    logger.info("Generated %d signals (%d actionable).", len(signals), len(active))

    return signals


# ---------------------------------------------------------------------------
# Summary table (for dashboard)
# ---------------------------------------------------------------------------

def signals_to_dataframe(signals: List[PairSignal]) -> pd.DataFrame:
    """Convert a list of PairSignal objects to a display-ready DataFrame."""
    if not signals:
        return pd.DataFrame()

    rows = []
    for s in signals:
        rows.append({
            "Pair": f"{s.ticker_a} / {s.ticker_b}",
            "Signal": s.signal,
            "Z-Score": round(s.zscore, 3),
            "Hedge Ratio": round(s.hedge_ratio, 4),
            "Half-Life (days)": round(s.half_life, 1),
            "Coint p-value": round(s.coint_pvalue, 4),
            "Regime": s.regime or "N/A",
            "Regime Conf.": round(s.regime_confidence, 2) if s.regime_confidence else "N/A",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Full pipeline convenience
# ---------------------------------------------------------------------------

def run_pairs_pipeline(
    prices: pd.DataFrame,
    candidate_pairs: Optional[List[Tuple[str, str]]] = None,
    pvalue_threshold: float = DEFAULT_COINT_PVALUE,
    window: int = DEFAULT_LOOKBACK,
    regime: Optional[str] = None,
    regime_confidence: Optional[float] = None,
    entry_threshold: float = DEFAULT_ZSCORE_ENTRY,
    exit_threshold: float = DEFAULT_ZSCORE_EXIT,
    stop_threshold: float = DEFAULT_ZSCORE_STOP,
) -> Dict[str, Any]:
    """
    Run the complete pairs trading pipeline.

    Returns dict with:
      - "profiles": List[PairProfile] (all tested pairs)
      - "valid_pairs": List[PairProfile] (cointegrated only)
      - "signals": List[PairSignal]
      - "signals_df": pd.DataFrame (display-ready)
    """
    profiles = scan_cointegration(
        prices,
        candidate_pairs=candidate_pairs,
        pvalue_threshold=pvalue_threshold,
    )
    valid = [p for p in profiles if p.is_valid]

    signals = generate_pair_signals(
        prices,
        valid_pairs=valid,
        window=window,
        regime=regime,
        regime_confidence=regime_confidence,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        stop_threshold=stop_threshold,
    )

    return {
        "profiles": profiles,
        "valid_pairs": valid,
        "signals": signals,
        "signals_df": signals_to_dataframe(signals),
    }


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from data_loader import load_universe

    logger.info("Loading data...")
    data = load_universe(start="2023-01-01")
    prices = data["prices"]

    logger.info("Running pairs pipeline...")
    results = run_pairs_pipeline(prices)

    print(f"\nTested {len(results['profiles'])} pairs")
    print(f"Valid cointegrated pairs: {len(results['valid_pairs'])}")
    print(f"\nTop valid pairs:")
    for p in results["valid_pairs"][:10]:
        print(f"  {p.ticker_a:12s} / {p.ticker_b:12s}  p={p.coint_pvalue:.4f}  HL={p.half_life:.1f}d  HR={p.hedge_ratio:.4f}")

    print(f"\nActive signals:")
    active = [s for s in results["signals"] if s.signal != "FLAT"]
    for s in active[:10]:
        print(f"  {s.ticker_a:12s} / {s.ticker_b:12s}  {s.signal:20s}  z={s.zscore:+.3f}")

    if not active:
        print("  (no actionable signals at this time)")

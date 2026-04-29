"""
risk_manager.py – Position sizing, leverage management, and risk controls.

This is the most critical module in the system given 5x CFD leverage.
A single unchecked position can trigger a margin call that wipes the fund.

Two operating modes:
  - CONSERVATIVE: Tight stops, lower leverage utilisation, requires high HMM confidence
  - AGGRESSIVE:   Full leverage available, wider parameters, optimised for alpha

Core methodology: Volatility Target Sizing
  - Position size is scaled inversely to current realised volatility / ATR
  - Higher vol → smaller positions (protect capital)
  - Lower vol → larger positions (capture more when risk is manageable)

All outputs are notional weights suitable for the dashboard display.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import (
    AGGRESSIVE_MAX_LEVERAGE_UTIL,
    AGGRESSIVE_MIN_REGIME_CONF,
    AGGRESSIVE_STOP_MULT,
    AGGRESSIVE_VOL_TARGET,
    AGGRESSIVE_ZSCORE_ENTRY,
    AGGRESSIVE_ZSCORE_STOP,
    ANNUAL_TRADING_DAYS,
    CONSERVATIVE_MAX_LEVERAGE_UTIL,
    CONSERVATIVE_MIN_REGIME_CONF,
    CONSERVATIVE_STOP_MULT,
    CONSERVATIVE_VOL_TARGET,
    CONSERVATIVE_ZSCORE_ENTRY,
    CONSERVATIVE_ZSCORE_STOP,
    MARGIN_BUFFER_PCT,
    MAX_LEVERAGE,
    MAX_POSITION_PCT,
    MAX_TOTAL_EXPOSURE_PCT,
    MIN_STOP_PCT,
)
from hmm_engine import RegimeState
from pairs_engine import PairSignal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Risk mode enum and profile
# ---------------------------------------------------------------------------

class RiskMode(Enum):
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"


@dataclass
class RiskProfile:
    """Parameters for a given risk mode — pulled from config.py."""
    mode: RiskMode
    max_leverage_util: float
    vol_target: float
    stop_atr_mult: float
    min_regime_confidence: float
    zscore_entry: float
    zscore_stop: float


def get_risk_profile(mode: RiskMode) -> RiskProfile:
    """Build a RiskProfile from config.py constants based on the selected mode."""
    if mode == RiskMode.CONSERVATIVE:
        return RiskProfile(
            mode=mode,
            max_leverage_util=CONSERVATIVE_MAX_LEVERAGE_UTIL,
            vol_target=CONSERVATIVE_VOL_TARGET,
            stop_atr_mult=CONSERVATIVE_STOP_MULT,
            min_regime_confidence=CONSERVATIVE_MIN_REGIME_CONF,
            zscore_entry=CONSERVATIVE_ZSCORE_ENTRY,
            zscore_stop=CONSERVATIVE_ZSCORE_STOP,
        )
    return RiskProfile(
        mode=mode,
        max_leverage_util=AGGRESSIVE_MAX_LEVERAGE_UTIL,
        vol_target=AGGRESSIVE_VOL_TARGET,
        stop_atr_mult=AGGRESSIVE_STOP_MULT,
        min_regime_confidence=AGGRESSIVE_MIN_REGIME_CONF,
        zscore_entry=AGGRESSIVE_ZSCORE_ENTRY,
        zscore_stop=AGGRESSIVE_ZSCORE_STOP,
    )


# ---------------------------------------------------------------------------
# Position sizing output
# ---------------------------------------------------------------------------

@dataclass
class PositionSizing:
    """Recommended position sizing for a single pair trade."""
    ticker_a: str
    ticker_b: str
    signal: str

    # Current last-close prices (used for sizing and stop-loss calculation)
    price_a: float
    price_b: float

    # Notional weights as fraction of NAV (before leverage)
    weight_a: float                  # Positive = long, negative = short
    weight_b: float

    # Dollar amounts (if NAV is provided)
    notional_a: float
    notional_b: float

    # Risk metrics
    stop_loss_a: float               # Price-based stop for leg A
    stop_loss_b: float               # Price-based stop for leg B
    stop_triggered_a: bool           # True if last price already past stop
    stop_triggered_b: bool
    max_loss_pct: float              # Estimated max loss as % of NAV
    effective_leverage: float        # Gross notional / NAV for this pair
    vol_scalar: float                # The volatility scaling factor applied

    # Context
    risk_mode: str
    regime: str
    regime_confidence: float


@dataclass
class PortfolioRisk:
    """Aggregate portfolio-level risk summary."""
    total_gross_exposure: float      # Sum of |weight_a| + |weight_b| across all pairs
    total_net_exposure: float        # Sum of signed weights
    effective_leverage: float        # total_gross_exposure (accounts for leverage)
    margin_utilisation: float        # Estimated margin used as % of NAV
    margin_remaining: float          # 1.0 - margin_utilisation
    num_active_pairs: int
    risk_mode: str
    regime: str
    is_within_limits: bool
    warnings: List[str]


# ---------------------------------------------------------------------------
# Volatility-target position sizing
# ---------------------------------------------------------------------------

def compute_vol_scalar(
    current_vol: float,
    vol_target: float,
    floor: float = 0.1,
    cap: float = 3.0,
) -> float:
    """
    Compute the volatility scaling factor.

    vol_scalar = vol_target / current_vol

    This scales positions inversely to realised volatility:
      - High vol → scalar < 1 → smaller position
      - Low vol  → scalar > 1 → larger position

    Clamped to [floor, cap] to prevent degenerate sizing when vol
    is near zero or extremely high.
    """
    if current_vol <= 0 or np.isnan(current_vol):
        return floor

    scalar = vol_target / current_vol
    return float(np.clip(scalar, floor, cap))


def compute_pair_weight(
    hedge_ratio: float,
    vol_scalar: float,
    price_a: float,
    price_b: float,
    max_weight: float = MAX_POSITION_PCT,
) -> tuple:
    """
    Compute the notional weight (% of NAV) for each leg of a pairs trade.

    The OLS hedge ratio β is a SHARE-quantity ratio: "for every 1 share of A
    long, short β shares of B." We must convert it to a DOLLAR ratio using
    the current price ratio:

        dollar_ratio = β * (price_b / price_a)

    Without this conversion, pairs with mismatched prices (e.g. IGO at $7
    and PMT at $0.50) get wildly oversized short legs because the code
    would interpret β as a dollar multiplier.
    """
    abs_h = abs(hedge_ratio) if hedge_ratio != 0 else 1.0

    # Convert share-ratio to dollar-ratio at current prices
    if price_a > 0 and price_b > 0:
        dollar_ratio = abs_h * (price_b / price_a)
    else:
        dollar_ratio = abs_h

    # Calibrate base weight so gross exposure |w_a| + |w_b| ≈ max_weight
    base_weight = max_weight / (1.0 + dollar_ratio)

    weight_a = base_weight * vol_scalar
    # weight_b in dollars = weight_a * β * (price_b / price_a)
    # Sign of hedge_ratio is preserved (rarely negative for cointegrated pairs)
    sign = 1.0 if hedge_ratio >= 0 else -1.0
    weight_b = weight_a * dollar_ratio * sign

    # Final clamp: neither leg exceeds max_weight
    max_leg = max(abs(weight_a), abs(weight_b))
    if max_leg > max_weight:
        scale_down = max_weight / max_leg
        weight_a *= scale_down
        weight_b *= scale_down

    return weight_a, weight_b


# ---------------------------------------------------------------------------
# Stop-loss calculation
# ---------------------------------------------------------------------------

def compute_stop_loss(
    current_price: float,
    atr: float,
    atr_multiplier: float,
    is_long: bool,
    min_pct: float = MIN_STOP_PCT,
) -> float:
    """
    Compute an ATR-based stop-loss price with a percentage floor.

    Long position:  stop = price - max(ATR * multiplier, price * min_pct)
    Short position: stop = price + max(ATR * multiplier, price * min_pct)

    The min_pct floor ensures penny stocks don't get absurdly tight stops
    where the ATR in absolute dollars is a fraction of a cent.
    """
    atr_distance = atr * atr_multiplier
    min_distance = current_price * min_pct
    distance = max(atr_distance, min_distance)
    if is_long:
        return current_price - distance
    return current_price + distance


# ---------------------------------------------------------------------------
# Regime confidence adjustment
# ---------------------------------------------------------------------------

def regime_confidence_scalar(
    regime: Optional[RegimeState],
    profile: RiskProfile,
) -> float:
    """
    Scale position sizes based on regime confidence.

    - If no regime data → return 0.5 (cautious default)
    - If regime is favorable (CHOP) with high confidence → up to 1.0
    - If regime is unfavorable (BULL/BEAR) with high confidence → scale down
    - Low confidence in any regime → reduce sizing

    This creates a smooth gradient rather than a binary on/off filter.
    """
    if regime is None:
        return 0.5

    confidence = regime.confidence

    if regime.is_favorable_for_pairs:
        # CHOP regime: scale up with confidence
        # At min_confidence threshold → 0.6, at 100% → 1.0
        return float(np.clip(0.4 + 0.6 * confidence, 0.3, 1.0))
    else:
        # Trending regime: scale down with confidence
        # High confidence in trending → heavily reduce (pairs will bleed)
        if confidence >= profile.min_regime_confidence:
            # Strong trending signal: cut to 20-40% of normal sizing
            return float(np.clip(0.6 - 0.4 * confidence, 0.1, 0.4))
        else:
            # Weak trending signal: moderate reduction
            return float(np.clip(0.7 - 0.3 * confidence, 0.3, 0.7))


# ---------------------------------------------------------------------------
# Main sizing engine
# ---------------------------------------------------------------------------

def size_pair_position(
    signal: PairSignal,
    prices: pd.DataFrame,
    volatility: pd.DataFrame,
    atr: pd.DataFrame,
    profile: RiskProfile,
    regime: Optional[RegimeState] = None,
    nav: float = 1_000_000.0,
) -> Optional[PositionSizing]:
    """
    Compute full position sizing for a single pair signal.

    Returns None if the signal is FLAT or should be skipped.
    """
    if signal.signal == "FLAT":
        return None

    ticker_a = signal.ticker_a
    ticker_b = signal.ticker_b

    # Get current prices
    price_a = prices[ticker_a].dropna().iloc[-1] if ticker_a in prices.columns else None
    price_b = prices[ticker_b].dropna().iloc[-1] if ticker_b in prices.columns else None
    if price_a is None or price_b is None:
        logger.warning("Missing price data for %s/%s, skipping.", ticker_a, ticker_b)
        return None

    # Get current volatility for the pair (use average of both legs)
    vol_a = volatility[ticker_a].dropna().iloc[-1] if ticker_a in volatility.columns else 0.3
    vol_b = volatility[ticker_b].dropna().iloc[-1] if ticker_b in volatility.columns else 0.3
    pair_vol = (vol_a + vol_b) / 2.0

    # Get current ATR for stop-loss placement
    atr_a = atr[ticker_a].dropna().iloc[-1] if ticker_a in atr.columns else price_a * 0.02
    atr_b = atr[ticker_b].dropna().iloc[-1] if ticker_b in atr.columns else price_b * 0.02

    # Step 1: Volatility-target scalar
    vol_scalar = compute_vol_scalar(pair_vol, profile.vol_target)

    # Step 2: Regime confidence adjustment
    regime_scalar = regime_confidence_scalar(regime, profile)
    adjusted_scalar = vol_scalar * regime_scalar

    # Step 3: Compute weights (price-aware so dollar-ratio is correct)
    weight_a, weight_b = compute_pair_weight(
        signal.hedge_ratio, adjusted_scalar, price_a, price_b,
        max_weight=MAX_POSITION_PCT,
    )

    # Step 4: Apply signal direction
    if signal.signal == "LONG_A_SHORT_B":
        # Long A, Short B
        weight_a = abs(weight_a)
        weight_b = -abs(weight_b)
        long_a = True
    elif signal.signal == "SHORT_A_LONG_B":
        # Short A, Long B
        weight_a = -abs(weight_a)
        weight_b = abs(weight_b)
        long_a = False
    elif signal.signal in ("EXIT", "STOP"):
        # For exit/stop signals, report zero weights (close position)
        weight_a = 0.0
        weight_b = 0.0
        long_a = True  # Doesn't matter for exits
    else:
        return None

    # Step 5: Notional amounts
    notional_a = weight_a * nav
    notional_b = weight_b * nav

    # Step 6: Stop-loss levels
    stop_a = compute_stop_loss(price_a, atr_a, profile.stop_atr_mult, is_long=(weight_a > 0))
    stop_b = compute_stop_loss(price_b, atr_b, profile.stop_atr_mult, is_long=(weight_b > 0))

    # Check if the current price has already crossed the stop (dead-on-arrival)
    # For a long: stop is below entry, so triggered if price <= stop
    # For a short: stop is above entry, so triggered if price >= stop
    if weight_a > 0:
        stop_triggered_a = price_a <= stop_a
    elif weight_a < 0:
        stop_triggered_a = price_a >= stop_a
    else:
        stop_triggered_a = False

    if weight_b > 0:
        stop_triggered_b = price_b <= stop_b
    elif weight_b < 0:
        stop_triggered_b = price_b >= stop_b
    else:
        stop_triggered_b = False

    # Step 7: Estimated max loss
    loss_a = abs(price_a - stop_a) / price_a * abs(weight_a)
    loss_b = abs(price_b - stop_b) / price_b * abs(weight_b)
    max_loss_pct = loss_a + loss_b

    # Step 8: Effective leverage for this pair
    gross = abs(weight_a) + abs(weight_b)
    effective_lev = gross * MAX_LEVERAGE

    regime_label = regime.label if regime else "N/A"
    regime_conf = regime.confidence if regime else 0.0

    return PositionSizing(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        signal=signal.signal,
        price_a=round(price_a, 4),
        price_b=round(price_b, 4),
        weight_a=round(weight_a, 6),
        weight_b=round(weight_b, 6),
        notional_a=round(notional_a, 2),
        notional_b=round(notional_b, 2),
        stop_loss_a=round(stop_a, 4),
        stop_loss_b=round(stop_b, 4),
        stop_triggered_a=stop_triggered_a,
        stop_triggered_b=stop_triggered_b,
        max_loss_pct=round(max_loss_pct, 6),
        effective_leverage=round(effective_lev, 2),
        vol_scalar=round(adjusted_scalar, 4),
        risk_mode=profile.mode.value,
        regime=regime_label,
        regime_confidence=round(regime_conf, 4),
    )


# ---------------------------------------------------------------------------
# Portfolio-level risk aggregation
# ---------------------------------------------------------------------------

def compute_portfolio_risk(
    positions: List[PositionSizing],
    profile: RiskProfile,
    regime: Optional[RegimeState] = None,
) -> PortfolioRisk:
    """
    Aggregate all active positions into a portfolio-level risk summary.

    This is the key output for the dashboard's margin utilisation display.
    """
    warnings: List[str] = []

    if not positions:
        regime_label = regime.label if regime else "N/A"
        return PortfolioRisk(
            total_gross_exposure=0.0,
            total_net_exposure=0.0,
            effective_leverage=0.0,
            margin_utilisation=0.0,
            margin_remaining=1.0,
            num_active_pairs=0,
            risk_mode=profile.mode.value,
            regime=regime_label,
            is_within_limits=True,
            warnings=[],
        )

    # Aggregate weights
    total_gross = sum(abs(p.weight_a) + abs(p.weight_b) for p in positions)
    total_net = sum(p.weight_a + p.weight_b for p in positions)
    active_count = sum(1 for p in positions if p.signal not in ("EXIT", "STOP", "FLAT"))

    # Effective leverage = gross exposure * broker leverage
    effective_leverage = total_gross * MAX_LEVERAGE

    # Margin utilisation estimate
    # With 5x leverage, margin requirement is ~20% of notional
    margin_rate = 1.0 / MAX_LEVERAGE
    margin_utilisation = total_gross * margin_rate

    # Remaining margin (with safety buffer)
    margin_remaining = 1.0 - margin_utilisation

    # Limit checks
    is_within_limits = True

    if total_gross > MAX_TOTAL_EXPOSURE_PCT:
        is_within_limits = False
        warnings.append(
            f"Gross exposure {total_gross:.1%} exceeds limit {MAX_TOTAL_EXPOSURE_PCT:.1%}"
        )

    if effective_leverage > profile.max_leverage_util:
        is_within_limits = False
        warnings.append(
            f"Effective leverage {effective_leverage:.1f}x exceeds "
            f"{profile.mode.value} limit {profile.max_leverage_util:.1f}x"
        )

    if margin_remaining < MARGIN_BUFFER_PCT:
        is_within_limits = False
        warnings.append(
            f"Margin remaining {margin_remaining:.1%} below "
            f"safety buffer {MARGIN_BUFFER_PCT:.1%} — MARGIN CALL RISK"
        )

    # Regime-based warnings
    if regime is not None and not regime.is_favorable_for_pairs:
        if regime.confidence > profile.min_regime_confidence:
            warnings.append(
                f"Trending regime ({regime.label}) with {regime.confidence:.0%} confidence — "
                f"pairs positions carry elevated risk"
            )

    regime_label = regime.label if regime else "N/A"

    return PortfolioRisk(
        total_gross_exposure=round(total_gross, 4),
        total_net_exposure=round(total_net, 4),
        effective_leverage=round(effective_leverage, 2),
        margin_utilisation=round(margin_utilisation, 4),
        margin_remaining=round(margin_remaining, 4),
        num_active_pairs=active_count,
        risk_mode=profile.mode.value,
        regime=regime_label,
        is_within_limits=is_within_limits,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Full risk pipeline
# ---------------------------------------------------------------------------

def run_risk_pipeline(
    signals: List[PairSignal],
    prices: pd.DataFrame,
    volatility: pd.DataFrame,
    atr: pd.DataFrame,
    mode: RiskMode = RiskMode.CONSERVATIVE,
    regime: Optional[RegimeState] = None,
    nav: float = 1_000_000.0,
) -> Dict:
    """
    Run the complete risk management pipeline:
      1. Get risk profile for the selected mode
      2. Size each pair position
      3. Aggregate portfolio risk
      4. Return structured results

    Returns dict with:
      - "profile": RiskProfile
      - "positions": List[PositionSizing]
      - "portfolio_risk": PortfolioRisk
      - "positions_df": pd.DataFrame (display-ready)
      - "thresholds": dict of mode-adjusted Z-score thresholds
    """
    profile = get_risk_profile(mode)

    positions = []
    for signal in signals:
        sizing = size_pair_position(
            signal, prices, volatility, atr,
            profile=profile, regime=regime, nav=nav,
        )
        if sizing is not None:
            positions.append(sizing)

    portfolio_risk = compute_portfolio_risk(positions, profile, regime)

    # Mode-adjusted thresholds for the pairs engine
    thresholds = {
        "zscore_entry": profile.zscore_entry,
        "zscore_stop": profile.zscore_stop,
        "min_regime_confidence": profile.min_regime_confidence,
    }

    return {
        "profile": profile,
        "positions": positions,
        "portfolio_risk": portfolio_risk,
        "positions_df": positions_to_dataframe(positions),
        "thresholds": thresholds,
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def positions_to_dataframe(positions: List[PositionSizing]) -> pd.DataFrame:
    """Convert position sizing results to a display-ready DataFrame."""
    if not positions:
        return pd.DataFrame()

    rows = []
    for p in positions:
        rows.append({
            "Pair": f"{p.ticker_a} / {p.ticker_b}",
            "Signal": p.signal,
            "Weight A": f"{p.weight_a:+.2%}",
            "Weight B": f"{p.weight_b:+.2%}",
            "Notional A": f"${p.notional_a:,.0f}",
            "Notional B": f"${p.notional_b:,.0f}",
            "Stop A": f"${p.stop_loss_a:,.2f}",
            "Stop B": f"${p.stop_loss_b:,.2f}",
            "Max Loss": f"{p.max_loss_pct:.2%}",
            "Eff. Leverage": f"{p.effective_leverage:.1f}x",
            "Vol Scalar": f"{p.vol_scalar:.2f}",
            "Mode": p.risk_mode,
        })

    return pd.DataFrame(rows)


def portfolio_risk_summary(risk: PortfolioRisk) -> Dict[str, str]:
    """Format portfolio risk for dashboard display."""
    return {
        "Gross Exposure": f"{risk.total_gross_exposure:.1%}",
        "Net Exposure": f"{risk.total_net_exposure:+.1%}",
        "Effective Leverage": f"{risk.effective_leverage:.1f}x / {MAX_LEVERAGE:.0f}x",
        "Margin Used": f"{risk.margin_utilisation:.1%}",
        "Margin Remaining": f"{risk.margin_remaining:.1%}",
        "Active Pairs": str(risk.num_active_pairs),
        "Risk Mode": risk.risk_mode.upper(),
        "Regime": risk.regime,
        "Within Limits": "YES" if risk.is_within_limits else "NO",
    }


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from data_loader import load_universe
    from hmm_engine import detect_regimes, get_current_regime
    from pairs_engine import run_pairs_pipeline

    logger.info("Loading data...")
    data = load_universe(start="2023-01-01")

    logger.info("Detecting regime...")
    history = detect_regimes(data["returns"], data["volatility"])
    regime = get_current_regime(history)

    logger.info("Running pairs pipeline...")
    pairs_result = run_pairs_pipeline(
        data["prices"], regime=regime.label, regime_confidence=regime.confidence,
    )

    for mode in [RiskMode.CONSERVATIVE, RiskMode.AGGRESSIVE]:
        logger.info("Running risk pipeline (%s)...", mode.value)
        risk_result = run_risk_pipeline(
            signals=pairs_result["signals"],
            prices=data["prices"],
            volatility=data["volatility"],
            atr=data["atr"],
            mode=mode,
            regime=regime,
            nav=1_000_000,
        )

        print(f"\n{'='*60}")
        print(f"RISK MODE: {mode.value.upper()}")
        print(f"{'='*60}")

        summary = portfolio_risk_summary(risk_result["portfolio_risk"])
        for k, v in summary.items():
            print(f"  {k:20s}: {v}")

        if risk_result["portfolio_risk"].warnings:
            print(f"\n  WARNINGS:")
            for w in risk_result["portfolio_risk"].warnings:
                print(f"    - {w}")

        active = [p for p in risk_result["positions"] if p.signal not in ("EXIT", "STOP")]
        if active:
            print(f"\n  Top positions:")
            for p in active[:5]:
                print(
                    f"    {p.ticker_a:10s}/{p.ticker_b:10s}  "
                    f"{p.signal:20s}  "
                    f"wA={p.weight_a:+.2%}  wB={p.weight_b:+.2%}  "
                    f"lev={p.effective_leverage:.1f}x"
                )
        else:
            print(f"\n  (no active positions)")

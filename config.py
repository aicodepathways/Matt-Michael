"""
config.py – Central configuration for the trading system.

All ticker mappings, thresholds, and tunable parameters live here.
To swap broker-specific CFD instruments, edit the dictionaries below.
No need to touch core logic in any other module.
"""

# ---------------------------------------------------------------------------
# Commodity Ticker Mappings
# ---------------------------------------------------------------------------
# Key   = Human-readable commodity name (used in UI labels)
# Value = yfinance-compatible ticker symbol
#
# IMPORTANT: When your broker offers specific CFD instruments, swap the
# ticker values here. The rest of the system references these by key.

COMMODITIES = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Copper": "HG=F",
    "Lithium": "LIT",          # Global X Lithium & Battery Tech ETF
    "Uranium": "URA",          # Global X Uranium ETF as proxy
    "Coal": "GLNCY",          # Glencore OTC as coal proxy
    "Nickel": "NICK.L",       # WisdomTree Nickel ETC (LSE)
    "Iron Ore": "BHP",        # BHP as iron ore proxy (most liquid)
    "Rare Earths": "REMX",    # VanEck Rare Earth/Strategic Metals ETF
}

# ---------------------------------------------------------------------------
# ASX Mining Stocks (append .AX automatically in data_loader)
# ---------------------------------------------------------------------------

ASX_STOCKS = [
    # Gold / Precious
    "ALK", "BGL", "CMM", "EMR", "EVN", "GMD", "GGP", "NST", "OBM",
    "PNR", "PRU", "RMS", "RRL", "RSG", "VAU", "WAF", "WGX",
    # Lithium
    "LTR", "PLS", "IGO", "MIN",
    # Rare Earths
    "LYC", "WA1", "AIS",
    # Copper / Base Metals
    "FFM", "SFR", "DVP", "A1M", "29M",
    # Coal
    "CRN", "SMR", "WHC", "NHC", "YAL",
    # Nickel
    "ASL", "BML", "POL",
    # Uranium
    "BMN", "BOE", "DYL", "LOT", "NXG", "PDN", "PEN",
]

GLOBAL_STOCKS = ["CCJ"]  # Cameco (uranium)

# ---------------------------------------------------------------------------
# Derived ticker lists (computed from above — do not edit directly)
# ---------------------------------------------------------------------------

ASX_TICKERS = [f"{t}.AX" for t in ASX_STOCKS]
GLOBAL_TICKERS = GLOBAL_STOCKS
COMMODITY_TICKERS = list(COMMODITIES.values())
ALL_TICKERS = COMMODITY_TICKERS + ASX_TICKERS + GLOBAL_TICKERS

# Reverse lookup: ticker symbol → human-readable display name
# Used throughout the UI so clients see "Gold" instead of "GC=F"
TICKER_DISPLAY_NAMES = {v: k for k, v in COMMODITIES.items()}
TICKER_DISPLAY_NAMES.update({
    "CCJ": "Cameco (Uranium)",
})
# ASX stocks: strip the .AX suffix for display
for _t in ASX_STOCKS:
    TICKER_DISPLAY_NAMES[f"{_t}.AX"] = _t


def display_name(ticker: str) -> str:
    """Return a human-readable name for a ticker. Falls back to the raw ticker."""
    return TICKER_DISPLAY_NAMES.get(ticker, ticker)


def get_effective_tickers() -> list:
    """
    Return the full ticker list including any user-added tickers and
    excluding any user-removed tickers from the Ticker Management page.

    Reads ticker_overrides.json if it exists, merges with base lists.
    """
    import json
    import os

    overrides_path = os.path.join(os.path.dirname(__file__), "ticker_overrides.json")
    if not os.path.exists(overrides_path):
        return ALL_TICKERS

    try:
        with open(overrides_path) as f:
            overrides = json.load(f)
    except (json.JSONDecodeError, IOError):
        return ALL_TICKERS

    tickers = list(ALL_TICKERS)

    # Add custom tickers
    for t in overrides.get("added_asx", []):
        if t not in tickers:
            tickers.append(t)
            # Also register display name
            TICKER_DISPLAY_NAMES[t] = t.replace(".AX", "")
    for t in overrides.get("added_global", []):
        if t not in tickers:
            tickers.append(t)
    for name, t in overrides.get("added_commodities", {}).items():
        if t not in tickers:
            tickers.append(t)
            TICKER_DISPLAY_NAMES[t] = name

    # Remove tickers
    removed = set(overrides.get("removed", []))
    tickers = [t for t in tickers if t not in removed]

    return tickers

# ---------------------------------------------------------------------------
# Data Pipeline Settings
# ---------------------------------------------------------------------------

FORWARD_FILL_LIMIT = 3          # Max business days to forward-fill (tightened for 5x leverage)
DEFAULT_START_DATE = "2018-01-01"

# ---------------------------------------------------------------------------
# Pairs Engine Defaults
# ---------------------------------------------------------------------------

DEFAULT_COINT_PVALUE = 0.05     # Engle-Granger significance threshold
DEFAULT_LOOKBACK = 252           # Rolling window (trading days)
DEFAULT_ZSCORE_ENTRY = 2.0       # Open position when |z| exceeds this
DEFAULT_ZSCORE_EXIT = 0.5        # Close position when |z| falls below this
DEFAULT_ZSCORE_STOP = 3.5        # Emergency stop-loss threshold
MIN_HALF_LIFE = 5                # Reject pairs reverting faster than this (noise)
MAX_HALF_LIFE = 120              # Reject pairs reverting slower than this

# ---------------------------------------------------------------------------
# HMM Regime Detection
# ---------------------------------------------------------------------------

HMM_N_REGIMES = 3               # Number of hidden states (Bull / Bear / Chop)
HMM_TRAINING_WINDOW = 504       # ~2 years of training data
HMM_N_ITER = 200                # EM iterations for convergence
HMM_COV_TYPE = "full"           # Covariance type for Gaussian HMM
HMM_RANDOM_SEED = 42            # Reproducibility

# Regime label mapping (assigned post-training by sorting state means)
REGIME_LABELS = {
    0: "BEAR",
    1: "CHOP",       # Mean-reverting / range-bound — ideal for pairs
    2: "BULL",
}

# ---------------------------------------------------------------------------
# Risk Management
# ---------------------------------------------------------------------------

MAX_LEVERAGE = 5.0               # Broker leverage cap
ANNUAL_TRADING_DAYS = 252

# Conservative mode
CONSERVATIVE_MAX_LEVERAGE_UTIL = 2.5    # Max effective leverage
CONSERVATIVE_VOL_TARGET = 0.10          # 10% annualised vol target
CONSERVATIVE_STOP_MULT = 1.5            # ATR multiplier for stop-loss
CONSERVATIVE_MIN_REGIME_CONF = 0.75     # Require high HMM confidence
CONSERVATIVE_ZSCORE_ENTRY = 2.5         # Tighter entry
CONSERVATIVE_ZSCORE_STOP = 3.0          # Tighter stop

# Aggressive mode
AGGRESSIVE_MAX_LEVERAGE_UTIL = 5.0      # Full leverage available
AGGRESSIVE_VOL_TARGET = 0.20            # 20% annualised vol target
AGGRESSIVE_STOP_MULT = 2.5              # Wider ATR stop
AGGRESSIVE_MIN_REGIME_CONF = 0.55       # Lower confidence threshold OK
AGGRESSIVE_ZSCORE_ENTRY = 1.8           # Wider entry (more signals)
AGGRESSIVE_ZSCORE_STOP = 4.0            # Wider stop

# Position sizing
MAX_POSITION_PCT = 0.20                 # Max 20% of NAV in a single pair
MAX_TOTAL_EXPOSURE_PCT = 1.0            # Max 100% NAV gross exposure (before leverage)
MARGIN_BUFFER_PCT = 0.15                # Keep 15% margin buffer vs maintenance

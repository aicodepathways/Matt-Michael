"""
data_loader.py – Institutional-grade data ingestion pipeline for commodity/mining hedge fund.

Handles:
  - yfinance downloads (uses yfinance's built-in curl_cffi session)
  - Strict timezone alignment between global commodities and ASX equities
  - Forward-filling across asynchronous trading calendars
  - Return calculation and basic data quality checks
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    ALL_TICKERS,
    ASX_TICKERS,
    COMMODITY_TICKERS,
    DEFAULT_START_DATE,
    FORWARD_FILL_LIMIT,
    GLOBAL_TICKERS,
    get_effective_tickers,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-series timezone normalisation
# ---------------------------------------------------------------------------

def _normalize_series(series: pd.Series) -> pd.Series:
    """
    Convert a single ticker's Series to tz-naive, date-only index.

    Each exchange returns its own timezone (America/New_York, Australia/Sydney,
    Europe/London, etc.). We strip the tz → date-only so all tickers align
    on the same calendar axis BEFORE being combined into a DataFrame.
    """
    idx = series.index
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    idx = pd.to_datetime(idx).normalize()
    series.index = idx
    # Drop duplicate dates (can happen at DST boundaries)
    series = series[~series.index.duplicated(keep="last")]
    return series


# ---------------------------------------------------------------------------
# Core download function
# ---------------------------------------------------------------------------

def download_prices(
    tickers: Optional[List[str]] = None,
    start: str = DEFAULT_START_DATE,
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download adjusted close prices for all tickers in the universe.

    yfinance >= 0.2.31 manages its own curl_cffi session with built-in
    retry and rate-limit handling. We do not pass a custom session.

    Each ticker's timestamps are normalised to tz-naive dates BEFORE
    being combined, ensuring proper alignment across exchanges.

    Returns a DataFrame indexed by **date** (timezone-naive)
    with one column per ticker.
    """
    if tickers is None:
        tickers = get_effective_tickers()
    if end is None:
        end = datetime.utcnow().strftime("%Y-%m-%d")

    logger.info("Downloading %d tickers from %s to %s", len(tickers), start, end)

    frames: Dict[str, pd.Series] = {}
    failed: List[str] = []

    for ticker in tickers:
        try:
            obj = yf.Ticker(ticker)
            hist = obj.history(start=start, end=end, interval=interval, auto_adjust=True)
            if hist.empty:
                logger.warning("No data returned for %s", ticker)
                failed.append(ticker)
                continue
            # Normalise timezone BEFORE combining into DataFrame
            frames[ticker] = _normalize_series(hist["Close"])
        except Exception as e:
            logger.error("Failed to download %s: %s", ticker, e)
            failed.append(ticker)
        # Small delay between requests to avoid rate-limiting
        time.sleep(0.15)

    if failed:
        logger.warning("Failed tickers (%d): %s", len(failed), failed)

    if not frames:
        raise RuntimeError("No data downloaded for any ticker.")

    raw = pd.DataFrame(frames)
    aligned = _align_timezones(raw)
    return aligned


# ---------------------------------------------------------------------------
# Timezone alignment & forward-fill
# ---------------------------------------------------------------------------

def _align_timezones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final alignment pass after per-series normalisation.

    At this point each column is already tz-naive and date-normalised.
    This function:
      1. Sorts the index
      2. Forward-fills gaps up to FORWARD_FILL_LIMIT days
      3. Drops columns with excessive missing data (>50% NaN)
      4. Trims leading rows so that at least 80% of tickers have data
         (avoids a single late-starting ticker from killing the whole history)
    """
    df = df.sort_index()

    # Remove any remaining duplicate dates
    df = df[~df.index.duplicated(keep="last")]

    # Forward-fill across weekends / holidays (capped for leverage safety)
    df = df.ffill(limit=FORWARD_FILL_LIMIT)

    # Drop tickers that are mostly NaN (delisted, broken feed, etc.)
    nan_pct = df.isna().sum() / len(df)
    bad_tickers = nan_pct[nan_pct > 0.50].index.tolist()
    if bad_tickers:
        logger.warning("Dropping %d tickers with >50%% NaN: %s", len(bad_tickers), bad_tickers)
        df = df.drop(columns=bad_tickers)

    # Trim leading rows: require at least 80% of tickers to have data
    min_available = int(len(df.columns) * 0.80)
    valid_counts = df.notna().sum(axis=1)
    mask = valid_counts >= min_available
    if mask.any():
        first_valid = mask.idxmax()
        df = df.loc[first_valid:]

    logger.info(
        "Aligned data: %d rows x %d columns, %s → %s (ffill limit=%d)",
        len(df), len(df.columns),
        df.index.min().date(), df.index.max().date(),
        FORWARD_FILL_LIMIT,
    )
    return df


# ---------------------------------------------------------------------------
# Derived data helpers
# ---------------------------------------------------------------------------

def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Compute daily returns (log or simple)."""
    if method == "log":
        return np.log(prices / prices.shift(1)).dropna(how="all")
    return prices.pct_change().dropna(how="all")


def compute_volatility(
    returns: pd.DataFrame,
    window: int = 21,
) -> pd.DataFrame:
    """Rolling realised volatility (annualised, 252 trading days)."""
    return returns.rolling(window).std() * np.sqrt(252)


def compute_atr(
    prices: pd.DataFrame,
    window: int = 14,
) -> pd.DataFrame:
    """
    Simplified ATR using daily close-to-close range as a proxy.

    For CFDs where we only have close prices, this is a reasonable
    volatility-scaling metric.
    """
    daily_range = prices.diff().abs()
    return daily_range.rolling(window).mean()


# ---------------------------------------------------------------------------
# Convenience: full pipeline
# ---------------------------------------------------------------------------

def load_universe(
    start: str = DEFAULT_START_DATE,
    end: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run the full data pipeline and return a dict with:
      - "prices":      timezone-aligned close prices
      - "returns":     log returns
      - "volatility":  21-day rolling realised vol
      - "atr":         14-day ATR proxy
    """
    prices = download_prices(start=start, end=end)
    returns = compute_returns(prices)
    volatility = compute_volatility(returns)
    atr = compute_atr(prices)

    return {
        "prices": prices,
        "returns": returns,
        "volatility": volatility,
        "atr": atr,
    }


# ---------------------------------------------------------------------------
# Ticker classification helpers (useful downstream)
# ---------------------------------------------------------------------------

def classify_ticker(ticker: str) -> str:
    """Return 'commodity', 'asx', or 'global' for a given ticker."""
    if ticker in COMMODITY_TICKERS:
        return "commodity"
    if ticker in ASX_TICKERS:
        return "asx"
    if ticker in GLOBAL_TICKERS:
        return "global"
    return "unknown"


def get_ticker_pairs(available_tickers: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """
    Generate candidate pair combinations for cointegration testing.

    If available_tickers is provided (e.g. from downloaded prices columns),
    uses those to classify tickers dynamically. This ensures custom tickers
    added via the Ticker Management page are included in the scan.

    Rules:
      - Stock vs Stock
      - Stock vs Commodity
      - Global vs Commodity
    We do NOT pair two commodities (no edge there for this fund).
    """
    from itertools import combinations
    import json
    import os

    # Build effective ticker lists including overrides
    asx = list(ASX_TICKERS)
    comms = list(COMMODITY_TICKERS)
    globs = list(GLOBAL_TICKERS)

    overrides_path = os.path.join(os.path.dirname(__file__), "ticker_overrides.json")
    if os.path.exists(overrides_path):
        try:
            with open(overrides_path) as f:
                overrides = json.load(f)
            for t in overrides.get("added_asx", []):
                if t not in asx:
                    asx.append(t)
            for t in overrides.get("added_global", []):
                if t not in globs:
                    globs.append(t)
            for name, t in overrides.get("added_commodities", {}).items():
                if t not in comms:
                    comms.append(t)
            removed = set(overrides.get("removed", []))
            asx = [t for t in asx if t not in removed]
            comms = [t for t in comms if t not in removed]
            globs = [t for t in globs if t not in removed]
        except (json.JSONDecodeError, IOError):
            pass

    # If we know what was actually downloaded, filter to those
    if available_tickers is not None:
        avail = set(available_tickers)
        asx = [t for t in asx if t in avail]
        comms = [t for t in comms if t in avail]
        globs = [t for t in globs if t in avail]

    pairs = []

    # Stock vs Stock (ASX + Global combined)
    all_stocks = asx + globs
    pairs.extend(combinations(all_stocks, 2))

    # Stock vs Commodity
    for stock in all_stocks:
        for comm in comms:
            pairs.append((stock, comm))

    return pairs


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data = load_universe(start="2023-01-01")
    print(f"\nPrices shape: {data['prices'].shape}")
    print(f"Date range:   {data['prices'].index[0].date()} → {data['prices'].index[-1].date()}")
    print(f"Tickers:      {list(data['prices'].columns[:10])}...")
    print(f"\nSample returns:\n{data['returns'].tail(3)}")

"""
signal_history.py – Persists daily signal snapshots so the dashboard can show
what changed since yesterday and a rolling history of past signals.

Stores data as a JSON file (signal_history.json). Each entry is keyed by date
and contains the list of actionable signals for that day.
"""

import json
import os
from datetime import date, datetime
from typing import Dict, List, Optional

from config import display_name

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "signal_history.json")
MAX_HISTORY_DAYS = 60  # Keep up to 60 days of history


def _load_history() -> Dict:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_history(history: Dict):
    # Trim to last MAX_HISTORY_DAYS entries
    if len(history) > MAX_HISTORY_DAYS:
        sorted_dates = sorted(history.keys())
        for old_date in sorted_dates[:-MAX_HISTORY_DAYS]:
            del history[old_date]
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def save_today_signals(positions: list, data_date: str):
    """
    Save today's signals to history. Called once per data refresh.

    positions: list of PositionSizing objects from risk_manager
    data_date: the date string of the latest price data (e.g. "2026-04-02")
    """
    history = _load_history()

    # Don't overwrite if we already logged this date
    if data_date in history:
        return

    entries = []
    for p in positions:
        if p.signal in ("LONG_A_SHORT_B", "SHORT_A_LONG_B"):
            if p.signal == "LONG_A_SHORT_B":
                long_tk, short_tk = p.ticker_a, p.ticker_b
            else:
                long_tk, short_tk = p.ticker_b, p.ticker_a
            entries.append({
                "type": "ENTRY",
                "long": p.ticker_a,
                "short": p.ticker_b,
                "long_display": display_name(long_tk),
                "short_display": display_name(short_tk),
                "signal": p.signal,
                "weight_a": p.weight_a,
                "weight_b": p.weight_b,
                "zscore": round(
                    (p.weight_a + p.weight_b) * 100, 3  # Placeholder — overridden below
                ),
            })
        elif p.signal in ("EXIT", "STOP"):
            entries.append({
                "type": p.signal,
                "ticker_a": p.ticker_a,
                "ticker_b": p.ticker_b,
                "a_display": display_name(p.ticker_a),
                "b_display": display_name(p.ticker_b),
                "signal": p.signal,
            })

    history[data_date] = entries
    _save_history(history)


def save_today_signals_full(positions: list, signals: list, data_date: str):
    """
    Save today's signals with Z-scores from the pairs engine signals list.
    """
    history = _load_history()

    if data_date in history:
        return

    # Build a Z-score lookup from pairs engine signals
    zscore_lookup = {}
    for s in signals:
        zscore_lookup[(s.ticker_a, s.ticker_b)] = s.zscore

    entries = []
    for p in positions:
        if p.signal in ("LONG_A_SHORT_B", "SHORT_A_LONG_B"):
            if p.signal == "LONG_A_SHORT_B":
                long_tk, short_tk = p.ticker_a, p.ticker_b
            else:
                long_tk, short_tk = p.ticker_b, p.ticker_a

            zs = zscore_lookup.get((p.ticker_a, p.ticker_b), 0.0)

            entries.append({
                "type": "ENTRY",
                "long": long_tk,
                "short": short_tk,
                "long_display": display_name(long_tk),
                "short_display": display_name(short_tk),
                "signal": p.signal,
                "weight_a": p.weight_a,
                "weight_b": p.weight_b,
                "zscore": round(zs, 3),
            })
        elif p.signal in ("EXIT", "STOP"):
            entries.append({
                "type": p.signal,
                "ticker_a": p.ticker_a,
                "ticker_b": p.ticker_b,
                "a_display": display_name(p.ticker_a),
                "b_display": display_name(p.ticker_b),
                "signal": p.signal,
            })

    history[data_date] = entries
    _save_history(history)


def get_history() -> Dict:
    """Return the full signal history dict, keyed by date string."""
    return _load_history()


def get_yesterday_signals(today: str) -> Optional[List]:
    """Return yesterday's signals, or None if not available."""
    history = _load_history()
    dates = sorted(history.keys())
    if today in dates:
        idx = dates.index(today)
        if idx > 0:
            return history[dates[idx - 1]]
    # If today isn't logged yet, return the most recent entry
    if dates:
        return history[dates[-1]]
    return None


def get_previous_date(today: str) -> Optional[str]:
    """Return the date string of the previous logged day."""
    history = _load_history()
    dates = sorted(history.keys())
    if today in dates:
        idx = dates.index(today)
        if idx > 0:
            return dates[idx - 1]
    if dates:
        return dates[-1]
    return None


def compute_changes(today_positions: list, today_signals: list, today_date: str) -> Dict:
    """
    Compare today's signals to yesterday's and return a change summary.

    Returns dict with:
      - "new_entries": pairs that are entries today but weren't yesterday
      - "closed": pairs that were entries yesterday but are now EXIT/STOP/gone
      - "continued": pairs that remain as entries
      - "previous_date": the date we're comparing against
    """
    yesterday = get_yesterday_signals(today_date)
    prev_date = get_previous_date(today_date)

    # Helper to normalise pair identity — same pair regardless of long/short order
    # or ticker_a/ticker_b convention. Sort the pair tuple alphabetically.
    def _norm(a: str, b: str) -> tuple:
        return tuple(sorted((a, b)))

    # Today's entry pairs (normalised)
    today_entry_pairs = set()
    today_lookup = {}  # normalised → (long_display, short_display) for output
    for p in today_positions:
        if p.signal in ("LONG_A_SHORT_B", "SHORT_A_LONG_B"):
            key = _norm(p.ticker_a, p.ticker_b)
            today_entry_pairs.add(key)
            if p.signal == "LONG_A_SHORT_B":
                today_lookup[key] = (p.ticker_a, p.ticker_b)
            else:
                today_lookup[key] = (p.ticker_b, p.ticker_a)

    if yesterday is None:
        return {
            "new_entries": [today_lookup[k] for k in today_entry_pairs],
            "closed": [],
            "continued": [],
            "previous_date": None,
        }

    # Yesterday's entry pairs (normalised the same way)
    yesterday_entry_pairs = set()
    yesterday_lookup = {}
    for s in yesterday:
        if s.get("type") == "ENTRY":
            long_t = s.get("long", "")
            short_t = s.get("short", "")
            key = _norm(long_t, short_t)
            yesterday_entry_pairs.add(key)
            yesterday_lookup[key] = (long_t, short_t)

    new_keys = today_entry_pairs - yesterday_entry_pairs
    closed_keys = yesterday_entry_pairs - today_entry_pairs
    continued_keys = today_entry_pairs & yesterday_entry_pairs

    return {
        "new_entries": [today_lookup[k] for k in new_keys],
        "closed": [yesterday_lookup[k] for k in closed_keys],
        "continued": [today_lookup[k] for k in continued_keys],
        "previous_date": prev_date,
    }

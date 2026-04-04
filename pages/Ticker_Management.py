"""
Ticker Management page — View all tickers in the system and add/remove stocks.

Changes are saved to a local JSON file (ticker_overrides.json) and merged
with the base config on next data refresh.
"""

import json
import os

import pandas as pd
import streamlit as st
import yfinance as yf

from config import (
    ASX_STOCKS,
    ASX_TICKERS,
    COMMODITIES,
    COMMODITY_TICKERS,
    GLOBAL_STOCKS,
    GLOBAL_TICKERS,
    display_name,
)

st.set_page_config(
    page_title="Ticker Management",
    page_icon="$",
    layout="wide",
)

OVERRIDES_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ticker_overrides.json")


def load_overrides() -> dict:
    if os.path.exists(OVERRIDES_FILE):
        with open(OVERRIDES_FILE) as f:
            return json.load(f)
    return {"added_asx": [], "added_global": [], "added_commodities": {}, "removed": []}


def save_overrides(overrides: dict):
    with open(OVERRIDES_FILE, "w") as f:
        json.dump(overrides, f, indent=2)


overrides = load_overrides()

st.title("Ticker Management")
st.caption("View all tickers in the system. Add or remove stocks without editing code.")

# ---------------------------------------------------------------------------
# Current Universe
# ---------------------------------------------------------------------------

st.subheader("Current Ticker Universe")

tab1, tab2, tab3 = st.tabs(["ASX Mining Stocks", "Commodities", "Global Stocks"])

with tab1:
    asx_rows = []
    for ticker in ASX_STOCKS:
        full = f"{ticker}.AX"
        removed = full in overrides.get("removed", [])
        asx_rows.append({
            "Ticker": ticker,
            "yfinance Symbol": full,
            "Status": "Removed" if removed else "Active",
        })
    # Add any user-added ASX tickers
    for ticker in overrides.get("added_asx", []):
        asx_rows.append({
            "Ticker": ticker.replace(".AX", ""),
            "yfinance Symbol": ticker,
            "Status": "Added (custom)",
        })

    asx_df = pd.DataFrame(asx_rows)
    st.dataframe(asx_df, use_container_width=True, hide_index=True)
    st.caption(f"{len([r for r in asx_rows if r['Status'] == 'Active'])} active ASX stocks")

with tab2:
    comm_rows = []
    for name, ticker in COMMODITIES.items():
        removed = ticker in overrides.get("removed", [])
        comm_rows.append({
            "Commodity": name,
            "yfinance Symbol": ticker,
            "Status": "Removed" if removed else "Active",
        })
    for name, ticker in overrides.get("added_commodities", {}).items():
        comm_rows.append({
            "Commodity": name,
            "yfinance Symbol": ticker,
            "Status": "Added (custom)",
        })

    comm_df = pd.DataFrame(comm_rows)
    st.dataframe(comm_df, use_container_width=True, hide_index=True)

with tab3:
    global_rows = []
    for ticker in GLOBAL_STOCKS:
        removed = ticker in overrides.get("removed", [])
        global_rows.append({
            "Ticker": ticker,
            "Display Name": display_name(ticker),
            "Status": "Removed" if removed else "Active",
        })
    for ticker in overrides.get("added_global", []):
        global_rows.append({
            "Ticker": ticker,
            "Display Name": ticker,
            "Status": "Added (custom)",
        })

    global_df = pd.DataFrame(global_rows)
    st.dataframe(global_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Add Tickers
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Add a Ticker")

add_col1, add_col2 = st.columns(2)

with add_col1:
    add_type = st.selectbox("Type", ["ASX Stock", "Global Stock", "Commodity"])
    if add_type == "Commodity":
        add_name = st.text_input("Commodity Name (e.g. Zinc)", help="Human-readable name displayed in the dashboard")
    add_ticker = st.text_input(
        "yfinance Ticker Symbol",
        help="For ASX stocks enter just the code (e.g. FMG), the .AX suffix is added automatically. "
             "For global/commodity enter the full yfinance symbol (e.g. VALE, ZN=F).",
    )

with add_col2:
    st.markdown("**Verify before adding:**")
    if add_ticker:
        verify_ticker = f"{add_ticker}.AX" if add_type == "ASX Stock" and not add_ticker.endswith(".AX") else add_ticker
        try:
            test = yf.Ticker(verify_ticker)
            hist = test.history(period="5d", auto_adjust=True)
            if not hist.empty:
                last_price = hist["Close"].iloc[-1]
                st.success(f"{verify_ticker} is valid. Last price: ${last_price:,.2f}")
            else:
                st.error(f"{verify_ticker} returned no data. Check the symbol.")
        except Exception as e:
            st.error(f"Failed to verify {verify_ticker}: {e}")
    else:
        st.info("Enter a ticker symbol to verify it.")

if st.button("Add Ticker", type="primary"):
    if not add_ticker:
        st.error("Please enter a ticker symbol.")
    else:
        if add_type == "ASX Stock":
            full_ticker = f"{add_ticker}.AX" if not add_ticker.endswith(".AX") else add_ticker
            if full_ticker not in overrides["added_asx"] and full_ticker not in ASX_TICKERS:
                overrides["added_asx"].append(full_ticker)
                save_overrides(overrides)
                st.success(f"Added {full_ticker} to ASX stocks. Refresh data on the main page to include it.")
                st.rerun()
            else:
                st.warning(f"{full_ticker} is already in the system.")
        elif add_type == "Global Stock":
            if add_ticker not in overrides["added_global"] and add_ticker not in GLOBAL_TICKERS:
                overrides["added_global"].append(add_ticker)
                save_overrides(overrides)
                st.success(f"Added {add_ticker} to global stocks. Refresh data on the main page to include it.")
                st.rerun()
            else:
                st.warning(f"{add_ticker} is already in the system.")
        elif add_type == "Commodity":
            if not add_name:
                st.error("Please enter a commodity name.")
            elif add_ticker not in COMMODITY_TICKERS and add_ticker not in overrides.get("added_commodities", {}).values():
                overrides.setdefault("added_commodities", {})[add_name] = add_ticker
                save_overrides(overrides)
                st.success(f"Added {add_name} ({add_ticker}) to commodities. Refresh data on the main page to include it.")
                st.rerun()
            else:
                st.warning(f"{add_ticker} is already in the system.")

# ---------------------------------------------------------------------------
# Remove Tickers
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Remove a Ticker")

all_active = []
for t in ASX_TICKERS:
    if t not in overrides.get("removed", []):
        all_active.append(t)
for t in COMMODITY_TICKERS:
    if t not in overrides.get("removed", []):
        all_active.append(t)
for t in GLOBAL_TICKERS:
    if t not in overrides.get("removed", []):
        all_active.append(t)
for t in overrides.get("added_asx", []):
    all_active.append(t)
for t in overrides.get("added_global", []):
    all_active.append(t)
for t in overrides.get("added_commodities", {}).values():
    all_active.append(t)

display_options = [f"{display_name(t)} ({t})" for t in sorted(all_active)]
remove_selection = st.selectbox("Select ticker to remove", options=display_options)

if st.button("Remove Ticker", type="secondary"):
    if remove_selection:
        # Extract the raw ticker from "Name (TICKER)" format
        raw_ticker = remove_selection.split("(")[-1].rstrip(")")
        if raw_ticker in overrides.get("added_asx", []):
            overrides["added_asx"].remove(raw_ticker)
        elif raw_ticker in overrides.get("added_global", []):
            overrides["added_global"].remove(raw_ticker)
        elif raw_ticker in [v for v in overrides.get("added_commodities", {}).values()]:
            overrides["added_commodities"] = {k: v for k, v in overrides["added_commodities"].items() if v != raw_ticker}
        else:
            overrides.setdefault("removed", []).append(raw_ticker)
        save_overrides(overrides)
        st.success(f"Removed {raw_ticker}. Refresh data on the main page to apply.")
        st.rerun()

# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

st.markdown("---")
if st.button("Reset All Changes", help="Remove all custom additions and removals, reverting to the default ticker universe."):
    if os.path.exists(OVERRIDES_FILE):
        os.remove(OVERRIDES_FILE)
    st.success("All customizations cleared. Refresh data on the main page.")
    st.rerun()

st.caption(
    f"Custom overrides are saved to `ticker_overrides.json`. "
    f"After adding or removing tickers, click **Refresh Data** on the main dashboard to apply changes."
)

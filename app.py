"""
app.py – Streamlit dashboard for the commodity/mining pairs trading system.

Layout:
  Sidebar:   Risk mode toggle, pair selector, NAV input, data controls
  Top row:   Regime, confidence, margin utilisation (color-coded)
  Main:      Actionable signals table, Z-score chart, regime heatmap

Run:  streamlit run app.py
"""

import logging
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config import (
    ANNUAL_TRADING_DAYS,
    COMMODITIES,
    DEFAULT_LOOKBACK,
    DEFAULT_START_DATE,
    DEFAULT_ZSCORE_EXIT,
    MARGIN_BUFFER_PCT,
    MAX_LEVERAGE,
    display_name,
)
from data_loader import compute_atr, compute_returns, compute_volatility, download_prices
from hmm_engine import (
    RegimeHistory,
    detect_regimes,
    get_current_regime,
    get_transition_matrix,
    regime_stability,
)
from pairs_engine import (
    PairSignal,
    compute_pair_analytics,
    run_pairs_pipeline,
    signals_to_dataframe,
)
from risk_manager import (
    RiskMode,
    get_risk_profile,
    portfolio_risk_summary,
    run_risk_pipeline,
)
from signal_history import (
    compute_changes,
    get_history,
    save_today_signals_full,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: display-friendly ticker
# ---------------------------------------------------------------------------

def dn(ticker: str) -> str:
    """Shorthand for display_name throughout this file."""
    return display_name(ticker)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Pairs Trading System",
    page_icon="$",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached data loading — survives mode toggles and pair selection changes
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Downloading market data...")
def cached_download_prices(start: str) -> pd.DataFrame:
    return download_prices(start=start)


@st.cache_data(ttl=3600, show_spinner="Computing returns & volatility...")
def cached_compute_derived(prices_hash: str, _prices: pd.DataFrame):
    returns = compute_returns(_prices)
    volatility = compute_volatility(returns)
    atr = compute_atr(_prices)
    return returns, volatility, atr


@st.cache_data(ttl=3600, show_spinner="Training HMM regime model...")
def cached_detect_regimes(returns_hash: str, _returns: pd.DataFrame, _volatility: pd.DataFrame):
    return detect_regimes(_returns, _volatility)


@st.cache_data(ttl=3600, show_spinner="Scanning cointegrated pairs...")
def cached_pairs_pipeline(prices_hash: str, _prices: pd.DataFrame, regime_label: str, regime_confidence: float, entry_threshold: float, stop_threshold: float):
    return run_pairs_pipeline(_prices, regime=regime_label, regime_confidence=regime_confidence, entry_threshold=entry_threshold, stop_threshold=stop_threshold)


def df_hash(df: pd.DataFrame) -> str:
    return f"{df.shape}_{df.index[-1]}_{df.iloc[-1].sum():.6f}"


@st.cache_data(ttl=3600, show_spinner="Computing pair analytics...")
def cached_pair_analytics(prices_hash: str, _prices: pd.DataFrame, ticker_a: str, ticker_b: str, window: int):
    return compute_pair_analytics(_prices, ticker_a, ticker_b, window=window)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Control Panel")

risk_mode_str = st.sidebar.radio(
    "Risk Mode",
    options=["Conservative", "Aggressive"],
    index=0,
    help="Conservative: tighter stops, lower leverage. Aggressive: full 5x leverage, wider bands.",
)
risk_mode = RiskMode.CONSERVATIVE if risk_mode_str == "Conservative" else RiskMode.AGGRESSIVE
profile = get_risk_profile(risk_mode)

nav = st.sidebar.number_input(
    "Portfolio NAV ($)",
    min_value=10_000,
    max_value=100_000_000,
    value=1_000_000,
    step=100_000,
    format="%d",
)

start_date = st.sidebar.text_input("Data Start Date", value="2022-01-01")

st.sidebar.markdown("---")

# FIX #2: Renamed from "Re-download Data" to "Refresh Data"
reload_btn = st.sidebar.button("Refresh Data", type="primary", help="Pull the latest close prices from Yahoo Finance. Hit this once a day.")

if reload_btn:
    cached_download_prices.clear()
    cached_compute_derived.clear()
    cached_detect_regimes.clear()
    cached_pairs_pipeline.clear()
    st.rerun()


# ---------------------------------------------------------------------------
# Data pipeline (cached)
# ---------------------------------------------------------------------------

try:
    prices = cached_download_prices(start_date)
except Exception as e:
    st.error(f"Failed to download data: {e}")
    st.stop()

p_hash = df_hash(prices)
returns, volatility, atr = cached_compute_derived(p_hash, prices)

r_hash = df_hash(returns)
regime_history: RegimeHistory = cached_detect_regimes(r_hash, returns, volatility)
current_regime = get_current_regime(regime_history)
stability = regime_stability(regime_history)

pairs_result = cached_pairs_pipeline(
    p_hash, prices,
    regime_label=current_regime.label,
    regime_confidence=current_regime.confidence,
    entry_threshold=profile.zscore_entry,
    stop_threshold=profile.zscore_stop,
)

# Risk sizing (NOT cached — recalculates instantly on mode toggle)
risk_result = run_risk_pipeline(
    signals=pairs_result["signals"],
    prices=prices, volatility=volatility, atr=atr,
    mode=risk_mode, regime=current_regime, nav=nav,
)
portfolio_risk = risk_result["portfolio_risk"]

valid_pairs = pairs_result["valid_pairs"]

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Mode: **{risk_mode_str}** | "
    f"Leverage cap: {profile.max_leverage_util:.1f}x | "
    f"Vol target: {profile.vol_target:.0%}"
)


# ---------------------------------------------------------------------------
# Drill-down fragment: pair selector + Z-score chart
#
# Wrapped in @st.fragment so changing the pair ONLY reruns this section —
# the rest of the app (data pipeline, top-row metrics, signals table)
# stays put and doesn't flicker on each pair change.
# ---------------------------------------------------------------------------

@st.fragment
def drill_down_chart(valid_pairs, prices, p_hash, profile):
    pair_options = [f"{dn(p.ticker_a)} / {dn(p.ticker_b)}" for p in valid_pairs]

    st.subheader("Z-Score: Selected Pair")

    if not pair_options:
        st.info("No cointegrated pairs found.")
        return

    # Selector lives INSIDE the fragment (not the sidebar) so changing it
    # only reruns this fragment. Streamlit disallows sidebar widgets
    # inside fragments.
    selected_pair_str = st.selectbox(
        "Drill-down Pair",
        options=pair_options,
        index=0,
        key="drill_down_selector",
        help="Change this to view the Z-score chart for any cointegrated pair. Only this chart reloads.",
    )
    sel_idx = pair_options.index(selected_pair_str)
    sel_pair = valid_pairs[sel_idx]
    try:
        analytics = cached_pair_analytics(
            p_hash, prices, sel_pair.ticker_a, sel_pair.ticker_b, window=DEFAULT_LOOKBACK,
        )
        zscore_series = analytics["zscore"].dropna()
        display_window = min(252, len(zscore_series))
        zs = zscore_series.iloc[-display_window:]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=zs.index, y=zs.values,
            mode="lines", name="Z-Score",
            line=dict(color="white", width=2),
        ))

        fig.add_hline(y=profile.zscore_entry, line_dash="dash", line_color="cyan", annotation_text=f"Entry +{profile.zscore_entry}")
        fig.add_hline(y=-profile.zscore_entry, line_dash="dash", line_color="cyan", annotation_text=f"Entry -{profile.zscore_entry}")
        fig.add_hline(y=DEFAULT_ZSCORE_EXIT, line_dash="dot", line_color="lime", annotation_text=f"Exit +{DEFAULT_ZSCORE_EXIT}")
        fig.add_hline(y=-DEFAULT_ZSCORE_EXIT, line_dash="dot", line_color="lime", annotation_text=f"Exit -{DEFAULT_ZSCORE_EXIT}")
        fig.add_hline(y=profile.zscore_stop, line_dash="solid", line_color="red", annotation_text=f"Stop +{profile.zscore_stop}")
        fig.add_hline(y=-profile.zscore_stop, line_dash="solid", line_color="red", annotation_text=f"Stop -{profile.zscore_stop}")
        fig.add_hline(y=0, line_color="grey", line_width=0.5)

        fig.add_hrect(y0=profile.zscore_entry, y1=profile.zscore_stop, fillcolor="rgba(255,165,0,0.08)", line_width=0)
        fig.add_hrect(y0=-profile.zscore_stop, y1=-profile.zscore_entry, fillcolor="rgba(255,165,0,0.08)", line_width=0)
        fig.add_hrect(y0=profile.zscore_stop, y1=max(zs.max() * 1.1, profile.zscore_stop + 1), fillcolor="rgba(255,0,0,0.1)", line_width=0)
        fig.add_hrect(y0=min(zs.min() * 1.1, -profile.zscore_stop - 1), y1=-profile.zscore_stop, fillcolor="rgba(255,0,0,0.1)", line_width=0)

        fig.update_layout(
            title=f"{dn(sel_pair.ticker_a)} / {dn(sel_pair.ticker_b)}  (HL={sel_pair.half_life:.0f}d, p={sel_pair.coint_pvalue:.4f})",
            yaxis_title="Z-Score",
            template="plotly_dark",
            height=450,
            margin=dict(l=50, r=20, t=50, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"zscore_{sel_pair.ticker_a}_{sel_pair.ticker_b}")

    except Exception as e:
        st.error(f"Failed to compute analytics for selected pair: {e}")

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

st.title("Commodity & Mining Pairs Trading System")

# ---------------------------------------------------------------------------
# Top row: Fatal Risk Monitors
# ---------------------------------------------------------------------------

col1, col2, col3, col4, col5 = st.columns(5)

regime_colors = {"BULL": "green", "BEAR": "red", "CHOP": "orange"}
regime_color = regime_colors.get(current_regime.label, "grey")
col1.metric("Market Regime", current_regime.label)
col1.markdown(
    f"<span style='color:{regime_color}; font-weight:bold; font-size:0.9em;'>"
    f"{'Favorable for Pairs' if current_regime.is_favorable_for_pairs else 'Unfavorable — Trending'}"
    f"</span>", unsafe_allow_html=True,
)

col2.metric("Regime Confidence", f"{current_regime.confidence:.1%}")
col2.caption(f"Stability (21d): {stability:.0%}")

margin_used = portfolio_risk.margin_utilisation
margin_remaining = portfolio_risk.margin_remaining
if margin_remaining < MARGIN_BUFFER_PCT:
    margin_color, margin_icon = "red", "DANGER"
elif margin_remaining < MARGIN_BUFFER_PCT * 2:
    margin_color, margin_icon = "orange", "CAUTION"
else:
    margin_color, margin_icon = "green", "OK"

col3.metric("Margin Used", f"{margin_used:.1%}")
col3.markdown(
    f"<span style='color:{margin_color}; font-weight:bold; font-size:0.9em;'>"
    f"Remaining: {margin_remaining:.1%} [{margin_icon}]</span>", unsafe_allow_html=True,
)

lev = portfolio_risk.effective_leverage
lev_color = "red" if lev > profile.max_leverage_util else ("orange" if lev > profile.max_leverage_util * 0.8 else "green")
col4.metric("Eff. Leverage", f"{lev:.1f}x / {profile.max_leverage_util:.0f}x")
col4.markdown(
    f"<span style='color:{lev_color}; font-weight:bold; font-size:0.9em;'>"
    f"Broker max: {MAX_LEVERAGE:.0f}x</span>", unsafe_allow_html=True,
)

col5.metric("Active Pairs", portfolio_risk.num_active_pairs)
col5.metric("Risk Mode", risk_mode_str.upper())

if portfolio_risk.warnings:
    for w in portfolio_risk.warnings:
        st.warning(w)

# ---------------------------------------------------------------------------
# Signal History — save today's snapshot and compute changes
# ---------------------------------------------------------------------------

data_date = str(prices.index[-1].date())
save_today_signals_full(
    positions=risk_result["positions"],
    signals=pairs_result["signals"],
    data_date=data_date,
)
changes = compute_changes(risk_result["positions"], pairs_result["signals"], data_date)

# ---------------------------------------------------------------------------
# What Changed Since Yesterday
# ---------------------------------------------------------------------------

st.markdown("---")

if changes["previous_date"]:
    change_col1, change_col2, change_col3 = st.columns(3)
    with change_col1:
        st.metric("New Entries Today", len(changes["new_entries"]))
    with change_col2:
        st.metric("Positions Closed", len(changes["closed"]))
    with change_col3:
        st.metric("Continuing Positions", len(changes["continued"]))

    if changes["new_entries"]:
        st.success(f"**New since {changes['previous_date']}:** " + ", ".join(f"{dn(a)}/{dn(b)}" for a, b in changes["new_entries"]))
    if changes["closed"]:
        st.error(f"**Closed since {changes['previous_date']}:** " + ", ".join(f"{dn(a)}/{dn(b)}" for a, b in changes["closed"]))
    if not changes["new_entries"] and not changes["closed"]:
        st.info(f"No changes since {changes['previous_date']}. All {len(changes['continued'])} positions continue.")
else:
    st.info("First run — no previous day to compare against. Signal history will build up from today.")

# ---------------------------------------------------------------------------
# Actionable Signals Table
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Actionable Signals")

positions = risk_result["positions"]
entry_positions = [p for p in positions if p.signal in ("LONG_A_SHORT_B", "SHORT_A_LONG_B")]
exit_positions = [p for p in positions if p.signal in ("EXIT", "STOP")]

def _fmt_price(price: float) -> str:
    """Format price with enough decimal places for penny stocks."""
    if price < 1:
        return f"${price:.4f}"
    return f"${price:,.2f}"


# Separate clean entries from those where the last close already crossed the stop
clean_entries = [p for p in entry_positions if not (p.stop_triggered_a or p.stop_triggered_b)]
triggered_entries = [p for p in entry_positions if p.stop_triggered_a or p.stop_triggered_b]

if clean_entries:
    rows = []
    for p in clean_entries:
        if p.signal == "LONG_A_SHORT_B":
            long_tk, short_tk = p.ticker_a, p.ticker_b
            long_w, short_w = p.weight_a, p.weight_b
            long_n, short_n = p.notional_a, p.notional_b
            long_stop, short_stop = p.stop_loss_a, p.stop_loss_b
            long_price, short_price = p.price_a, p.price_b
        else:
            long_tk, short_tk = p.ticker_b, p.ticker_a
            long_w, short_w = p.weight_b, p.weight_a
            long_n, short_n = p.notional_b, p.notional_a
            long_stop, short_stop = p.stop_loss_b, p.stop_loss_a
            long_price, short_price = p.price_b, p.price_a

        # Distance from current price to stop (shows how close we are to getting stopped out)
        long_to_stop = abs(long_price - long_stop) / long_price
        short_to_stop = abs(short_price - short_stop) / short_price

        rows.append({
            "Long": dn(long_tk),
            "Long Price": _fmt_price(long_price),
            "Long Stop": _fmt_price(long_stop),
            "L Stop %": f"{long_to_stop:.1%}",
            "Short": dn(short_tk),
            "Short Price": _fmt_price(short_price),
            "Short Stop": _fmt_price(short_stop),
            "S Stop %": f"{short_to_stop:.1%}",
            "Z-Score": round(
                next((s.zscore for s in pairs_result["signals"]
                      if s.ticker_a == p.ticker_a and s.ticker_b == p.ticker_b), 0.0), 3,
            ),
            "Long Notional": f"${long_n:,.0f}",
            "Short Notional": f"${short_n:,.0f}",
            "Max Loss": f"{p.max_loss_pct:.2%}",
            "Eff. Lev.": f"{p.effective_leverage:.1f}x",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "L Stop % / S Stop % show how far the current price is from the stop. "
        "A value near 0% means the price is right at the stop — execute with caution."
    )
elif not triggered_entries:
    st.info("No actionable entry signals at this time. The market may be in a trending regime or no pairs meet the Z-score threshold.")

if triggered_entries:
    st.warning(
        f"{len(triggered_entries)} entry signal(s) have prices that already crossed the stop (data is from last close; "
        f"these pairs gapped past the stop level overnight). Do NOT enter these trades."
    )
    with st.expander(f"Skipped — Entry Already Past Stop ({len(triggered_entries)})"):
        trig_rows = []
        for p in triggered_entries:
            trig_rows.append({
                "Ticker A": dn(p.ticker_a),
                "Price A": _fmt_price(p.price_a),
                "Stop A": _fmt_price(p.stop_loss_a),
                "A Triggered": "Yes" if p.stop_triggered_a else "No",
                "Ticker B": dn(p.ticker_b),
                "Price B": _fmt_price(p.price_b),
                "Stop B": _fmt_price(p.stop_loss_b),
                "B Triggered": "Yes" if p.stop_triggered_b else "No",
                "Signal": p.signal,
            })
        st.dataframe(pd.DataFrame(trig_rows), use_container_width=True, hide_index=True)

# STOP signals shown prominently (emergency — cointegration may be breaking down)
stop_positions = [p for p in positions if p.signal == "STOP"]
if stop_positions:
    st.subheader(f"STOP Signals ({len(stop_positions)})")
    st.caption("These pairs have blown past the emergency threshold. If you hold any of them, exit immediately.")
    stop_rows = []
    for p in stop_positions:
        zs = next((s.zscore for s in pairs_result["signals"] if s.ticker_a == p.ticker_a and s.ticker_b == p.ticker_b), 0.0)
        stop_rows.append({
            "Ticker A": dn(p.ticker_a),
            "Ticker B": dn(p.ticker_b),
            "Z-Score": round(zs, 3),
            "Signal": p.signal,
        })
    st.dataframe(pd.DataFrame(stop_rows), use_container_width=True, hide_index=True)

# EXIT signals in a less prominent section — these fire for ALL cointegrated pairs
# near fair value, not just ones the user actually holds
exit_only = [p for p in positions if p.signal == "EXIT"]
if exit_only:
    with st.expander(f"Pairs at Fair Value ({len(exit_only)})"):
        st.caption(
            "These cointegrated pairs have z-scores near zero, meaning the spread has reverted "
            "to its mean. If you hold any of these pairs from a previous entry signal, this is "
            "the take-profit zone. Pairs listed here were NOT necessarily previous entry signals."
        )
        exit_rows = []
        for p in exit_only:
            zs = next((s.zscore for s in pairs_result["signals"] if s.ticker_a == p.ticker_a and s.ticker_b == p.ticker_b), 0.0)
            exit_rows.append({
                "Ticker A": dn(p.ticker_a),
                "Ticker B": dn(p.ticker_b),
                "Z-Score": round(zs, 3),
            })
        st.dataframe(pd.DataFrame(exit_rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

st.markdown("---")

viz_col1, viz_col2 = st.columns([3, 2])

# --- Z-Score chart for selected pair (fragment — reruns independently) ---
with viz_col1:
    drill_down_chart(valid_pairs, prices, p_hash, profile)


# --- Regime Heatmap ---
with viz_col2:
    st.subheader("Market Regime History")

    posteriors = regime_history.posterior_probs
    display_days = min(120, len(posteriors))
    recent_posteriors = posteriors.iloc[-display_days:]

    fig_regime = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=["Regime Posterior Probabilities", "Regime Classification"],
    )

    for col_name in recent_posteriors.columns:
        color = {"BEAR": "rgba(239,83,80,0.7)", "CHOP": "rgba(255,183,77,0.7)", "BULL": "rgba(76,175,80,0.7)"}.get(col_name, "grey")
        fig_regime.add_trace(go.Scatter(
            x=recent_posteriors.index, y=recent_posteriors[col_name],
            mode="lines", name=col_name, stackgroup="one",
            line=dict(width=0.5), fillcolor=color,
        ), row=1, col=1)

    recent_labels = regime_history.labels.iloc[-display_days:]
    regime_numeric = recent_labels.map({"BEAR": 0, "CHOP": 1, "BULL": 2}).fillna(1)

    fig_regime.add_trace(go.Heatmap(
        x=recent_labels.index, z=[regime_numeric.values],
        colorscale=[[0, "rgba(239,83,80,0.9)"], [0.5, "rgba(255,183,77,0.9)"], [1, "rgba(76,175,80,0.9)"]],
        showscale=False, hovertext=[recent_labels.values], hoverinfo="text",
    ), row=2, col=1)

    fig_regime.update_layout(
        template="plotly_dark", height=450,
        margin=dict(l=50, r=20, t=40, b=40),
        showlegend=True, legend=dict(orientation="h", y=1.12),
    )
    fig_regime.update_yaxes(title_text="Probability", row=1, col=1)
    fig_regime.update_yaxes(showticklabels=False, row=2, col=1)

    st.plotly_chart(fig_regime, use_container_width=True)

    # FIX #5: Explanatory note about regime persistence
    st.caption(
        "Regimes are intentionally persistent — the HMM detects structural market states that last weeks to months, not daily fluctuations. "
        "BEAR persists ~96% of the time, BULL/CHOP ~87%. Small day-to-day changes are normal; look for color transitions at regime boundaries."
    )

    with st.expander("Regime Transition Matrix"):
        trans = get_transition_matrix(regime_history)
        st.dataframe(trans.style.format("{:.1%}").background_gradient(cmap="YlOrRd", axis=1))

# ---------------------------------------------------------------------------
# Portfolio Summary (bottom)
# ---------------------------------------------------------------------------

st.markdown("---")

summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    st.subheader("Portfolio Risk Summary")
    summary = portfolio_risk_summary(portfolio_risk)
    summary_df = pd.DataFrame(list(summary.items()), columns=["Metric", "Value"])
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

with summary_col2:
    st.subheader("Regime Detail")
    regime_detail = {
        "Current Regime": current_regime.label,
        "Confidence": f"{current_regime.confidence:.1%}",
        "21-Day Stability": f"{stability:.0%}",
        "Favorable for Pairs": "Yes" if current_regime.is_favorable_for_pairs else "No",
    }
    for label, prob in current_regime.posteriors.items():
        regime_detail[f"P({label})"] = f"{prob:.1%}"
    # FIX #6: Annualize mean returns so they're readable percentages
    for label, mean_ret in regime_history.state_means.items():
        annual_ret = mean_ret * ANNUAL_TRADING_DAYS * 100
        regime_detail[f"Avg Annual Return ({label})"] = f"{annual_ret:+.1f}%"

    regime_df = pd.DataFrame(list(regime_detail.items()), columns=["Metric", "Value"])
    st.dataframe(regime_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# All Pairs (expandable)
# ---------------------------------------------------------------------------

with st.expander(f"All Cointegrated Pairs ({len(valid_pairs)})"):
    if valid_pairs:
        all_rows = []
        for p in valid_pairs:
            all_rows.append({
                # FIX #3: Display names
                "Ticker A": dn(p.ticker_a),
                "Ticker B": dn(p.ticker_b),
                "Coint p-value": f"{p.coint_pvalue:.4f}",
                "Hedge Ratio": f"{p.hedge_ratio:.4f}",
                "Half-Life (d)": f"{p.half_life:.1f}",
            })
        st.dataframe(pd.DataFrame(all_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No cointegrated pairs found in the current data window.")

with st.expander("All Signals (including FLAT)"):
    signals_df = pairs_result["signals_df"]
    if not signals_df.empty:
        # FIX #3: Replace tickers with display names in the Pair column
        display_signals = signals_df.copy()
        if "Pair" in display_signals.columns:
            display_signals["Pair"] = display_signals["Pair"].apply(
                lambda x: " / ".join(dn(t.strip()) for t in x.split("/"))
            )
        st.dataframe(display_signals, use_container_width=True, hide_index=True)
    else:
        st.info("No signals generated.")

# ---------------------------------------------------------------------------
# Gold Hedge Tracker (Client-specific)
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Gold Hedge Tracker")
st.caption("The fund uses gold as a core hedge. This section tracks all gold-related pair signals — past and present — regardless of whether gold is currently selected above.")

gold_ticker = COMMODITIES.get("Gold", "GC=F")

gold_signals_today = [
    s for s in pairs_result["signals"]
    if gold_ticker in (s.ticker_a, s.ticker_b)
]

if gold_signals_today:
    gold_today_rows = []
    for s in gold_signals_today:
        partner = s.ticker_b if s.ticker_a == gold_ticker else s.ticker_a
        gold_side = "A" if s.ticker_a == gold_ticker else "B"
        matching_pos = next(
            (p for p in risk_result["positions"]
             if p.ticker_a == s.ticker_a and p.ticker_b == s.ticker_b), None,
        )

        gold_today_rows.append({
            "Partner": dn(partner),
            "Signal": s.signal,
            "Z-Score": round(s.zscore, 3),
            "Gold Side": f"{'Long' if s.signal == 'LONG_A_SHORT_B' and gold_side == 'A' or s.signal == 'SHORT_A_LONG_B' and gold_side == 'B' else 'Short'}" if s.signal not in ("FLAT", "EXIT", "STOP") else s.signal,
            "Hedge Ratio": round(s.hedge_ratio, 4),
            "Half-Life (d)": round(s.half_life, 1),
            "Coint p-value": round(s.coint_pvalue, 4) if s.coint_pvalue else "N/A",
            "Weight": f"{matching_pos.weight_a if gold_side == 'A' else matching_pos.weight_b:+.2%}" if matching_pos and s.signal not in ("FLAT",) else "---",
            "Notional": f"${matching_pos.notional_a if gold_side == 'A' else matching_pos.notional_b:,.0f}" if matching_pos and s.signal not in ("FLAT",) else "---",
        })

    gold_col1, gold_col2 = st.columns([3, 1])
    with gold_col1:
        st.markdown(f"**Today's Gold Signals** — {len(gold_signals_today)} pairs involving Gold")
        st.dataframe(pd.DataFrame(gold_today_rows), use_container_width=True, hide_index=True)
    with gold_col2:
        actionable_gold = [s for s in gold_signals_today if s.signal not in ("FLAT", "EXIT")]
        st.metric("Gold Pairs Active", len(actionable_gold))
        st.metric("Gold Pairs Total", len(gold_signals_today))
        if gold_ticker in prices.columns:
            gold_price = prices[gold_ticker].dropna().iloc[-1]
            gold_prev = prices[gold_ticker].dropna().iloc[-2] if len(prices[gold_ticker].dropna()) > 1 else gold_price
            gold_chg = (gold_price - gold_prev) / gold_prev * 100
            st.metric("Gold Price", f"${gold_price:,.2f}", f"{gold_chg:+.2f}%")
else:
    st.info(f"No cointegrated pairs currently involve Gold. Gold may not have a statistically significant relationship with any ASX miner in the current lookback window.")
    if gold_ticker in prices.columns:
        gold_price = prices[gold_ticker].dropna().iloc[-1]
        gold_prev = prices[gold_ticker].dropna().iloc[-2] if len(prices[gold_ticker].dropna()) > 1 else gold_price
        gold_chg = (gold_price - gold_prev) / gold_prev * 100
        st.metric("Gold Price", f"${gold_price:,.2f}", f"{gold_chg:+.2f}%")

gold_valid_pairs = [p for p in valid_pairs if gold_ticker in (p.ticker_a, p.ticker_b)]

if gold_valid_pairs:
    with st.expander(f"Gold Pair Z-Score History ({len(gold_valid_pairs)} cointegrated pairs)"):
        gold_zscore_data = {}
        for gp in gold_valid_pairs:
            try:
                ga = cached_pair_analytics(p_hash, prices, gp.ticker_a, gp.ticker_b, window=DEFAULT_LOOKBACK)
                partner = gp.ticker_b if gp.ticker_a == gold_ticker else gp.ticker_a
                zs = ga["zscore"].dropna()
                if len(zs) > 0:
                    gold_zscore_data[dn(partner)] = zs.iloc[-120:]
            except Exception:
                continue

        if gold_zscore_data:
            gold_zdf = pd.DataFrame(gold_zscore_data)
            fig_gold = go.Figure(data=go.Heatmap(
                z=gold_zdf.T.values, x=gold_zdf.index, y=gold_zdf.columns,
                colorscale=[
                    [0, "rgb(239,83,80)"], [0.35, "rgb(255,183,77)"],
                    [0.5, "rgb(255,255,255)"], [0.65, "rgb(129,199,132)"],
                    [1, "rgb(38,166,91)"],
                ],
                zmid=0, zmin=-3, zmax=3,
                colorbar=dict(title="Z-Score"),
                hovertemplate="Pair: Gold/%{y}<br>Date: %{x}<br>Z-Score: %{z:.2f}<extra></extra>",
            ))
            fig_gold.update_layout(
                title="Gold Pair Z-Scores (Last 120 Days) — Red = Entry Opportunity",
                template="plotly_dark",
                height=max(250, len(gold_zscore_data) * 35 + 100),
                margin=dict(l=100, r=20, t=50, b=40),
                yaxis=dict(dtick=1),
            )
            st.plotly_chart(fig_gold, use_container_width=True)
            st.caption(
                "Red cells = negative Z-score (gold leg is cheap relative to partner — potential long gold hedge). "
                "Green cells = positive Z-score (gold leg is expensive — potential short gold / long partner). "
                "White = fair value (Z near 0)."
            )
else:
    with st.expander("Gold Pair Z-Score History"):
        st.info("No cointegrated gold pairs to display.")


# ---------------------------------------------------------------------------
# Signal History (rolling log)
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Signal History")
st.caption("Rolling log of past daily signals. Builds up over time as you refresh data each day.")

history = get_history()
if history:
    history_rows = []
    for hist_date in sorted(history.keys(), reverse=True)[:30]:
        day_signals = history[hist_date]
        entries = [s for s in day_signals if s.get("type") == "ENTRY"]
        exits = [s for s in day_signals if s.get("type") in ("EXIT", "STOP")]
        for e in entries:
            history_rows.append({
                "Date": hist_date,
                "Type": "ENTRY",
                "Long": e.get("long_display", e.get("long", "")),
                "Short": e.get("short_display", e.get("short", "")),
                "Z-Score": e.get("zscore", ""),
            })
        for x in exits:
            history_rows.append({
                "Date": hist_date,
                "Type": x.get("signal", "EXIT"),
                "Long": x.get("a_display", x.get("ticker_a", "")),
                "Short": x.get("b_display", x.get("ticker_b", "")),
                "Z-Score": "",
            })
    if history_rows:
        st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)
    else:
        st.info("History exists but no entry/exit signals were recorded.")
else:
    st.info("No signal history yet. History builds up as you refresh data each day.")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    f"Data: {prices.index[0].date()} to {prices.index[-1].date()} | "
    f"{len(prices.columns)} tickers | "
    f"HMM trained on {len(regime_history.labels)} days | "
    f"Forward-fill limit: 3 days | "
    f"Mode: {risk_mode_str}"
)

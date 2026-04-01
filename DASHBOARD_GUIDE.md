# Pairs Trading Dashboard — Complete User Guide

This document explains every component of the Streamlit dashboard, how to interpret the results, and how to use the system for daily trade decision-making.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Sidebar — Control Panel](#sidebar--control-panel)
3. [Top Row — Fatal Risk Monitors](#top-row--fatal-risk-monitors)
4. [Actionable Signals Table](#actionable-signals-table)
5. [Z-Score Chart](#z-score-chart)
6. [Market Regime History](#market-regime-history)
7. [Portfolio Risk Summary](#portfolio-risk-summary)
8. [Regime Detail](#regime-detail)
9. [Expandable Sections](#expandable-sections)
10. [How the System Makes Decisions](#how-the-system-makes-decisions)
11. [Daily Workflow](#daily-workflow)
12. [Understanding the Current Results](#understanding-the-current-results)
13. [Configuration Reference](#configuration-reference)

---

## Getting Started

```bash
cd /Users/brendan/Matt_Michael
pip3 install -r requirements.txt
streamlit run app.py
```

Open **http://localhost:8501** in your browser. The first load takes 1–2 minutes as it downloads price data for all 54 tickers from Yahoo Finance, trains the HMM regime model, and scans ~1,350 candidate pairs for cointegration. After first load, everything is cached for 1 hour — toggling modes and selecting pairs is instant.

---

## Sidebar — Control Panel

The sidebar is your primary interface for controlling the system. It contains four elements.

### Risk Mode Toggle (Conservative / Aggressive)

This is the most important control. It changes **everything downstream** — which pairs generate signals, how large the positions are, where the stops sit, and how the Z-score bands are drawn on the chart.

| Parameter | Conservative | Aggressive |
|---|---|---|
| Max leverage utilisation | 2.5x | 5.0x |
| Annualised volatility target | 10% | 20% |
| ATR stop-loss multiplier | 1.5x | 2.5x |
| Min HMM regime confidence | 75% | 55% |
| Z-score entry threshold | 2.5 | 1.8 |
| Z-score stop threshold | 3.0 | 4.0 |

**Conservative** is the default. It requires wider Z-score deviations before entering (2.5 vs 1.8), uses tighter stops (1.5x ATR vs 2.5x), and caps effective leverage at 2.5x. Use this when you want capital preservation or when the regime is uncertain.

**Aggressive** opens the full 5x leverage envelope, enters on smaller Z-score deviations (catching more trades), and allows wider stops. Use this when the regime is CHOP (mean-reverting) with high confidence and you want maximum alpha.

**Key behavior**: Toggling between modes does NOT re-download data or re-train the HMM. It instantly recalculates position sizing and re-draws the Z-score bands. The data pipeline and regime detection are cached.

### Portfolio NAV ($)

Enter your current account value. This drives all the dollar-denominated outputs in the signals table (notional amounts). Default is $1,000,000. The percentage-based metrics (weights, margin utilisation, max loss) are unaffected by NAV — they scale proportionally.

### Data Start Date

Controls how far back the historical data goes. Default is `2022-01-01`. More history gives the cointegration tests and HMM more data to work with (better statistical power), but takes longer to download. The HMM uses the most recent ~504 trading days (~2 years) for training regardless of how far back the data goes, so the start date primarily affects the cointegration scanner's lookback.

### Re-download Data

Clears all caches and forces a fresh download from Yahoo Finance. Use this:
- At the start of each trading day (to get the latest close prices)
- If you suspect stale data
- After market holidays where forward-filling may have kicked in

### Drill-down Pair Selector

A dropdown listing all cointegrated pairs that passed validation (p-value < 0.05, half-life between 5–120 days). Selecting a pair updates the Z-Score chart in the main panel. The pairs are sorted by cointegration p-value (strongest statistical relationship first).

### Sidebar Footer

Shows a summary of the active mode's key parameters: leverage cap, volatility target. This is a quick reference so you always know which profile is active.

---

## Top Row — Fatal Risk Monitors

This is the row of five metric cards across the top of the dashboard. These are the numbers you check first every time you open the dashboard. They tell you whether it is safe to trade today.

### 1. Market Regime

Displays the current HMM-detected regime: **BULL**, **BEAR**, or **CHOP**.

Below the label, a colored annotation tells you whether the regime is favorable for pairs trading:
- **Green "Favorable for Pairs"** — CHOP regime. Mean-reversion is working. This is when pairs trading has its highest expected return.
- **Red/Orange "Unfavorable — Trending"** — BULL or BEAR regime. Markets are directional. Pairs trades tend to bleed during trends because the spread keeps widening instead of reverting.

**What to do**: In CHOP, you can trade with higher conviction. In BULL/BEAR, the system automatically reduces position sizes, but you should also apply extra manual scrutiny to any entry signals.

### 2. Regime Confidence

The posterior probability that the HMM assigns to the current regime. Ranges from 0% to 100%.

- **>80%**: The model is very certain about the current state. Trust the regime label.
- **50–80%**: Moderate certainty. The market may be transitioning between regimes.
- **<50%**: Low confidence. The model is unsure — treat the regime label with skepticism.

**Stability (21d)** is shown below — this is the percentage of the last 21 trading days that had the same regime label as today. 100% means rock-solid, no flickering. Below 60% means the regime has been unstable and you should be cautious.

### 3. Margin Used — THE CRITICAL METRIC

This is the estimated percentage of your NAV consumed by margin requirements across all active positions. With 5x leverage, each dollar of notional exposure requires ~$0.20 of margin.

The color coding:
- **Green [OK]**: Margin remaining is above 30%. Safe operating zone.
- **Orange [CAUTION]**: Margin remaining is between 15% and 30%. You are approaching the buffer. Consider closing some positions or switching to Conservative mode.
- **Red [DANGER]**: Margin remaining is below 15%. **You are at risk of a margin call.** Immediately reduce exposure or add capital.

The 15% buffer threshold is configured in `config.py` as `MARGIN_BUFFER_PCT`. This exists because margin calls are described as a "fatal risk" for the fund — once triggered, the broker liquidates positions at the worst possible time.

### 4. Effective Leverage

Shows the current effective leverage as a ratio: `current / mode_limit`. For example, `1.1x / 5x` means you are using 1.1x of a maximum 5.0x in Aggressive mode.

- **Green**: Below 80% of the mode's limit
- **Orange**: Between 80% and 100% of the limit
- **Red**: Exceeding the limit (the system will flag this as a warning)

The "Broker max: 5x" annotation reminds you of the absolute broker-level cap regardless of which mode is selected.

### 5. Active Pairs / Risk Mode

Shows the count of pairs that currently have entry signals (LONG_A_SHORT_B or SHORT_A_LONG_B). This does NOT count EXIT or STOP signals.

Below it, the active risk mode is displayed as a confirmation of your sidebar selection.

### Warning Banners

If any risk limit is breached or the regime is unfavorable, yellow warning banners appear below the top row. These are generated by the risk manager and include:
- Gross exposure exceeding limits
- Effective leverage exceeding mode limits
- Margin remaining below safety buffer (MARGIN CALL RISK)
- Trending regime with high confidence (elevated risk for pairs)

**Never ignore these warnings.** They are the system telling you something is structurally wrong with the current exposure.

---

## Actionable Signals Table

This is the main trading output — the table of pairs that the system recommends entering today.

### Columns

| Column | Meaning |
|---|---|
| **Long** | The ticker you BUY (go long) |
| **Short** | The ticker you SELL (go short) |
| **Z-Score** | How many standard deviations the spread has deviated from its rolling mean. Negative means the spread is cheap (buy A, sell B). Positive means the spread is rich (sell A, buy B). |
| **Long Weight** | Recommended capital allocation for the long leg, as a percentage of NAV. Example: `+0.64%` means allocate 0.64% of NAV long. |
| **Short Weight** | Same for the short leg. Always negative (short position). Example: `-0.41%` means short 0.41% of NAV. |
| **Long Notional** | Dollar amount for the long leg based on your NAV input. |
| **Short Notional** | Dollar amount for the short leg. |
| **Long Stop** | ATR-based stop-loss price for the long leg. If the long position drops to this price, exit. |
| **Short Stop** | ATR-based stop-loss price for the short leg. If the short position rises to this price, exit. |
| **Max Loss** | Estimated maximum loss as a percentage of NAV if both stops are hit simultaneously. This is the worst-case scenario for this pair. |
| **Eff. Leverage** | The effective leverage this single pair contributes to the portfolio. |

### How to read a signal

Example row:
```
Long: A1M.AX | Short: BML.AX | Z-Score: -2.840
Long Weight: +0.64% | Short Weight: -0.41%
Long Notional: $6,370 | Short Notional: -$4,118
```

This means: The spread between A1M.AX and BML.AX has deviated -2.84 standard deviations below its mean. The system recommends going long A1M.AX with $6,370 and short BML.AX with $4,118. The weights are asymmetric because of the hedge ratio — $0.65 of A1M for every $1 of BML keeps the pair market-neutral.

### Exit / Stop Signals

Below the entry table, an expandable section shows pairs where the Z-score has reverted to within the exit band (|z| < 0.5) or blown past the stop threshold. These are positions you should be closing if you currently hold them.

### "No actionable entry signals"

If you see this message, it means either:
- No pairs have Z-scores beyond the entry threshold (market is tightly priced)
- All valid pairs are in the exit zone (spreads have already reverted)

This is normal and expected — the system deliberately waits for statistical extremes. Not every day produces trades.

---

## Z-Score Chart

The large chart on the left side of the visualisation row. It shows the rolling Z-score history for whichever pair you selected in the sidebar dropdown.

### Chart Elements

**White line**: The Z-score over the last ~252 trading days (1 year). This is the spread between the two tickers, normalised by its rolling mean and standard deviation.

**Cyan dashed lines (Entry bands)**: These shift based on your risk mode.
- Conservative: +/- 2.5
- Aggressive: +/- 1.8
When the white line crosses outside these bands, a trade signal fires.

**Lime dotted lines (Exit bands)**: Fixed at +/- 0.5. When the white line returns inside these bands, the position should be closed (mean-reversion complete, profit taken).

**Red solid lines (Stop bands)**: These also shift by mode.
- Conservative: +/- 3.0
- Aggressive: +/- 4.0
If the Z-score blows past these, exit immediately — the cointegration relationship may be breaking down.

**Orange shaded zones**: Between entry and stop bands. This is the "active trade" zone — positions are open but not yet at emergency levels.

**Red shaded zones**: Beyond the stop bands. Danger territory.

### Chart Title Metadata

The title shows: `TICKER_A / TICKER_B (HL=Xd, p=0.XXXX)`
- **HL**: Half-life in days — how quickly the spread is expected to revert to its mean. Lower = faster reversion = better for swing trading. The system accepts 5–120 days.
- **p**: Cointegration p-value from the Engle-Granger test. Lower = stronger statistical evidence that these two assets move together. The system requires < 0.05.

### What to look for

- **Z-score approaching entry band**: A trade may fire soon. Prepare execution.
- **Z-score inside exit band**: If you hold this pair, close it — the spread has reverted.
- **Z-score trending steadily in one direction**: The cointegration relationship may be weakening. Be cautious even if the Z-score hasn't hit the stop.
- **Frequent oscillation between entry bands**: Ideal pairs trading behavior — the spread is actively mean-reverting.

---

## Market Regime History

The chart on the right side of the visualisation row. Two panels stacked vertically.

### Top Panel — Posterior Probabilities (Stacked Area)

Shows the last ~120 trading days of the HMM's probability estimates for each regime:
- **Red area (BEAR)**: Probability the market is in a bearish/declining regime
- **Orange area (CHOP)**: Probability the market is range-bound/mean-reverting
- **Green area (BULL)**: Probability the market is in a bullish/trending-up regime

The three areas always sum to 100% on any given day. When one color dominates, the model is confident. When colors are mixed, the model is uncertain (regime transition may be underway).

**What to look for**:
- Long stretches of orange (CHOP) = ideal environment for pairs trading
- Sudden shift from orange to red or green = regime change, consider reducing exposure
- Gradual color mixing = transition period, be cautious

### Bottom Panel — Regime Classification (Color Bar)

A simplified view showing the discrete regime label assigned to each day:
- Red = BEAR
- Orange = CHOP
- Green = BULL

This makes regime transitions easy to spot at a glance. Look for the boundaries between color blocks — those are the days the model detected a regime shift.

### Regime Transition Matrix (Expandable)

Click to expand. Shows the probability of transitioning from one regime to another on any given day:

```
       CHOP   BULL   BEAR
CHOP  86.7%  12.8%   0.5%
BULL  10.9%  86.5%   2.5%
BEAR   2.1%   2.3%  95.6%
```

Reading: "If today is CHOP, there is an 86.7% chance tomorrow is also CHOP, 12.8% chance it shifts to BULL, and 0.5% chance it shifts to BEAR."

Key insight: BEAR has the highest persistence (95.6%) — once the market enters a bear regime, it tends to stay there. This is why the system aggressively scales down pairs positions during BEAR — you cannot wait for a quick regime change.

---

## Portfolio Risk Summary

Bottom-left table. Aggregates all active positions into portfolio-level risk metrics.

| Metric | Meaning |
|---|---|
| **Gross Exposure** | Sum of the absolute values of all position weights. 22.6% means your total long + short notional equals 22.6% of NAV (before leverage). |
| **Net Exposure** | Sum of signed weights. Close to 0% means the portfolio is approximately market-neutral (longs offset shorts). A large positive or negative number means directional bias. |
| **Effective Leverage** | Gross exposure multiplied by broker leverage (5x). Shows `current / mode_limit`. |
| **Margin Used** | Estimated margin consumed. With 5x leverage, margin requirement is ~20% of gross notional. |
| **Margin Remaining** | 100% minus margin used. Must stay above 15% safety buffer. |
| **Active Pairs** | Number of pairs with entry signals (not counting exits/stops). |
| **Risk Mode** | Confirms which mode is active. |
| **Regime** | Current HMM regime. |
| **Within Limits** | YES/NO — whether all risk constraints are satisfied. If NO, check warnings. |

---

## Regime Detail

Bottom-right table. Deep dive into the HMM output.

| Metric | Meaning |
|---|---|
| **Current Regime** | BULL, BEAR, or CHOP |
| **Confidence** | Posterior probability of the current state |
| **21-Day Stability** | % of last 21 days with the same label (regime persistence) |
| **Favorable for Pairs** | Yes only if CHOP — the regime where mean-reversion strategies work best |
| **P(BULL), P(BEAR), P(CHOP)** | Full posterior distribution across all three states |
| **Mean Return (BULL/BEAR/CHOP)** | The average daily return associated with each regime. Positive for BULL, negative for BEAR, near-zero for CHOP. These are learned by the HMM from the data. |

---

## Expandable Sections

### All Cointegrated Pairs

Lists every pair that passed the cointegration filter with its p-value, hedge ratio, and half-life. Use this to understand the full opportunity set, not just today's signals. A pair can be cointegrated (valid relationship) without having an actionable Z-score today.

### All Signals (including FLAT)

Shows every signal for every valid pair, including FLAT (no action). Useful for seeing which pairs are close to entry thresholds — tomorrow's trades often come from today's FLAT signals with Z-scores near the entry band.

---

## How the System Makes Decisions

The pipeline runs in this order:

### 1. Data Ingestion (data_loader.py)
Downloads daily close prices for all tickers. Normalises timezones across ASX (Sydney), US (New York), and LSE (London) exchanges. Forward-fills gaps up to 3 business days (tightened from the standard 5 because of the 5x leverage risk — stale data with leverage is dangerous). Drops tickers with >50% missing data.

### 2. Regime Detection (hmm_engine.py)
Builds four cross-sectional features from the price data:
- **Market return**: Average daily return across all tickers (direction)
- **Market volatility**: Average realised vol across all tickers (stress)
- **Return dispersion**: Standard deviation of returns across tickers (convergence/divergence)
- **Vol acceleration**: Rate of change of volatility (early warning of regime shifts)

Feeds these into a 3-state Gaussian HMM trained on the last ~504 trading days. The states are labelled BEAR/CHOP/BULL by sorting their mean returns.

### 3. Cointegration Scanning (pairs_engine.py)
Tests ~1,350 candidate pairs (ASX vs ASX, ASX vs Commodity, Global vs Commodity) using the Engle-Granger two-step test. Pairs that pass (p < 0.05) get their half-life estimated via AR(1) regression. Pairs with half-lives outside 5–120 days are rejected (too fast = noise, too slow = unusable for swing trading).

### 4. Z-Score & Signal Generation (pairs_engine.py)
For each valid pair, computes a rolling 252-day OLS hedge ratio and dynamic Z-score. Entry signals fire when |Z| exceeds the mode's entry threshold. Exit signals fire when |Z| drops below 0.5. Stop signals fire when |Z| exceeds the mode's stop threshold.

### 5. Position Sizing (risk_manager.py)
This is where the regime and risk mode have their biggest impact.

**Volatility-target sizing**: `vol_scalar = vol_target / current_volatility`. High vol = smaller positions. Low vol = larger positions. This is the core mechanism that prevents oversized bets during turbulent markets.

**Regime confidence scaling**: Multiplied on top of the vol scalar.
- CHOP regime with high confidence → scalar near 1.0 (full sizing)
- BULL/BEAR regime with high confidence → scalar drops to 0.1–0.4 (heavy reduction)
- No regime data → scalar 0.5 (cautious default)

**Position limits**: No single pair can exceed 20% of NAV. Total gross exposure capped at 100% of NAV. Margin must stay above 15% buffer.

---

## Daily Workflow

### Morning Routine (Before Market Open)

1. **Open the dashboard** at http://localhost:8501
2. **Click "Re-download Data"** in the sidebar to get the latest close prices
3. **Check the top row**:
   - What regime are we in? Has it changed from yesterday?
   - What is the margin utilisation? Any warnings?
4. **Review the Actionable Signals table**:
   - Are there new entry signals?
   - Are any existing positions showing EXIT signals?
5. **Check the Z-Score chart** for pairs you currently hold — is the spread reverting or diverging?
6. **Toggle to Aggressive** to see if additional signals appear that may warrant manual review, then toggle back to your operating mode

### Execution Checklist

For each entry signal you decide to act on:
1. Note the Long ticker, Short ticker, and recommended notional amounts
2. Note the stop-loss prices for both legs
3. Execute the trades with your broker (this is a signal-generation tool, not an auto-execution bot)
4. Record the entry Z-score and date for your own tracking
5. Monitor daily — when the Z-score reverts to the exit band, close both legs

### Position Monitoring

For existing positions:
- Check if they appear in the "Exit / Stop Signals" section
- If a pair shows EXIT: the spread has reverted — take profit
- If a pair shows STOP: the spread has blown out — cut the loss immediately
- If the regime shifts from CHOP to BULL/BEAR: consider tightening stops on all positions

---

## Understanding the Current Results

As of the latest run, here is what the system is showing and why:

### Regime: BEAR (100% confidence, 100% stability)

The HMM has classified the current market as BEAR with maximum confidence. The 21-day stability is 100%, meaning every single day for the past month has been BEAR. The transition matrix shows BEAR has 95.6% persistence — once entered, it tends to stay.

**Impact**: The risk manager is aggressively scaling down all position sizes. In Conservative mode, weights are tiny (0.03%–0.7% per leg). In Aggressive mode, weights are larger but still capped (0.6%–1.4% per leg). This is the system protecting you from oversized pairs bets in a trending market.

### Conservative Mode: 4 Entry Signals

Only 4 pairs have Z-scores beyond the Conservative entry threshold of 2.5. The strongest is A1M.AX/BML.AX at z=-2.84. Total gross exposure is just 2.1% with 0.1x effective leverage and 0.4% margin used. This is deliberately ultra-conservative given the BEAR regime.

### Aggressive Mode: 18 Entry Signals

With the lower entry threshold of 1.8, 18 pairs qualify. Total gross exposure rises to 22.6% with 1.1x effective leverage and 4.5% margin used. Still well within the 5x limit because the vol-target sizing and regime scalar are both damping the positions.

### Why positions are so small

Three factors compound:
1. **BEAR regime** → regime confidence scalar drops to ~0.2 (80% reduction)
2. **Elevated volatility** → vol-target scalar further reduces sizing
3. **Conservative mode** → lower vol target (10%) means even smaller base sizing

This is exactly the behavior the system is designed for — **protecting capital during adverse regimes** while still surfacing opportunities for review.

### The warning

"Trending regime (BEAR) with 100% confidence — pairs positions carry elevated risk"

This is informational. It means: pairs trading historically underperforms during trending markets because spreads tend to widen rather than revert. The system has not blocked the trades — it has reduced their size. The final execution decision is yours.

---

## Configuration Reference

All tunable parameters are in `config.py`. Here are the most important ones:

### Ticker Mappings
```python
COMMODITIES = {
    "Gold": "GC=F",
    "Lithium": "LIT",
    ...
}
```
To swap broker-specific CFD instruments, change the ticker values here. No other file needs editing.

### Risk Parameters
```python
FORWARD_FILL_LIMIT = 3        # Max days to carry forward stale prices
MAX_LEVERAGE = 5.0             # Broker leverage cap
MARGIN_BUFFER_PCT = 0.15       # 15% safety buffer
MAX_POSITION_PCT = 0.20        # Max 20% NAV in one pair
MAX_TOTAL_EXPOSURE_PCT = 1.0   # Max 100% gross before leverage
```

### Pairs Engine
```python
DEFAULT_COINT_PVALUE = 0.05    # Cointegration significance
MIN_HALF_LIFE = 5              # Reject pairs < 5 day reversion
MAX_HALF_LIFE = 120            # Reject pairs > 120 day reversion
DEFAULT_LOOKBACK = 252         # Rolling window for Z-score
```

### HMM
```python
HMM_N_REGIMES = 3             # BULL / BEAR / CHOP
HMM_TRAINING_WINDOW = 504     # ~2 years of training data
```

---

## Glossary

| Term | Definition |
|---|---|
| **Cointegration** | A statistical relationship where two price series, while individually non-stationary, have a linear combination that is stationary (mean-reverting). Stronger than correlation. |
| **Engle-Granger test** | A two-step test for cointegration. Step 1: regress Y on X. Step 2: test the residuals for stationarity (ADF test). |
| **Hedge ratio** | The OLS beta from regressing asset A on asset B. Determines how many dollars of B to trade for each dollar of A to create a market-neutral pair. |
| **Half-life** | The expected number of days for the spread to revert halfway to its mean. Estimated from an AR(1) model. |
| **Z-score** | Number of standard deviations the current spread is from its rolling mean. Z=0 means at the mean. Z=+2 means 2 std devs above (spread is rich). |
| **ATR** | Average True Range — a measure of daily price volatility used to set stop-loss distances. |
| **NAV** | Net Asset Value — total portfolio value. |
| **Gross exposure** | Sum of absolute position values. $50k long + $50k short = $100k gross. |
| **Net exposure** | Sum of signed position values. $50k long + $50k short = $0 net (market-neutral). |
| **Margin utilisation** | Percentage of NAV required as collateral for leveraged positions. |
| **HMM** | Hidden Markov Model — a statistical model that infers unobserved "hidden" states (regimes) from observed data (returns, volatility). |
| **Posterior probability** | The model's belief about which state the market is currently in, given all observed data up to today. |
| **Vol-target sizing** | Position sizing method where size = target_volatility / current_volatility. Larger positions when calm, smaller when volatile. |

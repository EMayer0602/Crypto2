"""
Analyze how many entry signals are blocked by different filters.
Compare entries with and without JMA Trend Filter.
"""
import pandas as pd
import Supertrend_5Min as st

# Test symbol
SYMBOL = "ETH/EUR"
INDICATOR = "jma"
HTF = "6h"
PARAM_A = 20
PARAM_B = 50

print(f"\n{'='*60}")
print(f"Filter Impact Analysis: {SYMBOL} ({INDICATOR}, HTF={HTF})")
print(f"{'='*60}\n")

# Fetch data
df = st.fetch_data(SYMBOL, st.TIMEFRAME, st.LOOKBACK)
df = st.attach_higher_timeframe_trend(df, SYMBOL)
df = st.attach_momentum_filter(df)
df = st.attach_jma_trend_filter(df)

# Compute indicator
df = st.compute_indicator(df, PARAM_A, PARAM_B)

# Count trend flips (raw entry signals)
raw_signals = 0
for i in range(1, len(df)):
    trend = int(df["trend_flag"].iloc[i])
    prev_trend = int(df["trend_flag"].iloc[i - 1])
    if prev_trend == -1 and trend == 1:  # Long entry signal
        raw_signals += 1

print(f"1. Raw Entry Signals (Indicator only): {raw_signals}")

# Count with HTF filter
htf_allowed = 0
for i in range(1, len(df)):
    trend = int(df["trend_flag"].iloc[i])
    prev_trend = int(df["trend_flag"].iloc[i - 1])
    if prev_trend == -1 and trend == 1:
        htf_value = int(df["htf_trend"].iloc[i])
        if htf_value >= 1:  # HTF allows long
            htf_allowed += 1

print(f"2. After HTF Filter: {htf_allowed} (blocked: {raw_signals - htf_allowed})")

# Count with HTF + JMA filter
jma_allowed = 0
for i in range(1, len(df)):
    trend = int(df["trend_flag"].iloc[i])
    prev_trend = int(df["trend_flag"].iloc[i - 1])
    if prev_trend == -1 and trend == 1:
        htf_value = int(df["htf_trend"].iloc[i])
        jma_trend = df["jma_trend_direction"].iloc[i]
        if htf_value >= 1 and jma_trend == "UP":
            jma_allowed += 1

print(f"3. After HTF + JMA Filter: {jma_allowed} (blocked: {htf_allowed - jma_allowed})")

# Analyze JMA trend distribution
jma_trends = df["jma_trend_direction"].value_counts()
print(f"\n{'='*60}")
print("JMA Trend Distribution:")
print(f"{'='*60}")
for trend, count in jma_trends.items():
    pct = (count / len(df)) * 100
    print(f"  {trend:>6}: {count:>5} bars ({pct:>5.1f}%)")

# Show JMA slope statistics
print(f"\n{'='*60}")
print("JMA Slope Statistics:")
print(f"{'='*60}")
slope = df["jma_trend_slope"].dropna()
print(f"  Mean:   {slope.mean():.8f}")
print(f"  Median: {slope.median():.8f}")
print(f"  Std:    {slope.std():.8f}")
print(f"  Min:    {slope.min():.8f}")
print(f"  Max:    {slope.max():.8f}")

print(f"\n{'='*60}")
print("Current Thresholds:")
print(f"{'='*60}")
print(f"  THRESH_UP:   {st.JMA_TREND_THRESH_UP:.8f}")
print(f"  THRESH_DOWN: {st.JMA_TREND_THRESH_DOWN:.8f}")

# Suggest better thresholds
suggested_up = slope.quantile(0.6)  # 60th percentile
suggested_down = slope.quantile(0.4)  # 40th percentile
print(f"\n{'='*60}")
print("Suggested Thresholds (60/40 percentile):")
print(f"{'='*60}")
print(f"  JMA_TREND_THRESH_UP   = {suggested_up:.8f}")
print(f"  JMA_TREND_THRESH_DOWN = {suggested_down:.8f}")
print(f"\nThis would classify ~40% as UP, ~40% as DOWN, ~20% as FLAT")

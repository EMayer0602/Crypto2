#!/usr/bin/env python3
"""
Update OHLCV cache with the latest data from Binance (today's data).

Usage:
    python update_latest_data.py
"""

import os
import sys
import time
import pandas as pd
import Supertrend_5Min as st

# Symbols and timeframes to update
SYMBOLS = [
    "BTC/EUR", "ETH/EUR", "XRP/EUR", "LINK/EUR",
    "LUNC/USDT", "SOL/EUR", "SUI/EUR", "TNSR/USDC", "ZEC/USDC"
]

TIMEFRAMES = ["1h", "2h", "3h", "4h", "5h", "6h", "8h", "12h", "23h", "24h"]

def update_latest_data():
    """Update cache with latest data from Binance."""

    # Configure exchange
    st.configure_exchange(use_testnet=False)

    # Create cache directory
    os.makedirs(st.OHLCV_CACHE_DIR, exist_ok=True)

    # Calculate date range: last 3 days to now (to ensure overlap)
    end_date = pd.Timestamp.now(tz=st.BERLIN_TZ)
    start_date = end_date - pd.Timedelta(days=3)

    print(f"\n{'='*70}")
    print(f"UPDATING LATEST DATA")
    print(f"Period: {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"Symbols: {len(SYMBOLS)}")
    print(f"Timeframes: {len(TIMEFRAMES)}")
    print(f"{'='*70}\n")

    total = len(SYMBOLS) * len(TIMEFRAMES)
    count = 0
    updated_count = 0

    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            count += 1
            print(f"[{count}/{total}] {symbol} {timeframe}...", end=" ")

            try:
                # Download latest data
                df = st.download_historical_ohlcv(symbol, timeframe, start_date, end_date)

                if not df.empty:
                    latest = df.index[-1].strftime('%Y-%m-%d %H:%M')
                    print(f"✓ Updated to {latest} ({len(df)} bars)")
                    updated_count += 1
                else:
                    print("⚠ No data")

                # Small delay to avoid rate limits
                time.sleep(0.2)

            except Exception as exc:
                print(f"✗ Error: {exc}")
                continue

    print(f"\n{'='*70}")
    print(f"UPDATE COMPLETE!")
    print(f"Updated: {updated_count}/{total} symbol/timeframe combinations")
    print(f"Cache location: {st.OHLCV_CACHE_DIR}/")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    update_latest_data()

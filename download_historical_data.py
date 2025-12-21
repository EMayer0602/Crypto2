#!/usr/bin/env python3
"""
Download historical OHLCV data from Binance and save to CSV files.

Usage:
    python download_historical_data.py --start 2025-09-01 --end 2025-12-21
"""

import os
import sys
import time
import argparse
import pandas as pd
import Supertrend_5Min as st

# Symbols and timeframes to download
SYMBOLS = [
    "BTC/EUR", "ETH/EUR", "XRP/EUR", "LINK/EUR",
    "LUNC/USDT", "SOL/EUR", "SUI/EUR", "TNSR/USDC", "ZEC/USDC"
]

TIMEFRAMES = ["1h", "3h", "4h", "6h", "8h", "12h", "23h", "24h"]

def download_all_data(start_date, end_date):
    """Download historical data for all symbols and timeframes."""

    # Create cache directory
    os.makedirs(st.OHLCV_CACHE_DIR, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"DOWNLOADING HISTORICAL DATA")
    print(f"Period: {start_date} to {end_date}")
    print(f"Symbols: {len(SYMBOLS)}")
    print(f"Timeframes: {len(TIMEFRAMES)}")
    print(f"Total downloads: {len(SYMBOLS) * len(TIMEFRAMES)}")
    print(f"{'='*70}\n")

    total = len(SYMBOLS) * len(TIMEFRAMES)
    count = 0

    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            count += 1
            print(f"\n[{count}/{total}] {symbol} {timeframe}")
            print("-" * 70)

            try:
                # Download data
                df = st.download_historical_ohlcv(symbol, timeframe, start_date, end_date)

                if df.empty:
                    print(f"⚠️  No data received for {symbol} {timeframe}")
                else:
                    print(f"✓ Successfully downloaded {len(df)} bars")

                # Small delay to avoid rate limits
                time.sleep(0.3)

            except Exception as exc:
                print(f"✗ ERROR downloading {symbol} {timeframe}: {exc}")
                continue

    print(f"\n{'='*70}")
    print(f"DOWNLOAD COMPLETE!")
    print(f"Data saved to: {st.OHLCV_CACHE_DIR}/")
    print(f"{'='*70}\n")

    # Show what was downloaded
    print("Downloaded files:")
    if os.path.exists(st.OHLCV_CACHE_DIR):
        files = os.listdir(st.OHLCV_CACHE_DIR)
        for f in sorted(files):
            filepath = os.path.join(st.OHLCV_CACHE_DIR, f)
            size = os.path.getsize(filepath) / 1024  # KB
            print(f"  {f:30s} ({size:>8.1f} KB)")
        print(f"\nTotal files: {len(files)}")
    else:
        print("  (No cache directory found)")


def main():
    parser = argparse.ArgumentParser(description="Download historical OHLCV data from Binance")
    parser.add_argument("--start", type=str, required=True, help="Start date (e.g., 2025-09-01)")
    parser.add_argument("--end", type=str, default=None, help="End date (default: today)")

    args = parser.parse_args()

    # Configure exchange
    st.configure_exchange(use_testnet=False)

    # Parse dates
    start_date = args.start
    end_date = args.end or pd.Timestamp.now(tz=st.BERLIN_TZ).strftime("%Y-%m-%d")

    # Download
    download_all_data(start_date, end_date)

    print("\nYou can now run simulations with:")
    print(f"  python paper_trader.py --simulate --start {args.start} --clear-outputs")


if __name__ == "__main__":
    main()

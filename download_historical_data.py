#!/usr/bin/env python3
"""
Historical Data Downloader for Crypto2 Trading System

Downloads historical OHLCV data from Binance, saves to parquet cache files,
and synthesizes the current bar from 1-minute data.

Usage:
    python download_historical_data.py                    # Update all symbols
    python download_historical_data.py --symbol BTC/EUR   # Update specific symbol
    python download_historical_data.py --since 2024-01-01 # Download from specific date
    python download_historical_data.py --force            # Force re-download all data
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import time

import ccxt
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import pytz

# Load environment variables
load_dotenv()

# Configuration
BERLIN_TZ = pytz.timezone("Europe/Berlin")
CACHE_DIR = Path("ohlcv_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Default symbols from Supertrend_5Min.py
DEFAULT_SYMBOLS = [
    "BTC/EUR",
    "ETH/EUR",
    "XRP/EUR",
    "LINK/EUR",
    "LUNC/USDT",
    "SOL/EUR",
    "SUI/EUR",
    "TNSR/USDC",
    "ZEC/USDC",
]

# Timeframes to cache
TIMEFRAMES = ["1h", "4h", "6h", "8h", "12h"]

# Default start date for historical data (2 years back)
DEFAULT_SINCE = datetime(2023, 1, 1, tzinfo=pytz.UTC)


def get_exchange():
    """Create Binance exchange instance."""
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {
            "warnOnFetchCurrenciesWithoutPermission": False,
        }
    })
    # Disable currency fetching to avoid permission errors
    if hasattr(exchange, "has") and isinstance(exchange.has, dict):
        exchange.has["fetchCurrencies"] = False
    return exchange


def cache_path(symbol: str, timeframe: str) -> Path:
    """Generate cache file path for symbol/timeframe."""
    symbol_clean = symbol.replace("/", "_")
    return CACHE_DIR / f"{symbol_clean}_{timeframe}.parquet"


def load_cached_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load cached data from parquet file."""
    path = cache_path(symbol, timeframe)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        if not df.empty and df.index.tz is None:
            df.index = df.index.tz_localize(pytz.UTC).tz_convert(BERLIN_TZ)
        return df
    except Exception as exc:
        print(f"[Warn] Failed to load cache {path}: {exc}")
        return pd.DataFrame()


def save_cached_data(symbol: str, timeframe: str, df: pd.DataFrame) -> None:
    """Save data to parquet cache file."""
    if df.empty:
        return
    path = cache_path(symbol, timeframe)
    try:
        # Ensure index is timezone-aware
        if df.index.tz is None:
            df.index = df.index.tz_localize(pytz.UTC)
        df.to_parquet(path)
        print(f"[Cache] Saved {len(df)} bars for {symbol} {timeframe} to {path}")
    except Exception as exc:
        print(f"[Error] Failed to save cache {path}: {exc}")


def fetch_ohlcv_paginated(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since: datetime,
    until: datetime = None,
    limit_per_request: int = 1000,
) -> pd.DataFrame:
    """
    Fetch OHLCV data with pagination to get full historical range.
    """
    if until is None:
        until = datetime.now(pytz.UTC)

    since_ms = int(since.timestamp() * 1000)
    until_ms = int(until.timestamp() * 1000)

    all_data = []
    current_since = since_ms

    # Calculate timeframe in milliseconds for pagination
    tf_map = {
        "1m": 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "6h": 6 * 60 * 60 * 1000,
        "8h": 8 * 60 * 60 * 1000,
        "12h": 12 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
    }
    tf_ms = tf_map.get(timeframe, 60 * 60 * 1000)

    request_count = 0
    max_requests = 500  # Safety limit

    print(f"[Fetch] {symbol} {timeframe} from {since.strftime('%Y-%m-%d')} to {until.strftime('%Y-%m-%d')}")

    while current_since < until_ms and request_count < max_requests:
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=current_since,
                limit=limit_per_request
            )
        except ccxt.BadSymbol as exc:
            print(f"[Skip] Symbol {symbol} not available: {exc}")
            return pd.DataFrame()
        except Exception as exc:
            print(f"[Error] Fetch failed for {symbol} {timeframe}: {exc}")
            time.sleep(2)
            request_count += 1
            continue

        if not ohlcv:
            break

        all_data.extend(ohlcv)

        # Move to next page
        last_ts = ohlcv[-1][0]
        if last_ts <= current_since:
            break
        current_since = last_ts + tf_ms

        request_count += 1

        # Progress indicator
        if request_count % 10 == 0:
            pct = min(100, (current_since - since_ms) / max(1, until_ms - since_ms) * 100)
            print(f"  ... {request_count} requests, {len(all_data)} bars ({pct:.0f}%)")

        # Rate limiting
        time.sleep(0.1)

    if not all_data:
        return pd.DataFrame()

    # Convert to DataFrame
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(all_data, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(BERLIN_TZ)
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    # Filter to requested range
    df = df[(df.index >= since) & (df.index <= until)]

    return df


def fetch_1m_for_synthetic(exchange: ccxt.Exchange, symbol: str, minutes: int = 60) -> pd.DataFrame:
    """Fetch recent 1-minute data for synthetic bar creation."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1m", limit=minutes + 10)
    except Exception as exc:
        print(f"[Warn] Failed to fetch 1m data for {symbol}: {exc}")
        return pd.DataFrame()

    if not ohlcv:
        return pd.DataFrame()

    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(ohlcv, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(BERLIN_TZ)
    df = df.set_index("timestamp")
    return df


def synthesize_current_bar(df: pd.DataFrame, minute_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Synthesize the current (incomplete) bar from 1-minute data.
    """
    if minute_df.empty:
        return df

    # Parse timeframe to minutes
    tf_map = {"1h": 60, "4h": 240, "6h": 360, "8h": 480, "12h": 720, "1d": 1440}
    tf_minutes = tf_map.get(timeframe)
    if not tf_minutes:
        return df

    now = pd.Timestamp.now(BERLIN_TZ)
    bucket = pd.Timedelta(minutes=tf_minutes)
    current_end = now.floor(f"{tf_minutes}min") + bucket
    current_start = current_end - bucket

    # Get 1m bars within current period
    slice_df = minute_df[(minute_df.index > current_start) & (minute_df.index <= now)]
    if slice_df.empty:
        return df

    # Create synthetic bar
    synthetic = pd.DataFrame({
        "open": float(slice_df["open"].iloc[0]),
        "high": float(slice_df["high"].max()),
        "low": float(slice_df["low"].min()),
        "close": float(slice_df["close"].iloc[-1]),
        "volume": float(slice_df["volume"].sum()),
    }, index=[current_end])

    # Remove any existing bar at current_end and append synthetic
    if not df.empty and current_end in df.index:
        df = df.drop(index=current_end)

    combined = pd.concat([df, synthetic])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

    print(f"[Synthetic] Created current bar for {timeframe} using {len(slice_df)} 1m bars (ends {current_end})")

    return combined


def update_symbol_data(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframes: list = None,
    since: datetime = None,
    force: bool = False,
) -> dict:
    """
    Update cached data for a symbol.

    Returns dict with timeframe -> DataFrame mapping.
    """
    if timeframes is None:
        timeframes = TIMEFRAMES
    if since is None:
        since = DEFAULT_SINCE

    results = {}

    # Fetch 1-minute data once for synthetic bar creation
    minute_df = fetch_1m_for_synthetic(exchange, symbol)

    for tf in timeframes:
        print(f"\n[Update] {symbol} {tf}")

        # Load existing cache
        cached = load_cached_data(symbol, tf)

        if force or cached.empty:
            # Full download
            fetch_since = since
        else:
            # Incremental update - fetch from last cached bar
            fetch_since = cached.index.max() - timedelta(hours=2)  # Small overlap

        # Fetch new data
        new_data = fetch_ohlcv_paginated(exchange, symbol, tf, fetch_since)

        if new_data.empty and cached.empty:
            print(f"[Skip] No data available for {symbol} {tf}")
            results[tf] = pd.DataFrame()
            continue

        # Merge cached and new data
        if not cached.empty and not new_data.empty:
            combined = pd.concat([cached, new_data])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
        elif not new_data.empty:
            combined = new_data
        else:
            combined = cached

        # Add synthetic current bar
        combined = synthesize_current_bar(combined, minute_df, tf)

        # Save to cache
        save_cached_data(symbol, tf, combined)

        # Stats
        if not combined.empty:
            start = combined.index.min().strftime("%Y-%m-%d")
            end = combined.index.max().strftime("%Y-%m-%d %H:%M")
            print(f"[Done] {symbol} {tf}: {len(combined)} bars from {start} to {end}")

        results[tf] = combined

    return results


def main():
    parser = argparse.ArgumentParser(description="Download and update historical OHLCV data")
    parser.add_argument("--symbol", "-s", type=str, default=None,
                        help="Specific symbol to update (default: all)")
    parser.add_argument("--since", type=str, default=None,
                        help="Start date for download (YYYY-MM-DD, default: 2023-01-01)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Force re-download all data (ignore cache)")
    parser.add_argument("--timeframe", "-t", type=str, default=None,
                        help="Specific timeframe to update (default: all)")
    args = parser.parse_args()

    # Parse since date
    since = DEFAULT_SINCE
    if args.since:
        since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=pytz.UTC)

    # Get symbols to update
    symbols = [args.symbol] if args.symbol else DEFAULT_SYMBOLS

    # Get timeframes
    timeframes = [args.timeframe] if args.timeframe else TIMEFRAMES

    print("=" * 60)
    print("Historical Data Downloader for Crypto2")
    print("=" * 60)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Timeframes: {', '.join(timeframes)}")
    print(f"Since: {since.strftime('%Y-%m-%d')}")
    print(f"Force: {args.force}")
    print("=" * 60)

    exchange = get_exchange()

    for symbol in symbols:
        print(f"\n{'='*40}")
        print(f"Processing {symbol}")
        print(f"{'='*40}")

        try:
            update_symbol_data(
                exchange,
                symbol,
                timeframes=timeframes,
                since=since,
                force=args.force,
            )
        except Exception as exc:
            print(f"[Error] Failed to update {symbol}: {exc}")
            continue

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

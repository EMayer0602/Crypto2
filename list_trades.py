#!/usr/bin/env python3
"""
List open trades and recent closed trades.

This script displays current open positions and recent trading history.
"""

import json
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

def load_state():
    """Load the current paper trading state."""
    state_file = Path("paper_trading_state.json")
    if state_file.exists():
        with open(state_file, "r") as f:
            return json.load(f)
    return {"positions": [], "total_capital": 14000.0}


def load_trade_log():
    """Load the cumulative trade log."""
    log_file = Path("paper_trading_log.csv")
    if log_file.exists():
        try:
            df = pd.read_csv(log_file)
            return df
        except Exception as exc:
            print(f"Error loading trade log: {exc}")
    return pd.DataFrame()


def format_timestamp(ts_str):
    """Format timestamp to readable string."""
    try:
        if pd.isna(ts_str) or not ts_str:
            return "N/A"
        dt = pd.to_datetime(ts_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return str(ts_str)


def list_open_positions():
    """Display all currently open positions."""
    state = load_state()
    positions = state.get("positions", [])

    print("\n" + "=" * 100)
    print("OPEN POSITIONS")
    print("=" * 100)

    if not positions:
        print("\n✓ No open positions")
        print()
        return

    # Calculate current P&L for each position
    print(f"\nTotal Open: {len(positions)} position(s)")
    print()

    for i, pos in enumerate(positions, 1):
        symbol = pos.get("symbol", "?")
        direction = pos.get("direction", "?")
        entry_price = float(pos.get("entry_price", 0))
        stake = float(pos.get("stake", 0))
        entry_time = format_timestamp(pos.get("entry_time"))
        indicator = pos.get("indicator", "?")
        htf = pos.get("htf", "?")
        param_a = pos.get("param_a", "?")
        param_b = pos.get("param_b", "?")
        atr_mult = pos.get("atr_mult", "None")
        min_hold_bars = pos.get("min_hold_bars", 0)

        print(f"[{i}] {symbol} - {direction.upper()}")
        print(f"    Entry: {entry_price:.8f} @ {entry_time}")
        print(f"    Stake: {stake:.2f} USDT")
        print(f"    Strategy: {indicator} (HTF: {htf}, ParamA: {param_a}, ParamB: {param_b})")
        print(f"    ATR Stop: {atr_mult}, Min Hold: {min_hold_bars} bars")
        print()

    print(f"Total Capital Allocated: {sum(float(p.get('stake', 0)) for p in positions):.2f} USDT")
    print()


def list_recent_trades(days=7, max_trades=50):
    """Display recent closed trades."""
    df = load_trade_log()

    print("\n" + "=" * 100)
    print(f"RECENT CLOSED TRADES (Last {days} days, max {max_trades})")
    print("=" * 100)

    if df.empty:
        print("\n✓ No trade history")
        print()
        return

    # Convert timestamps
    if "exit_time" in df.columns:
        df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce", utc=True)
        df = df.dropna(subset=["exit_time"])

        # Filter by date
        cutoff = pd.Timestamp.now(timezone.utc) - pd.Timedelta(days=days)
        # Ensure both sides are timezone-aware
        if df["exit_time"].dt.tz is None:
            df["exit_time"] = df["exit_time"].dt.tz_localize("UTC")
        recent = df[df["exit_time"] >= cutoff].tail(max_trades)

        if recent.empty:
            print(f"\n✓ No trades in the last {days} days")
            print()
            return

        print(f"\nShowing {len(recent)} trade(s)")
        print()

        # Display summary statistics
        total_pnl = recent["pnl"].sum() if "pnl" in recent.columns else 0
        winners = len(recent[recent["pnl"] > 0]) if "pnl" in recent.columns else 0
        losers = len(recent[recent["pnl"] < 0]) if "pnl" in recent.columns else 0
        win_rate = (winners / len(recent) * 100) if len(recent) > 0 else 0

        print(f"Summary:")
        print(f"  Total P&L: {total_pnl:.2f} USDT")
        print(f"  Winners: {winners}, Losers: {losers}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print()

        # Display trades
        print("-" * 100)

        for _, trade in recent.iterrows():
            symbol = trade.get("symbol", "?")
            direction = trade.get("direction", "?")
            entry_time = format_timestamp(trade.get("entry_time"))
            exit_time = format_timestamp(trade.get("exit_time"))
            entry_price = float(trade.get("entry_price", 0))
            exit_price = float(trade.get("exit_price", 0))
            pnl = float(trade.get("pnl", 0))
            reason = trade.get("reason", "?")
            indicator = trade.get("indicator", "?")
            htf = trade.get("htf", "?")

            status = "✓ WIN" if pnl > 0 else "✗ LOSS" if pnl < 0 else "= FLAT"
            print(f"{status:8s} | {symbol:12s} | {direction:5s} | {indicator:10s} ({htf})")
            print(f"         | Entry: {entry_price:.8f} @ {entry_time}")
            print(f"         | Exit:  {exit_price:.8f} @ {exit_time}")
            print(f"         | P&L: {pnl:+.2f} USDT | Reason: {reason}")
            print("-" * 100)

        print()


def display_account_summary():
    """Display overall account status."""
    state = load_state()
    df = load_trade_log()

    print("\n" + "=" * 100)
    print("ACCOUNT SUMMARY")
    print("=" * 100)

    base_capital = float(state.get("total_capital", 14000.0))
    positions = state.get("positions", [])
    allocated = sum(float(p.get("stake", 0)) for p in positions)
    available = base_capital - allocated

    print(f"\nBase Capital: {base_capital:.2f} USDT")
    print(f"Allocated:    {allocated:.2f} USDT ({len(positions)} position(s))")
    print(f"Available:    {available:.2f} USDT")

    if not df.empty and "pnl" in df.columns:
        all_time_pnl = df["pnl"].sum()
        total_trades = len(df)
        winners = len(df[df["pnl"] > 0])
        losers = len(df[df["pnl"] < 0])
        win_rate = (winners / total_trades * 100) if total_trades > 0 else 0

        print(f"\nAll-Time Performance:")
        print(f"  Total Trades: {total_trades}")
        print(f"  Cumulative P&L: {all_time_pnl:+.2f} USDT")
        print(f"  Win Rate: {win_rate:.1f}% ({winners}W / {losers}L)")
        if total_trades > 0:
            print(f"  Avg P&L per Trade: {all_time_pnl/total_trades:+.2f} USDT")

    print()


def filter_trades_from_date(start_date="2025-12-09"):
    """Filter and display trades from a specific date onwards."""
    df = load_trade_log()

    print("\n" + "=" * 100)
    print(f"TRADES FROM {start_date} ONWARDS")
    print("=" * 100)

    if df.empty:
        print("\n✓ No trade history")
        print()
        return

    if "entry_time" in df.columns:
        df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce", utc=True)
        df = df.dropna(subset=["entry_time"])

        cutoff = pd.to_datetime(start_date, utc=True)
        # Ensure both sides are timezone-aware
        if df["entry_time"].dt.tz is None:
            df["entry_time"] = df["entry_time"].dt.tz_localize("UTC")
        filtered = df[df["entry_time"] >= cutoff]

        if filtered.empty:
            print(f"\n✓ No trades since {start_date}")
            print()
            return

        print(f"\nFound {len(filtered)} trade(s) since {start_date}")
        print()

        # Summary
        total_pnl = filtered["pnl"].sum() if "pnl" in filtered.columns else 0
        print(f"Cumulative P&L: {total_pnl:+.2f} USDT")
        print()

        # Save to CSV
        output_file = f"trades_from_{start_date.replace('-', '')}.csv"
        filtered.to_csv(output_file, index=False)
        print(f"✓ Exported to: {output_file}")
        print()

        # Display trades
        display_cols = [
            "symbol", "direction", "indicator", "htf",
            "entry_time", "entry_price", "exit_time", "exit_price",
            "stake", "pnl", "reason"
        ]
        display_cols = [c for c in display_cols if c in filtered.columns]

        print(filtered[display_cols].to_string(index=False))
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="List trades and positions")
    parser.add_argument("--open", action="store_true", help="Show open positions")
    parser.add_argument("--recent", type=int, default=7, help="Show recent trades (days)")
    parser.add_argument("--from-date", type=str, help="Show trades from date (YYYY-MM-DD)")
    parser.add_argument("--summary", action="store_true", help="Show account summary")
    parser.add_argument("--all", action="store_true", help="Show everything")

    args = parser.parse_args()

    if args.all:
        display_account_summary()
        list_open_positions()
        list_recent_trades(days=args.recent)
    elif args.from_date:
        filter_trades_from_date(args.from_date)
    elif args.open:
        list_open_positions()
    elif args.summary:
        display_account_summary()
    else:
        # Default: show everything
        display_account_summary()
        list_open_positions()
        list_recent_trades(days=args.recent)

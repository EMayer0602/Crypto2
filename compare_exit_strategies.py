"""
Compare Exit Strategies - Old vs New

This script compares the old exit strategy with the new improved strategy
by simulating both on the same historical data and showing the differences.
"""

import pandas as pd
import numpy as np
import sys
import os
from typing import List, Dict, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import Supertrend_5Min as st
import paper_trader as pt
import improved_exit_strategy as improved


def simulate_with_old_strategy(
    df: pd.DataFrame,
    position: Dict,
    atr_mult: float,
    min_hold_bars: int
) -> Tuple[Dict, List[Dict]]:
    """
    Simulate using the OLD exit strategy.
    Returns: (final_position, exit_history)
    """
    exit_history = []
    current_pos = position.copy()

    for i in range(len(df)):
        if i < 2:
            continue

        df_slice = df.iloc[:i+1]

        # Use old evaluate_exit
        exit_result = pt.evaluate_exit(current_pos, df_slice, atr_mult, min_hold_bars)

        if exit_result:
            exit_history.append({
                "bar_index": i,
                "timestamp": df.index[i],
                "exit_price": exit_result["exit_price"],
                "reason": exit_result["reason"],
                "pnl": exit_result["pnl"],
                "bars_held": i - 0  # Simplified
            })
            break

    return current_pos, exit_history


def simulate_with_new_strategy(
    df: pd.DataFrame,
    position: Dict,
    atr_mult: float,
    min_hold_bars: int
) -> Tuple[Dict, List[Dict]]:
    """
    Simulate using the NEW exit strategy.
    Returns: (final_position, exit_history)
    """
    exit_history = []
    current_pos = position.copy()

    for i in range(len(df)):
        if i < 2:
            continue

        df_slice = df.iloc[:i+1]

        # Use new evaluate_exit_improved
        exit_result = improved.evaluate_exit_improved(current_pos, df_slice, atr_mult, min_hold_bars)

        if exit_result:
            exit_history.append({
                "bar_index": i,
                "timestamp": df.index[i],
                "exit_price": exit_result["exit_price"],
                "reason": exit_result["reason"],
                "pnl": exit_result["pnl"],
                "regime": exit_result.get("regime", "unknown"),
                "bars_held": exit_result.get("bars_held", i)
            })
            break

    return current_pos, exit_history


def compare_single_trade(
    symbol: str,
    direction: str,
    entry_price: float,
    entry_time: pd.Timestamp,
    stake: float = 2000.0,
    atr_mult: float = 1.5,
    min_hold_days: int = 2
) -> Dict:
    """
    Compare both strategies on a single trade setup.
    """
    print(f"\n{'='*80}")
    print(f"Testing: {symbol} {direction.upper()} @ {entry_price:.8f} on {entry_time}")
    print(f"{'='*80}")

    # Load data for this symbol
    try:
        lookback = 500  # ~1.7 days of 5min bars
        df = st.fetch_data(symbol, st.TIMEFRAME, lookback)

        # Build indicator dataframe
        # Determine which indicator/htf to use (simplified - use first available)
        best_df = pt.load_best_rows()
        symbol_rows = best_df[best_df["Symbol"] == symbol]

        if symbol_rows.empty:
            print(f"[SKIP] No strategy config found for {symbol}")
            return None

        row = symbol_rows.iloc[0]
        indicator = row["Indicator"]
        htf = row["HTF"]

        # Build context and dataframe
        from paper_trader import build_strategy_context, build_indicator_dataframe
        context = build_strategy_context(row)
        df_full = build_indicator_dataframe(symbol, indicator, htf, context.param_a, context.param_b)

        if df_full is None or df_full.empty:
            print(f"[SKIP] Could not build indicator dataframe for {symbol}")
            return None

        # Find entry bar
        entry_bar_idx = None
        for i, ts in enumerate(df_full.index):
            if ts >= entry_time:
                entry_bar_idx = i
                break

        if entry_bar_idx is None or entry_bar_idx < 2:
            print(f"[SKIP] Entry time not found in dataframe")
            return None

        # Get ATR at entry
        entry_bar = df_full.iloc[entry_bar_idx]
        entry_atr = float(entry_bar["atr"]) if "atr" in entry_bar and pd.notna(entry_bar["atr"]) else 0.0

        # Create position dict
        position = {
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "entry_time": entry_time,
            "entry_atr": entry_atr,
            "stake": stake,
        }

        # Get data from entry onwards
        df_after_entry = df_full.iloc[entry_bar_idx:]

        if len(df_after_entry) < 10:
            print(f"[SKIP] Not enough data after entry")
            return None

        # Convert min_hold_days to bars (5-min bars)
        bars_per_day = 288  # 24h * 60min / 5min
        min_hold_bars = min_hold_days * bars_per_day

        # Simulate both strategies
        print("\n[OLD STRATEGY]")
        _, old_exits = simulate_with_old_strategy(df_after_entry, position, atr_mult, min_hold_bars)

        print("\n[NEW STRATEGY]")
        _, new_exits = simulate_with_new_strategy(df_after_entry, position, atr_mult, min_hold_bars)

        # Compare results
        old_exit = old_exits[0] if old_exits else None
        new_exit = new_exits[0] if new_exits else None

        result = {
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "entry_time": entry_time,
            "stake": stake,
            "old_exit": old_exit,
            "new_exit": new_exit,
        }

        # Print comparison
        print(f"\n{'─'*80}")
        print("COMPARISON:")
        print(f"{'─'*80}")

        if old_exit:
            print(f"OLD: Exit @ {old_exit['exit_price']:.8f} after {old_exit['bars_held']} bars")
            print(f"     Reason: {old_exit['reason']}")
            print(f"     PnL: {old_exit['pnl']:.2f} USDT")
        else:
            print(f"OLD: NO EXIT (position still open)")

        print()

        if new_exit:
            print(f"NEW: Exit @ {new_exit['exit_price']:.8f} after {new_exit['bars_held']} bars")
            print(f"     Reason: {new_exit['reason']}")
            print(f"     Regime: {new_exit.get('regime', 'unknown')}")
            print(f"     PnL: {new_exit['pnl']:.2f} USDT")
        else:
            print(f"NEW: NO EXIT (position still open)")

        # Calculate difference
        if old_exit and new_exit:
            pnl_diff = new_exit['pnl'] - old_exit['pnl']
            bars_diff = new_exit['bars_held'] - old_exit['bars_held']

            print(f"\n{'─'*80}")
            print(f"DIFFERENCE:")
            print(f"  PnL Improvement: {pnl_diff:+.2f} USDT ({(pnl_diff/abs(old_exit['pnl'])*100):+.1f}%)")
            print(f"  Holding Time Difference: {bars_diff:+d} bars")

            if pnl_diff > 0:
                print(f"  ✓ NEW STRATEGY BETTER by {pnl_diff:.2f} USDT")
            elif pnl_diff < 0:
                print(f"  ✗ OLD STRATEGY BETTER by {abs(pnl_diff):.2f} USDT")
            else:
                print(f"  = SAME RESULT")

        return result

    except Exception as exc:
        print(f"[ERROR] {exc}")
        import traceback
        traceback.print_exc()
        return None


def run_batch_comparison():
    """
    Compare both strategies on multiple recent trades.
    """
    print("="*80)
    print("EXIT STRATEGY COMPARISON - OLD vs NEW")
    print("="*80)

    # Load recent closed trades
    log_file = "paper_trading_log.csv"

    if not os.path.exists(log_file):
        print(f"\n[ERROR] Trade log not found: {log_file}")
        print("Please run a simulation first to generate trades.")
        return

    df = pd.read_csv(log_file)

    if df.empty:
        print(f"\n[ERROR] No trades found in {log_file}")
        return

    # Load trades (handle both uppercase and lowercase column names)
    trades_df = pt._load_trade_log_dataframe(log_file)

    if trades_df.empty:
        print(f"\n[ERROR] Could not load trades from {log_file}")
        return

    print(f"\nFound {len(trades_df)} trades in history")
    print(f"Testing last 5 trades...\n")

    # Take last 5 trades
    recent_trades = trades_df.tail(5)

    results = []

    for idx, trade in recent_trades.iterrows():
        result = compare_single_trade(
            symbol=trade["symbol"],
            direction=trade["direction"],
            entry_price=float(trade["entry_price"]),
            entry_time=pd.to_datetime(trade["entry_time"]),
            stake=float(trade["stake"]),
            atr_mult=1.5,  # Default
            min_hold_days=2  # Default
        )

        if result:
            results.append(result)

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    if not results:
        print("No valid comparisons completed.")
        return

    old_pnls = [r["old_exit"]["pnl"] for r in results if r["old_exit"]]
    new_pnls = [r["new_exit"]["pnl"] for r in results if r["new_exit"]]

    if old_pnls and new_pnls:
        old_total = sum(old_pnls)
        new_total = sum(new_pnls)
        improvement = new_total - old_total

        print(f"\nTotal PnL Comparison:")
        print(f"  OLD Strategy: {old_total:.2f} USDT")
        print(f"  NEW Strategy: {new_total:.2f} USDT")
        print(f"  Improvement: {improvement:+.2f} USDT ({(improvement/abs(old_total)*100):+.1f}%)")

        old_avg = np.mean(old_pnls)
        new_avg = np.mean(new_pnls)

        print(f"\nAverage PnL per Trade:")
        print(f"  OLD Strategy: {old_avg:.2f} USDT")
        print(f"  NEW Strategy: {new_avg:.2f} USDT")
        print(f"  Improvement: {(new_avg - old_avg):+.2f} USDT")

        old_winners = sum(1 for p in old_pnls if p > 0)
        new_winners = sum(1 for p in new_pnls if p > 0)

        print(f"\nWin Rate:")
        print(f"  OLD Strategy: {old_winners}/{len(old_pnls)} ({old_winners/len(old_pnls)*100:.1f}%)")
        print(f"  NEW Strategy: {new_winners}/{len(new_pnls)} ({new_winners/len(new_pnls)*100:.1f}%)")

        if new_total > old_total:
            print(f"\n✓✓✓ NEW STRATEGY WINS by {improvement:.2f} USDT! ✓✓✓")
        elif new_total < old_total:
            print(f"\n✗✗✗ OLD STRATEGY BETTER by {abs(improvement):.2f} USDT")
        else:
            print(f"\n= STRATEGIES TIED")


if __name__ == "__main__":
    run_batch_comparison()

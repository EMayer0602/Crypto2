"""
Compare Exit Strategies - Three-Way Comparison

This script compares THREE exit strategies:
1. OLD: ATR stops + Trend flip (original strategy)
2. NEW: Regime-based adaptive exits (market regime detection)
3. HTF: Pure HTF crossover (simple crossover-based strategy)

All strategies are tested on the same historical data to determine which performs best.
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
import htf_crossover_strategy as htf_simple


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
    Simulate using the NEW exit strategy (regime-based).
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


def simulate_with_htf_simple_strategy(
    df: pd.DataFrame,
    position: Dict,
    atr_mult: float,  # Ignored
    min_hold_bars: int  # Ignored
) -> Tuple[Dict, List[Dict]]:
    """
    Simulate using SIMPLE HTF CROSSOVER strategy.
    Only exits when Close crosses HTF indicator.
    Returns: (final_position, exit_history)
    """
    exit_history = []
    current_pos = position.copy()

    for i in range(len(df)):
        if i < 2:
            continue

        df_slice = df.iloc[:i+1]

        # Use HTF crossover exit
        exit_result = htf_simple.evaluate_exit_htf_crossover(current_pos, df_slice)

        if exit_result:
            exit_history.append({
                "bar_index": i,
                "timestamp": df.index[i],
                "exit_price": exit_result["exit_price"],
                "reason": exit_result["reason"],
                "pnl": exit_result["pnl"],
                "bars_held": i
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

        # Simulate all three strategies
        print("\n[OLD STRATEGY - ATR + Trend Flip]")
        _, old_exits = simulate_with_old_strategy(df_after_entry, position, atr_mult, min_hold_bars)

        print("\n[NEW STRATEGY - Regime-Based]")
        _, new_exits = simulate_with_new_strategy(df_after_entry, position, atr_mult, min_hold_bars)

        print("\n[HTF SIMPLE - Pure Crossover]")
        _, htf_exits = simulate_with_htf_simple_strategy(df_after_entry, position, atr_mult, min_hold_bars)

        # Compare results
        old_exit = old_exits[0] if old_exits else None
        new_exit = new_exits[0] if new_exits else None
        htf_exit = htf_exits[0] if htf_exits else None

        result = {
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "entry_time": entry_time,
            "stake": stake,
            "old_exit": old_exit,
            "new_exit": new_exit,
            "htf_exit": htf_exit,
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

        print()

        if htf_exit:
            print(f"HTF: Exit @ {htf_exit['exit_price']:.8f} after {htf_exit['bars_held']} bars")
            print(f"     Reason: {htf_exit['reason']}")
            print(f"     PnL: {htf_exit['pnl']:.2f} USDT")
        else:
            print(f"HTF: NO EXIT (position still open)")

        # Calculate differences
        print(f"\n{'─'*80}")
        print(f"WINNER ANALYSIS:")
        print(f"{'─'*80}")

        strategies = []
        if old_exit:
            strategies.append(("OLD", old_exit['pnl']))
        if new_exit:
            strategies.append(("NEW", new_exit['pnl']))
        if htf_exit:
            strategies.append(("HTF", htf_exit['pnl']))

        if strategies:
            strategies.sort(key=lambda x: x[1], reverse=True)
            winner = strategies[0]
            print(f"🏆 WINNER: {winner[0]} Strategy with {winner[1]:.2f} USDT")

            if old_exit and new_exit:
                pnl_diff = new_exit['pnl'] - old_exit['pnl']
                print(f"\nNEW vs OLD: {pnl_diff:+.2f} USDT ({(pnl_diff/abs(old_exit['pnl'])*100 if old_exit['pnl'] != 0 else 0):+.1f}%)")

            if old_exit and htf_exit:
                htf_diff = htf_exit['pnl'] - old_exit['pnl']
                print(f"HTF vs OLD: {htf_diff:+.2f} USDT ({(htf_diff/abs(old_exit['pnl'])*100 if old_exit['pnl'] != 0 else 0):+.1f}%)")

            if new_exit and htf_exit:
                htf_new_diff = htf_exit['pnl'] - new_exit['pnl']
                print(f"HTF vs NEW: {htf_new_diff:+.2f} USDT ({(htf_new_diff/abs(new_exit['pnl'])*100 if new_exit['pnl'] != 0 else 0):+.1f}%)")

        return result

    except Exception as exc:
        print(f"[ERROR] {exc}")
        import traceback
        traceback.print_exc()
        return None


def run_batch_comparison(num_trades: int = 10, target_symbols: List[str] = None):
    """
    Compare both strategies on multiple recent trades.

    Args:
        num_trades: Number of recent trades to test (default: 10)
        target_symbols: List of symbols to filter (default: all)
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

    # Filter by symbols if specified
    if target_symbols:
        trades_df = trades_df[trades_df["symbol"].isin(target_symbols)]
        print(f"Filtered to {len(trades_df)} trades for symbols: {', '.join(target_symbols)}")

    if trades_df.empty:
        print(f"\n[ERROR] No trades found for specified symbols")
        return

    print(f"Testing last {num_trades} trades...\n")

    # Take last N trades
    recent_trades = trades_df.tail(num_trades)

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
    htf_pnls = [r["htf_exit"]["pnl"] for r in results if r["htf_exit"]]

    if old_pnls or new_pnls or htf_pnls:
        print(f"\nTotal PnL Comparison:")
        if old_pnls:
            old_total = sum(old_pnls)
            print(f"  OLD Strategy: {old_total:.2f} USDT")
        if new_pnls:
            new_total = sum(new_pnls)
            print(f"  NEW Strategy: {new_total:.2f} USDT")
        if htf_pnls:
            htf_total = sum(htf_pnls)
            print(f"  HTF Strategy: {htf_total:.2f} USDT")

        # Show improvements
        if old_pnls and new_pnls:
            improvement = new_total - old_total
            print(f"\n  NEW vs OLD: {improvement:+.2f} USDT ({(improvement/abs(old_total)*100 if old_total != 0 else 0):+.1f}%)")
        if old_pnls and htf_pnls:
            htf_improvement = htf_total - old_total
            print(f"  HTF vs OLD: {htf_improvement:+.2f} USDT ({(htf_improvement/abs(old_total)*100 if old_total != 0 else 0):+.1f}%)")
        if new_pnls and htf_pnls:
            htf_new_improvement = htf_total - new_total
            print(f"  HTF vs NEW: {htf_new_improvement:+.2f} USDT ({(htf_new_improvement/abs(new_total)*100 if new_total != 0 else 0):+.1f}%)")

        print(f"\nAverage PnL per Trade:")
        if old_pnls:
            old_avg = np.mean(old_pnls)
            print(f"  OLD Strategy: {old_avg:.2f} USDT")
        if new_pnls:
            new_avg = np.mean(new_pnls)
            print(f"  NEW Strategy: {new_avg:.2f} USDT")
        if htf_pnls:
            htf_avg = np.mean(htf_pnls)
            print(f"  HTF Strategy: {htf_avg:.2f} USDT")

        print(f"\nWin Rate:")
        if old_pnls:
            old_winners = sum(1 for p in old_pnls if p > 0)
            print(f"  OLD Strategy: {old_winners}/{len(old_pnls)} ({old_winners/len(old_pnls)*100:.1f}%)")
        if new_pnls:
            new_winners = sum(1 for p in new_pnls if p > 0)
            print(f"  NEW Strategy: {new_winners}/{len(new_pnls)} ({new_winners/len(new_pnls)*100:.1f}%)")
        if htf_pnls:
            htf_winners = sum(1 for p in htf_pnls if p > 0)
            print(f"  HTF Strategy: {htf_winners}/{len(htf_pnls)} ({htf_winners/len(htf_pnls)*100:.1f}%)")

        # Per-symbol breakdown
        symbols_tested = set(r["symbol"] for r in results)
        if len(symbols_tested) > 1:
            print(f"\n{'─'*80}")
            print("PER-SYMBOL BREAKDOWN:")
            print(f"{'─'*80}")

            for symbol in sorted(symbols_tested):
                symbol_results = [r for r in results if r["symbol"] == symbol]
                symbol_old = [r["old_exit"]["pnl"] for r in symbol_results if r["old_exit"]]
                symbol_new = [r["new_exit"]["pnl"] for r in symbol_results if r["new_exit"]]
                symbol_htf = [r["htf_exit"]["pnl"] for r in symbol_results if r["htf_exit"]]

                print(f"\n{symbol}:")
                print(f"  Trades: {len(symbol_results)}")
                if symbol_old:
                    print(f"  OLD: {sum(symbol_old):.2f} USDT")
                if symbol_new:
                    print(f"  NEW: {sum(symbol_new):.2f} USDT")
                if symbol_htf:
                    print(f"  HTF: {sum(symbol_htf):.2f} USDT")

        # Determine overall winner
        print(f"\n{'='*80}")
        print("OVERALL WINNER")
        print(f"{'='*80}")

        all_strategies = []
        if old_pnls:
            all_strategies.append(("OLD", old_total))
        if new_pnls:
            all_strategies.append(("NEW", new_total))
        if htf_pnls:
            all_strategies.append(("HTF", htf_total))

        if all_strategies:
            all_strategies.sort(key=lambda x: x[1], reverse=True)
            winner_name, winner_pnl = all_strategies[0]

            print(f"\n🏆 WINNER: {winner_name} Strategy")
            print(f"   Total PnL: {winner_pnl:.2f} USDT")

            if len(all_strategies) > 1:
                second_name, second_pnl = all_strategies[1]
                margin = winner_pnl - second_pnl
                print(f"   Beat {second_name} by: {margin:.2f} USDT")

            print(f"\n{'='*80}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare old vs new exit strategies")
    parser.add_argument("--trades", type=int, default=10, help="Number of recent trades to test (default: 10)")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols to test (e.g., BTC/EUR,ETH/EUR)")
    parser.add_argument("--all", action="store_true", help="Test all available trades")

    args = parser.parse_args()

    # Parse symbols
    target_symbols = None
    if args.symbols:
        target_symbols = [s.strip() for s in args.symbols.split(",")]

    # Determine number of trades
    num_trades = 999999 if args.all else args.trades

    run_batch_comparison(num_trades=num_trades, target_symbols=target_symbols)

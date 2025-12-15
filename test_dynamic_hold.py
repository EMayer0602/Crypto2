#!/usr/bin/env python3
"""
Test script for dynamic min_hold_days algorithm.

This demonstrates how to use the new calculate_dynamic_min_hold_days() function
to automatically adjust minimum hold periods based on market volatility and
recent trading performance.
"""

import pandas as pd
import Supertrend_5Min as st

def test_dynamic_hold_all_symbols():
    """Test dynamic hold calculation for all configured symbols."""
    print("=" * 80)
    print("DYNAMIC MIN HOLD DAYS - VOLATILITY-BASED CALCULATION")
    print("=" * 80)
    print()

    # Try to load recent trades for performance adjustment
    recent_trades = None
    try:
        recent_trades = pd.read_csv("paper_trading_log.csv")
        print(f"✓ Loaded {len(recent_trades)} recent trades for performance analysis")
    except FileNotFoundError:
        print("⚠ No trade history found - using volatility-only calculation")
    print()

    results = []

    for symbol in st.SYMBOLS:
        try:
            print(f"Analyzing {symbol}...")

            # Calculate optimal hold days
            optimal_days = st.calculate_dynamic_min_hold_days(
                symbol=symbol,
                recent_trades_df=recent_trades,
                lookback_days=30,
                min_days=0,
                max_days=7
            )

            # Get current market data for context
            df = st.fetch_data(symbol, st.TIMEFRAME, 100)
            if not df.empty:
                from ta.volatility import AverageTrueRange
                atr_series = AverageTrueRange(
                    df["high"], df["low"], df["close"], window=14
                ).average_true_range()

                current_price = float(df["close"].iloc[-1])
                current_atr = float(atr_series.iloc[-1])
                atr_pct = (current_atr / current_price) * 100

                results.append({
                    "Symbol": symbol,
                    "Price": f"{current_price:.4f}",
                    "ATR": f"{current_atr:.4f}",
                    "ATR%": f"{atr_pct:.2f}%",
                    "Optimal Hold": f"{optimal_days} days",
                    "Volatility": (
                        "Very High" if atr_pct > 7
                        else "High" if atr_pct > 4
                        else "Medium" if atr_pct > 2
                        else "Low"
                    )
                })

                print(f"  ✓ Price: {current_price:.4f}")
                print(f"  ✓ ATR: {current_atr:.4f} ({atr_pct:.2f}%)")
                print(f"  ✓ Optimal min hold: {optimal_days} days")
            else:
                results.append({
                    "Symbol": symbol,
                    "Price": "N/A",
                    "ATR": "N/A",
                    "ATR%": "N/A",
                    "Optimal Hold": f"{optimal_days} days",
                    "Volatility": "Unknown"
                })
                print(f"  ⚠ No data - using default: {optimal_days} days")

        except Exception as exc:
            print(f"  ✗ Error: {exc}")
            results.append({
                "Symbol": symbol,
                "Price": "Error",
                "ATR": "Error",
                "ATR%": "Error",
                "Optimal Hold": "Error",
                "Volatility": "Error"
            })

        print()

    # Display summary table
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print()

    # Save to CSV
    results_df.to_csv("dynamic_hold_analysis.csv", index=False)
    print("✓ Results saved to: dynamic_hold_analysis.csv")
    print()

    # Display recommendations
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("""
Based on volatility analysis:
- Low volatility (ATR < 2%): Use longer holds (5-7 days) to avoid whipsaws
- Medium volatility (ATR 2-7%): Use moderate holds (2-4 days)
- High volatility (ATR > 7%): Use shorter holds (0-2 days) for quick exits

The algorithm automatically adjusts based on:
1. Current volatility level (ATR as % of price)
2. Volatility trend (increasing/decreasing)
3. Recent trade performance (win rate and average P&L)

Usage in backtests:
    optimal_days = st.calculate_dynamic_min_hold_days(
        symbol="BTC/EUR",
        recent_trades_df=recent_trades,  # optional
        min_days=0,
        max_days=7
    )
    min_hold_bars = optimal_days * st.BARS_PER_DAY
    """)


def test_single_symbol(symbol="BTC/EUR"):
    """Test dynamic hold calculation for a single symbol with detailed output."""
    print(f"\n{'=' * 80}")
    print(f"DETAILED ANALYSIS: {symbol}")
    print(f"{'=' * 80}\n")

    # Load trade history if available
    recent_trades = None
    try:
        all_trades = pd.read_csv("paper_trading_log.csv")
        recent_trades = all_trades.tail(20)  # Last 20 trades
        print(f"Recent Performance (last {len(recent_trades)} trades):")
        winners = len(recent_trades[recent_trades["pnl"] > 0])
        print(f"  Win rate: {winners}/{len(recent_trades)} ({winners/len(recent_trades)*100:.1f}%)")
        print(f"  Avg P&L: {recent_trades['pnl'].mean():.2f} USDT")
        print()
    except:
        print("No trade history available\n")

    # Test across different scenarios
    scenarios = [
        (0, 3, "Conservative (max 3 days)"),
        (0, 5, "Moderate (max 5 days)"),
        (0, 7, "Aggressive (max 7 days)"),
        (1, 5, "With minimum (1-5 days)"),
    ]

    print("Scenario Testing:")
    for min_d, max_d, desc in scenarios:
        optimal = st.calculate_dynamic_min_hold_days(
            symbol=symbol,
            recent_trades_df=recent_trades,
            min_days=min_d,
            max_days=max_d
        )
        print(f"  {desc:30s} → {optimal} days")

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test dynamic min_hold_days calculation")
    parser.add_argument("--symbol", type=str, help="Test specific symbol")
    parser.add_argument("--all", action="store_true", help="Test all symbols")

    args = parser.parse_args()

    if args.symbol:
        test_single_symbol(args.symbol)
    elif args.all:
        test_dynamic_hold_all_symbols()
    else:
        # Default: test all symbols
        test_dynamic_hold_all_symbols()

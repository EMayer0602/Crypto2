"""
Find optimal EMA period and slope threshold for long trades per symbol.

Tests EMA periods from 70-200 and finds which gives best performance
when used as trend filter for long entries/exits.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import Supertrend_5Min as st

# EMA periods to test - now testing 10 to 70
EMA_PERIODS = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]

# Slope thresholds to test (percentage)
SLOPE_THRESHOLDS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]  # Added 0.05 and 0.15 for finer granularity


def calculate_ema_slope(df: pd.DataFrame, period: int, lookback: int = 1) -> pd.Series:
    """
    Calculate EMA slope as percentage change.

    Args:
        df: DataFrame with OHLCV data
        period: EMA period (e.g., 100)
        lookback: How many bars back to compare (default 1)

    Returns:
        Series with slope values as percentage
    """
    ema = df['close'].ewm(span=period, adjust=False).mean()
    slope = ((ema - ema.shift(lookback)) / ema.shift(lookback)) * 100
    return slope


def add_ema_slope_column(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Add EMA and slope columns to dataframe."""
    df = df.copy()
    df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    df[f'ema_{period}_slope'] = calculate_ema_slope(df, period)
    return df


def test_ema_filter_on_trades(
    trades_df: pd.DataFrame,
    symbol: str,
    ema_period: int,
    slope_threshold: float
) -> Dict:
    """
    Test how an EMA slope filter would affect existing long trades.

    Args:
        trades_df: DataFrame with trade history
        symbol: Symbol to test
        ema_period: EMA period to test
        slope_threshold: Minimum slope required for entry (%)

    Returns:
        Dict with performance metrics
    """
    # Filter for long trades of this symbol
    long_trades = trades_df[
        (trades_df['symbol'] == symbol) &
        (trades_df['direction'].str.lower() == 'long')
    ].copy()

    if long_trades.empty:
        return {
            'symbol': symbol,
            'ema_period': ema_period,
            'slope_threshold': slope_threshold,
            'trades_total': 0,
            'trades_filtered': 0,
            'pnl_total': 0,
            'pnl_filtered': 0,
            'win_rate_total': 0,
            'win_rate_filtered': 0,
            'improvement': 0
        }

    # Load price data for this symbol
    try:
        df = st.prepare_symbol_dataframe(symbol)
        df = add_ema_slope_column(df, ema_period)
    except Exception as e:
        print(f"[ERROR] Could not load data for {symbol}: {e}")
        return None

    # Check each trade entry time for EMA slope
    trades_passed_filter = []

    for idx, trade in long_trades.iterrows():
        entry_time = pd.Timestamp(trade['entry_time'])

        # Find closest bar
        if entry_time not in df.index:
            # Find nearest timestamp
            nearest_idx = df.index.get_indexer([entry_time], method='nearest')[0]
            entry_time = df.index[nearest_idx]

        if entry_time in df.index:
            slope = df.loc[entry_time, f'ema_{ema_period}_slope']

            # Check if slope meets threshold
            if pd.notna(slope) and slope >= slope_threshold:
                trades_passed_filter.append(trade)

    # Calculate metrics
    total_trades = len(long_trades)
    filtered_trades = len(trades_passed_filter)

    pnl_total = long_trades['pnl'].sum()
    pnl_filtered = sum(t['pnl'] for t in trades_passed_filter) if trades_passed_filter else 0

    winners_total = len(long_trades[long_trades['pnl'] > 0])
    win_rate_total = (winners_total / total_trades * 100) if total_trades > 0 else 0

    winners_filtered = len([t for t in trades_passed_filter if t['pnl'] > 0])
    win_rate_filtered = (winners_filtered / filtered_trades * 100) if filtered_trades > 0 else 0

    # Calculate improvement
    avg_pnl_total = pnl_total / total_trades if total_trades > 0 else 0
    avg_pnl_filtered = pnl_filtered / filtered_trades if filtered_trades > 0 else 0
    improvement = avg_pnl_filtered - avg_pnl_total

    return {
        'symbol': symbol,
        'ema_period': ema_period,
        'slope_threshold': slope_threshold,
        'trades_total': total_trades,
        'trades_filtered': filtered_trades,
        'filter_rate': (filtered_trades / total_trades * 100) if total_trades > 0 else 0,
        'pnl_total': pnl_total,
        'pnl_filtered': pnl_filtered,
        'avg_pnl_total': avg_pnl_total,
        'avg_pnl_filtered': avg_pnl_filtered,
        'win_rate_total': win_rate_total,
        'win_rate_filtered': win_rate_filtered,
        'improvement': improvement,
    }


def find_optimal_ema_per_symbol(trades_csv: str) -> pd.DataFrame:
    """
    Find optimal EMA period and slope threshold for each symbol.

    Args:
        trades_csv: Path to trades CSV file

    Returns:
        DataFrame with optimal parameters per symbol
    """
    # Load trades
    if not os.path.exists(trades_csv):
        print(f"[ERROR] Trades file not found: {trades_csv}")
        return pd.DataFrame()

    trades_df = pd.read_csv(trades_csv)

    # Get unique symbols with long trades
    symbols = trades_df[
        trades_df['direction'].str.lower() == 'long'
    ]['symbol'].unique()

    print(f"\n=== Testing EMA Slope Filter on {len(symbols)} Symbols ===")
    print(f"EMA Periods: {EMA_PERIODS}")
    print(f"Slope Thresholds: {SLOPE_THRESHOLDS}%\n")

    all_results = []

    for symbol in symbols:
        print(f"\n--- {symbol} ---")

        best_result = None
        best_improvement = -float('inf')

        for ema_period in EMA_PERIODS:
            for slope_threshold in SLOPE_THRESHOLDS:
                result = test_ema_filter_on_trades(
                    trades_df, symbol, ema_period, slope_threshold
                )

                if result is None or result['trades_total'] == 0:
                    continue

                all_results.append(result)

                # Track best combination
                if result['improvement'] > best_improvement:
                    best_improvement = result['improvement']
                    best_result = result

        if best_result:
            print(f"  Best: EMA-{best_result['ema_period']}, Slope≥{best_result['slope_threshold']}%")
            print(f"  Trades: {best_result['trades_total']} → {best_result['trades_filtered']} ({best_result['filter_rate']:.1f}% kept)")
            print(f"  PnL: {best_result['pnl_total']:.2f} → {best_result['pnl_filtered']:.2f} USDT")
            print(f"  Avg: {best_result['avg_pnl_total']:.2f} → {best_result['avg_pnl_filtered']:.2f} USDT")
            print(f"  Win Rate: {best_result['win_rate_total']:.1f}% → {best_result['win_rate_filtered']:.1f}%")
            print(f"  Improvement: +{best_result['improvement']:.2f} USDT per trade")

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Save all results
    results_df.to_csv('ema_slope_filter_results.csv', index=False)
    print(f"\n[Saved] All results to ema_slope_filter_results.csv")

    # Get best parameters per symbol
    optimal_df = results_df.loc[results_df.groupby('symbol')['improvement'].idxmax()]
    optimal_df = optimal_df.sort_values('improvement', ascending=False)

    # Save optimal parameters
    optimal_df.to_csv('optimal_ema_slope_per_symbol.csv', index=False)
    print(f"[Saved] Optimal parameters to optimal_ema_slope_per_symbol.csv")

    # Print summary
    print("\n" + "="*80)
    print("OPTIMAL EMA SLOPE PARAMETERS PER SYMBOL")
    print("="*80)
    print(optimal_df[[
        'symbol', 'ema_period', 'slope_threshold',
        'trades_total', 'trades_filtered', 'filter_rate',
        'avg_pnl_total', 'avg_pnl_filtered', 'improvement',
        'win_rate_total', 'win_rate_filtered'
    ]].to_string(index=False))

    return optimal_df


if __name__ == "__main__":
    # Use the most recent simulation trades
    trades_csv = "paper_trading_simulation_log.csv"

    if len(sys.argv) > 1:
        trades_csv = sys.argv[1]

    print(f"Analyzing trades from: {trades_csv}")

    # Configure exchange
    st.configure_exchange(use_testnet=True)

    # Find optimal EMA parameters
    optimal_df = find_optimal_ema_per_symbol(trades_csv)

    if not optimal_df.empty:
        print("\n✅ Analysis complete!")
        print("\nNext steps:")
        print("1. Review optimal_ema_slope_per_symbol.csv")
        print("2. Integrate EMA slope filter into paper_trader.py")
        print("3. Run backtest with filter enabled")
    else:
        print("\n❌ No results generated")

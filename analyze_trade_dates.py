#!/usr/bin/env python3
"""
Analyze trade dates from optimized parameter CSV files.

This script reads the individual trade CSV files referenced in best_params_overall.csv
and shows:
- First and last trade dates for each symbol/direction
- Trade distribution by month
- Whether any trades occurred in Q1 2025
"""

import os
import pandas as pd
from pathlib import Path

def analyze_trade_dates():
    """Analyze trade dates from CSV files."""

    # Read best_params_overall.csv
    params_file = "report_html/best_params_overall.csv"
    if not os.path.exists(params_file):
        print(f"Error: {params_file} not found")
        return

    df_params = pd.read_csv(params_file, sep=';')

    print("="*80)
    print("TRADE DATE ANALYSIS")
    print("="*80)
    print()

    all_trades = []
    missing_files = []

    for idx, row in df_params.iterrows():
        symbol = row['Symbol']
        direction = row['Direction']
        trades_csv = row.get('TradesCSV', '')

        if not trades_csv or pd.isna(trades_csv):
            continue

        # Convert Windows path to Unix path if needed
        trades_csv = trades_csv.replace('\\', '/')

        if not os.path.exists(trades_csv):
            missing_files.append(trades_csv)
            continue

        try:
            df_trades = pd.read_csv(trades_csv, sep=';')

            if df_trades.empty or 'EntryTime' not in df_trades.columns:
                continue

            # Parse entry time
            df_trades['EntryTime'] = pd.to_datetime(df_trades['EntryTime'])

            first_trade = df_trades['EntryTime'].min()
            last_trade = df_trades['EntryTime'].max()
            num_trades = len(df_trades)

            # Count trades by month
            df_trades['Month'] = df_trades['EntryTime'].dt.to_period('M')
            trades_by_month = df_trades.groupby('Month').size().to_dict()

            # Check Q1 2025 trades
            q1_2025 = df_trades[
                (df_trades['EntryTime'] >= '2025-01-01') &
                (df_trades['EntryTime'] < '2025-04-01')
            ]

            all_trades.append({
                'Symbol': symbol,
                'Direction': direction,
                'NumTrades': num_trades,
                'FirstTrade': first_trade,
                'LastTrade': last_trade,
                'Q1_2025_Trades': len(q1_2025),
                'Jan_2025': len(df_trades[df_trades['EntryTime'].dt.to_period('M') == '2025-01']),
                'Feb_2025': len(df_trades[df_trades['EntryTime'].dt.to_period('M') == '2025-02']),
                'Mar_2025': len(df_trades[df_trades['EntryTime'].dt.to_period('M') == '2025-03']),
                'TradesByMonth': trades_by_month
            })

        except Exception as e:
            print(f"Error reading {trades_csv}: {e}")
            continue

    if not all_trades:
        print("No trade data found to analyze")
        if missing_files:
            print(f"\nMissing {len(missing_files)} CSV files:")
            for f in missing_files[:5]:
                print(f"  - {f}")
            if len(missing_files) > 5:
                print(f"  ... and {len(missing_files) - 5} more")
        return

    # Summary table
    df_summary = pd.DataFrame(all_trades)
    df_summary = df_summary.sort_values('FirstTrade')

    print("\nTRADE DATE SUMMARY (sorted by first trade date):")
    print("-" * 80)
    print(f"{'Symbol':<12} {'Dir':<5} {'Trades':>7} {'First Trade':<20} {'Last Trade':<20} {'Jan 25':>7}")
    print("-" * 80)

    for _, row in df_summary.iterrows():
        print(f"{row['Symbol']:<12} {row['Direction']:<5} {row['NumTrades']:>7} "
              f"{row['FirstTrade'].strftime('%Y-%m-%d %H:%M'):<20} "
              f"{row['LastTrade'].strftime('%Y-%m-%d %H:%M'):<20} "
              f"{row['Jan_2025']:>7}")

    print("-" * 80)
    print(f"{'TOTAL':>25} {df_summary['NumTrades'].sum():>7} "
          f"{'':>41} {df_summary['Jan_2025'].sum():>7}")
    print()

    # Q1 2025 analysis
    q1_total = df_summary['Q1_2025_Trades'].sum()
    jan_total = df_summary['Jan_2025'].sum()
    feb_total = df_summary['Feb_2025'].sum()
    mar_total = df_summary['Mar_2025'].sum()

    print("\nQ1 2025 BREAKDOWN:")
    print("-" * 40)
    print(f"  January 2025:  {jan_total:>4} trades")
    print(f"  February 2025: {feb_total:>4} trades")
    print(f"  March 2025:    {mar_total:>4} trades")
    print(f"  Q1 2025 Total: {q1_total:>4} trades")
    print()

    # Earliest trades
    print("\nEARLIEST TRADES BY SYMBOL:")
    print("-" * 60)
    for _, row in df_summary.head(10).iterrows():
        print(f"  {row['Symbol']:<12} {row['Direction']:<5} : {row['FirstTrade'].strftime('%Y-%m-%d %H:%M')}")
    print()

    # Month distribution
    print("\nTRADE DISTRIBUTION BY MONTH:")
    print("-" * 60)
    all_months = {}
    for trades_info in all_trades:
        for month, count in trades_info['TradesByMonth'].items():
            month_str = str(month)
            all_months[month_str] = all_months.get(month_str, 0) + count

    for month in sorted(all_months.keys()):
        print(f"  {month}: {all_months[month]:>4} trades")
    print()

    # Key findings
    print("\nKEY FINDINGS:")
    print("-" * 60)

    if jan_total == 0:
        print("  ⚠ NO TRADES IN JANUARY 2025")
        print(f"    - Earliest trade: {df_summary['FirstTrade'].min().strftime('%Y-%m-%d')}")
        print(f"    - This explains why --start 2025-01-01 --end 2025-01-31 produced 0 trades")
    else:
        print(f"  ✓ Found {jan_total} trades in January 2025")

    if q1_total == 0:
        print("  ⚠ NO TRADES IN Q1 2025 (Jan-Mar)")
    else:
        print(f"  ✓ Found {q1_total} trades in Q1 2025")

    earliest = df_summary['FirstTrade'].min()
    if earliest.year == 2025 and earliest.month >= 4:
        print(f"  ⚠ First trade not until {earliest.strftime('%B %Y')}")
        print("    - This matches your earlier observation (first trades in April/September)")
    elif earliest.year == 2024:
        print(f"  ✓ Trades starting from {earliest.strftime('%B %Y')}")

    print()
    print("="*80)

if __name__ == "__main__":
    analyze_trade_dates()

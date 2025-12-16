"""
Visualize Trading Strategies with Charts

Creates charts showing:
- Price action
- HTF indicator
- Buy/sell markers for each strategy
- Entry/exit points comparison
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Import our strategy modules
import improved_exit_strategy as improved
import htf_crossover_strategy as htf_simple


def load_trade_data(symbol, direction, entry_time):
    """Load historical data for a specific trade"""
    log_file = Path("paper_trading_simulation_log.json")

    if not log_file.exists():
        print(f"Error: {log_file} not found")
        return None

    with open(log_file, 'r') as f:
        data = json.load(f)

    # Find matching entry
    for entry in data:
        if (entry["symbol"] == symbol and
            entry["direction"] == direction and
            entry["entry_time"] == entry_time):

            df = pd.DataFrame(entry["historical_data"])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

            return df, entry

    return None, None


def plot_trade_comparison(symbol, direction, entry_time):
    """Create chart comparing all three strategies"""

    df, trade_info = load_trade_data(symbol, direction, entry_time)

    if df is None:
        print(f"Trade not found: {symbol} {direction} @ {entry_time}")
        return

    entry_price = float(trade_info["entry_price"])
    entry_timestamp = pd.to_datetime(entry_time)
    stake = float(trade_info.get("stake", 1000))
    atr_mult = float(trade_info.get("atr_mult", 1.5))
    min_hold_bars = int(trade_info.get("min_hold_bars", 12))

    # Create position dict
    position = {
        "symbol": symbol,
        "direction": direction,
        "entry_price": entry_price,
        "entry_time": entry_timestamp,
        "stake": stake,
    }

    # Get entry index
    entry_idx = df[df['timestamp'] >= entry_timestamp].index[0]
    df_after_entry = df.iloc[entry_idx:].copy()

    # Simulate all three strategies
    old_exit = simulate_old_strategy(df_after_entry, position, atr_mult, min_hold_bars)
    new_exit = simulate_new_strategy(df_after_entry, position, atr_mult, min_hold_bars)
    htf_exit = simulate_htf_strategy(df_after_entry, position, atr_mult, min_hold_bars)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Prepare data
    timestamps = df['timestamp']
    close = df['close'].astype(float)
    htf_indicator = df['htf_indicator'].astype(float)

    # Plot 1: Price and HTF Indicator
    ax1.plot(timestamps, close, label='Close Price', color='black', linewidth=1.5)
    ax1.plot(timestamps, htf_indicator, label='HTF Indicator', color='blue', linewidth=1.5, alpha=0.7)

    # Mark entry
    ax1.scatter(entry_timestamp, entry_price, color='green', s=200, marker='^',
                label='Entry', zorder=5, edgecolors='black', linewidths=2)

    # Mark exits
    if old_exit:
        exit_time = df_after_entry.iloc[old_exit['bars_held']]['timestamp']
        ax1.scatter(exit_time, old_exit['exit_price'], color='red', s=150, marker='v',
                   label=f"OLD Exit ({old_exit['bars_held']} bars)", zorder=5, alpha=0.6)

    if new_exit:
        exit_time = df_after_entry.iloc[new_exit['bars_held']]['timestamp']
        ax1.scatter(exit_time, new_exit['exit_price'], color='orange', s=150, marker='v',
                   label=f"NEW Exit ({new_exit['bars_held']} bars)", zorder=5, alpha=0.6)

    if htf_exit:
        exit_time = df_after_entry.iloc[htf_exit['bars_held']]['timestamp']
        ax1.scatter(exit_time, htf_exit['exit_price'], color='purple', s=150, marker='v',
                   label=f"HTF Exit ({htf_exit['bars_held']} bars)", zorder=5, alpha=0.6)

    ax1.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax1.set_title(f'{symbol} {direction} - Entry @ {entry_price:.8f} on {entry_time}',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: PnL over time for each strategy
    old_pnl = []
    new_pnl = []
    htf_pnl = []

    for i in range(len(df_after_entry)):
        curr_price = float(df_after_entry.iloc[i]['close'])

        # Calculate unrealized PnL
        if direction == "LONG":
            pnl = (curr_price - entry_price) / entry_price * stake
        else:
            pnl = (entry_price - curr_price) / entry_price * stake

        old_pnl.append(pnl)
        new_pnl.append(pnl)
        htf_pnl.append(pnl)

    times_after = df_after_entry['timestamp']
    ax2.plot(times_after, old_pnl, label='OLD PnL', color='red', linewidth=1.5, alpha=0.7)
    ax2.plot(times_after, new_pnl, label='NEW PnL', color='orange', linewidth=1.5, alpha=0.7)
    ax2.plot(times_after, htf_pnl, label='HTF PnL', color='purple', linewidth=1.5, alpha=0.7)

    # Mark actual exit PnLs
    if old_exit:
        exit_time = df_after_entry.iloc[old_exit['bars_held']]['timestamp']
        ax2.scatter(exit_time, old_exit['pnl'], color='red', s=150, marker='o', zorder=5)
        ax2.text(exit_time, old_exit['pnl'], f" {old_exit['pnl']:.1f}", fontsize=9)

    if new_exit:
        exit_time = df_after_entry.iloc[new_exit['bars_held']]['timestamp']
        ax2.scatter(exit_time, new_exit['pnl'], color='orange', s=150, marker='o', zorder=5)
        ax2.text(exit_time, new_exit['pnl'], f" {new_exit['pnl']:.1f}", fontsize=9)

    if htf_exit:
        exit_time = df_after_entry.iloc[htf_exit['bars_held']]['timestamp']
        ax2.scatter(exit_time, htf_exit['pnl'], color='purple', s=150, marker='o', zorder=5)
        ax2.text(exit_time, htf_exit['pnl'], f" {htf_exit['pnl']:.1f}", fontsize=9)

    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax2.set_ylabel('PnL (USDT)', fontsize=12, fontweight='bold')
    ax2.set_title('Unrealized PnL Over Time', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Save chart
    filename = f"chart_{symbol.replace('/', '_')}_{direction}_{entry_time.replace(':', '-').replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Chart saved: {filename}")

    plt.close()

    # Print summary
    print(f"\n{'='*80}")
    print(f"TRADE SUMMARY: {symbol} {direction}")
    print(f"{'='*80}")
    print(f"Entry: {entry_price:.8f} @ {entry_time}")

    if old_exit:
        print(f"\nOLD: Exit @ {old_exit['exit_price']:.8f} after {old_exit['bars_held']} bars")
        print(f"     Reason: {old_exit['reason']}")
        print(f"     PnL: {old_exit['pnl']:.2f} USDT")

    if new_exit:
        print(f"\nNEW: Exit @ {new_exit['exit_price']:.8f} after {new_exit['bars_held']} bars")
        print(f"     Reason: {new_exit['reason']}")
        print(f"     PnL: {new_exit['pnl']:.2f} USDT")

    if htf_exit:
        print(f"\nHTF: Exit @ {htf_exit['exit_price']:.8f} after {htf_exit['bars_held']} bars")
        print(f"     Reason: {htf_exit['reason']}")
        print(f"     PnL: {htf_exit['pnl']:.2f} USDT")


def simulate_old_strategy(df, position, atr_mult, min_hold_bars):
    """Simulate old ATR + trend flip strategy"""
    from copy import deepcopy

    for i in range(len(df)):
        bars_held = i
        if bars_held < min_hold_bars:
            continue

        curr = df.iloc[i]
        close = float(curr["close"])
        atr = float(curr["atr"])
        htf_ind = float(curr["htf_indicator"])

        direction = position["direction"]
        entry_price = position["entry_price"]
        stake = position["stake"]

        # ATR stop
        if direction == "LONG":
            stop_price = entry_price - (atr * atr_mult)
            if close <= stop_price:
                pnl = (close - entry_price) / entry_price * stake
                return {"exit_price": close, "bars_held": bars_held, "reason": f"ATR stop x{atr_mult:.2f}", "pnl": pnl}
        else:
            stop_price = entry_price + (atr * atr_mult)
            if close >= stop_price:
                pnl = (entry_price - close) / entry_price * stake
                return {"exit_price": close, "bars_held": bars_held, "reason": f"ATR stop x{atr_mult:.2f}", "pnl": pnl}

        # Trend flip
        if direction == "LONG" and close < htf_ind:
            pnl = (close - entry_price) / entry_price * stake
            return {"exit_price": close, "bars_held": bars_held, "reason": "Trend flip", "pnl": pnl}

        if direction == "SHORT" and close > htf_ind:
            pnl = (entry_price - close) / entry_price * stake
            return {"exit_price": close, "bars_held": bars_held, "reason": "Trend flip", "pnl": pnl}

    return None


def simulate_new_strategy(df, position, atr_mult, min_hold_bars):
    """Simulate regime-based strategy"""
    result = improved.evaluate_exit_improved(position, df, atr_mult, min_hold_bars)
    return result


def simulate_htf_strategy(df, position, atr_mult, min_hold_bars):
    """Simulate HTF crossover strategy"""
    result = htf_simple.check_htf_crossover_exit(position, df)
    return result


def main():
    if len(sys.argv) < 4:
        print("Usage: python visualize_trades.py <symbol> <direction> <entry_time>")
        print('Example: python visualize_trades.py "ETH/EUR" "LONG" "2025-12-15 10:00:00+01:00"')
        return

    symbol = sys.argv[1]
    direction = sys.argv[2]
    entry_time = sys.argv[3]

    plot_trade_comparison(symbol, direction, entry_time)


if __name__ == "__main__":
    main()

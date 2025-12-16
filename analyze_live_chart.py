"""
Live Chart Analyzer - Analyze current market data for HTF crossover signals

Fetches live data from Binance and shows where HTF crossover entry/exit signals occur.
"""

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import ccxt

# Import indicator calculation
import Supertrend_5Min as st


def fetch_live_data(symbol, timeframe='5m', hours=24):
    """Fetch live data from Binance"""
    print(f"Fetching {hours}h of {timeframe} data for {symbol}...")

    exchange = ccxt.binance({
        'enableRateLimit': True,
    })

    # Calculate how many bars we need
    since = exchange.parse8601((datetime.now() - timedelta(hours=hours)).isoformat())

    # Fetch OHLCV data
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)

    # Convert to DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    print(f"✓ Fetched {len(df)} bars from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    return df


def calculate_htf_indicator(df, indicator_type='supertrend', length=10, multiplier=2.3, htf_factor=10):
    """Calculate HTF indicator"""

    print(f"Calculating {indicator_type.upper()} with length={length}, multiplier={multiplier}, HTF factor={htf_factor}...")

    # Calculate HTF length
    htf_length = int(length * htf_factor)

    if indicator_type.lower() == 'supertrend':
        # Compute Supertrend with HTF length
        df = st.compute_supertrend(df, htf_length, multiplier)
        df['htf_indicator'] = df['supertrend']

    elif indicator_type.lower() == 'jma':
        # Compute JMA with HTF length
        df = st.compute_jma(df, htf_length)
        df['htf_indicator'] = df['jma']

    elif indicator_type.lower() == 'kama':
        # Compute KAMA with HTF length
        df = st.compute_kama(df, htf_length)
        df['htf_indicator'] = df['kama']

    else:
        raise ValueError(f"Unknown indicator: {indicator_type}")

    print(f"✓ {indicator_type.upper()} calculated")

    return df


def detect_crossovers(df):
    """Detect HTF crossover signals"""

    signals = []

    for i in range(1, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i-1]

        close_curr = float(curr['close'])
        close_prev = float(prev['close'])
        htf_curr = float(curr['htf_indicator'])
        htf_prev = float(prev['htf_indicator'])

        # Long entry: Close crosses above HTF
        if close_prev < htf_prev and close_curr > htf_curr:
            signals.append({
                'index': i,
                'time': curr['timestamp'],
                'price': close_curr,
                'type': 'LONG_ENTRY',
                'label': 'BUY'
            })

        # Long exit: Close crosses below HTF
        elif close_prev > htf_prev and close_curr < htf_curr:
            signals.append({
                'index': i,
                'time': curr['timestamp'],
                'price': close_curr,
                'type': 'LONG_EXIT',
                'label': 'SELL'
            })

    print(f"✓ Found {len(signals)} crossover signals")

    return signals


def plot_chart(df, signals, symbol, indicator_type):
    """Create chart with HTF indicator and crossover signals"""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True,
                                     gridspec_kw={'height_ratios': [3, 1]})

    # Plot 1: Price and HTF Indicator
    timestamps = df['timestamp']
    close = df['close'].astype(float)
    htf = df['htf_indicator'].astype(float)

    ax1.plot(timestamps, close, label='Close Price', color='black', linewidth=1.5)
    ax1.plot(timestamps, htf, label=f'HTF {indicator_type.upper()}',
             color='blue', linewidth=2, alpha=0.7)

    # Mark signals
    for signal in signals:
        if signal['type'] == 'LONG_ENTRY':
            ax1.scatter(signal['time'], signal['price'],
                       color='green', s=200, marker='^',
                       label='BUY' if signals.index(signal) == 0 else '',
                       zorder=5, edgecolors='black', linewidths=2)
            ax1.text(signal['time'], signal['price'] * 0.995,
                    signal['label'],
                    ha='center', va='top', fontsize=9, fontweight='bold')

        elif signal['type'] == 'LONG_EXIT':
            ax1.scatter(signal['time'], signal['price'],
                       color='red', s=200, marker='v',
                       label='SELL' if signals.index(signal) == 0 else '',
                       zorder=5, edgecolors='black', linewidths=2)
            ax1.text(signal['time'], signal['price'] * 1.005,
                    signal['label'],
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax1.set_title(f'{symbol} - HTF {indicator_type.upper()} Crossover Signals',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Volume
    colors = ['green' if df.iloc[i]['close'] >= df.iloc[i]['open'] else 'red'
              for i in range(len(df))]
    ax2.bar(timestamps, df['volume'], color=colors, alpha=0.5, width=0.0008)
    ax2.set_ylabel('Volume', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Save chart
    filename = f"live_chart_{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Chart saved: {filename}")

    plt.show()


def calculate_trade_results(signals):
    """Calculate hypothetical trade results"""

    if len(signals) < 2:
        print("\nNot enough signals to calculate trades")
        return

    print(f"\n{'='*80}")
    print("TRADE SIMULATION (HTF Crossover Strategy)")
    print(f"{'='*80}")

    trades = []
    entry = None

    for signal in signals:
        if signal['type'] == 'LONG_ENTRY' and entry is None:
            entry = signal
            print(f"\n🟢 LONG ENTRY @ {signal['price']:.8f} on {signal['time']}")

        elif signal['type'] == 'LONG_EXIT' and entry is not None:
            exit_signal = signal
            pnl_pct = ((exit_signal['price'] - entry['price']) / entry['price']) * 100
            pnl_usdt = pnl_pct * 10  # Assuming 1000 USDT stake

            print(f"🔴 LONG EXIT  @ {exit_signal['price']:.8f} on {exit_signal['time']}")
            print(f"   PnL: {pnl_pct:+.2f}% ({pnl_usdt:+.2f} USDT on 1000 stake)")

            trades.append({
                'entry': entry,
                'exit': exit_signal,
                'pnl_pct': pnl_pct,
                'pnl_usdt': pnl_usdt
            })

            entry = None

    if entry is not None:
        print(f"\n⚠️  Position still OPEN (entered @ {entry['price']:.8f})")

    if trades:
        total_pnl = sum(t['pnl_usdt'] for t in trades)
        winners = sum(1 for t in trades if t['pnl_usdt'] > 0)

        print(f"\n{'='*80}")
        print(f"SUMMARY: {len(trades)} completed trades")
        print(f"Win Rate: {winners}/{len(trades)} ({winners/len(trades)*100:.1f}%)")
        print(f"Total PnL: {total_pnl:+.2f} USDT (1000 USDT per trade)")
        print(f"{'='*80}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_live_chart.py <SYMBOL> [--hours 24] [--indicator supertrend] [--length 10] [--multiplier 2.3] [--htf 10]")
        print("\nExamples:")
        print('  python analyze_live_chart.py SUI/USDC')
        print('  python analyze_live_chart.py ETH/EUR --hours 48 --indicator kama')
        print('  python analyze_live_chart.py BTC/USDT --indicator jma --length 20 --htf 5')
        return

    symbol = sys.argv[1]

    # Parse arguments
    hours = 24
    indicator = 'supertrend'
    length = 10
    multiplier = 2.3
    htf_factor = 10

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--hours':
            hours = int(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == '--indicator':
            indicator = sys.argv[i+1].lower()
            i += 2
        elif sys.argv[i] == '--length':
            length = int(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == '--multiplier':
            multiplier = float(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == '--htf':
            htf_factor = int(sys.argv[i+1])
            i += 2
        else:
            i += 1

    print(f"\n{'='*80}")
    print(f"LIVE CHART ANALYSIS: {symbol}")
    print(f"{'='*80}")
    print(f"Timeframe: 5m")
    print(f"Period: {hours} hours")
    print(f"Indicator: {indicator.upper()}")
    print(f"Length: {length} (HTF: {length * htf_factor})")
    if indicator == 'supertrend':
        print(f"Multiplier: {multiplier}")
    print(f"{'='*80}\n")

    # Fetch data
    df = fetch_live_data(symbol, '5m', hours)

    # Calculate indicator
    df = calculate_htf_indicator(df, indicator, length, multiplier, htf_factor)

    # Detect crossovers
    signals = detect_crossovers(df)

    # Calculate trades
    calculate_trade_results(signals)

    # Plot chart
    plot_chart(df, signals, symbol, indicator)


if __name__ == "__main__":
    main()

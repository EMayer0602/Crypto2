"""
Find optimal hold time (in bars) per symbol based on peak profit analysis.

Standalone version - doesn't require Supertrend_5Min.py imports.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import ccxt
from datetime import timezone
from zoneinfo import ZoneInfo

BERLIN_TZ = ZoneInfo("Europe/Berlin")


def fetch_data_simple(symbol, timeframe, limit=2000):
    """Simplified data fetch using ccxt directly."""
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
            'warnOnFetchCurrenciesWithoutPermission': False
        }
    })
    # Disable fetchCurrencies to avoid permission warnings
    if hasattr(exchange, 'has') and isinstance(exchange.has, dict):
        exchange.has['fetchCurrencies'] = False

    # Fetch more than needed as buffer
    fetch_limit = limit + 100
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=fetch_limit)

    # Convert to DataFrame
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(BERLIN_TZ)
    df = df.set_index("timestamp")

    return df.tail(limit)


def analyze_optimal_hold_time_per_symbol(trades_csv="paper_trading_simulation_log.csv"):
    """
    Analyze historical trades to find optimal hold time per symbol/direction.

    Returns dict: {(symbol, direction): optimal_bars}
    """
    if not Path(trades_csv).exists():
        print(f"❌ File not found: {trades_csv}")
        return {}

    trades_df = pd.read_csv(trades_csv)
    if trades_df.empty:
        print("❌ No trades found")
        return {}

    print(f"📊 Analyzing {len(trades_df)} trades for optimal hold times...\n")

    symbol_direction_stats = {}

    for symbol in trades_df["symbol"].unique():
        print(f"Analyzing {symbol}...")

        # Determine timeframe from first trade's param_desc
        symbol_trades = trades_df[trades_df["symbol"] == symbol]
        first_trade = symbol_trades.iloc[0]
        htf = first_trade.get("htf", "8h")  # Default to 8h if not specified

        print(f"  Using timeframe: {htf}")

        try:
            df = fetch_data_simple(symbol, htf, limit=2000)
        except Exception as e:
            print(f"  ⚠️  Failed to fetch {symbol}: {e}")
            continue

        for direction in ["long", "short"]:
            dir_trades = symbol_trades[symbol_trades["direction"].str.lower() == direction]

            if dir_trades.empty:
                continue

            bars_to_peak_list = []
            total_bars_list = []
            peak_pnl_list = []
            actual_pnl_list = []

            for _, trade in dir_trades.iterrows():
                entry_time = pd.to_datetime(trade.get("entry_time") or trade.get("Zeit"))
                exit_time = pd.to_datetime(trade.get("exit_time") or trade.get("ExitZeit"))
                entry_price = float(trade.get("entry_price") or trade.get("Entry"))
                stake = float(trade.get("stake") or trade.get("Stake", 1000))

                # Get price data during the trade
                trade_mask = (df.index >= entry_time) & (df.index <= exit_time)
                trade_df = df.loc[trade_mask].copy()

                if trade_df.empty or len(trade_df) < 2:
                    continue

                # Calculate unrealized PnL at each bar
                if direction == "long":
                    trade_df["unrealized_pnl"] = ((trade_df["high"] - entry_price) / entry_price) * stake
                else:
                    trade_df["unrealized_pnl"] = ((entry_price - trade_df["low"]) / entry_price) * stake

                # Find peak
                peak_pnl = trade_df["unrealized_pnl"].max()
                peak_idx = trade_df["unrealized_pnl"].idxmax()
                bars_to_peak = len(trade_df.loc[:peak_idx])
                total_bars = len(trade_df)

                bars_to_peak_list.append(bars_to_peak)
                total_bars_list.append(total_bars)
                peak_pnl_list.append(peak_pnl)
                actual_pnl_list.append(trade.get("pnl") or trade.get("PnL (USD)", 0))

            if not bars_to_peak_list:
                continue

            # Calculate statistics
            avg_bars_to_peak = np.mean(bars_to_peak_list)
            median_bars_to_peak = np.median(bars_to_peak_list)
            avg_total_bars = np.mean(total_bars_list)
            avg_peak_pnl = np.mean(peak_pnl_list)
            avg_actual_pnl = np.mean(actual_pnl_list)
            left_on_table = avg_peak_pnl - avg_actual_pnl

            # Recommended hold time: median bars to peak (more robust than mean)
            recommended_hold = int(median_bars_to_peak)

            symbol_direction_stats[(symbol, direction)] = {
                "avg_bars_to_peak": avg_bars_to_peak,
                "median_bars_to_peak": median_bars_to_peak,
                "recommended_hold": recommended_hold,
                "avg_total_bars": avg_total_bars,
                "avg_peak_pnl": avg_peak_pnl,
                "avg_actual_pnl": avg_actual_pnl,
                "left_on_table": left_on_table,
                "num_trades": len(bars_to_peak_list),
                "peak_at_pct": (median_bars_to_peak / avg_total_bars * 100) if avg_total_bars > 0 else 0,
            }

    return symbol_direction_stats


def generate_optimal_exit_config(stats_dict, output_file="optimal_hold_times.py"):
    """Generate Python config file with optimal hold times per symbol."""

    print("\n" + "="*80)
    print("OPTIMAL HOLD TIME ANALYSIS")
    print("="*80)

    # Sort by symbol
    sorted_items = sorted(stats_dict.items(), key=lambda x: (x[0][0], x[0][1]))

    config_lines = [
        '"""',
        'Optimal hold times per symbol/direction based on historical peak analysis.',
        '',
        'Generated from peak profit analysis showing when profit peaks on average.',
        'Use these values for time-based exits to capture maximum profits.',
        '"""',
        '',
        '# Optimal hold time in bars (when to exit for max profit)',
        'OPTIMAL_HOLD_BARS = {',
    ]

    for (symbol, direction), stats in sorted_items:
        recommended = stats["recommended_hold"]
        avg_peak_pnl = stats["avg_peak_pnl"]
        left = stats["left_on_table"]
        peak_pct = stats["peak_at_pct"]

        print(f"\n{symbol:12s} {direction.upper():5s}")
        print(f"  Trades:              {stats['num_trades']}")
        print(f"  Median bars to peak: {stats['median_bars_to_peak']:.1f}")
        print(f"  Avg total bars:      {stats['avg_total_bars']:.1f}")
        print(f"  Peak at:             {peak_pct:.1f}% of trade")
        print(f"  Avg peak PnL:        {avg_peak_pnl:>10.2f} USDT")
        print(f"  Avg actual PnL:      {stats['avg_actual_pnl']:>10.2f} USDT")
        print(f"  Left on table:       {left:>10.2f} USDT")
        print(f"  → RECOMMENDED:       Exit after {recommended} bars")

        config_lines.append(f'    ("{symbol}", "{direction}"): {recommended},  # Peak at {peak_pct:.0f}%, saves {left:.0f} USDT/trade')

    config_lines.append('}')
    config_lines.append('')
    config_lines.append('# How to use:')
    config_lines.append('# In paper_trader.py, add time-based exit logic:')
    config_lines.append('# from optimal_hold_times import OPTIMAL_HOLD_BARS')
    config_lines.append('# exit_after_bars = OPTIMAL_HOLD_BARS.get((symbol, direction), 999)')

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(config_lines))

    print(f"\n💾 Config saved to: {output_file}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    long_stats = [s for (sym, d), s in stats_dict.items() if d == "long"]
    short_stats = [s for (sym, d), s in stats_dict.items() if d == "short"]

    if long_stats:
        avg_long_hold = np.mean([s["recommended_hold"] for s in long_stats])
        avg_long_saves = np.mean([s["left_on_table"] for s in long_stats])
        print(f"\n📈 LONG TRADES:")
        print(f"  Avg optimal hold:    {avg_long_hold:.0f} bars")
        print(f"  Avg savings/trade:   {avg_long_saves:.2f} USDT")
        print(f"  Total potential:     {avg_long_saves * sum(s['num_trades'] for s in long_stats):.2f} USDT")

    if short_stats:
        avg_short_hold = np.mean([s["recommended_hold"] for s in short_stats])
        avg_short_saves = np.mean([s["left_on_table"] for s in short_stats])
        print(f"\n📉 SHORT TRADES:")
        print(f"  Avg optimal hold:    {avg_short_hold:.0f} bars")
        print(f"  Avg savings/trade:   {avg_short_saves:.2f} USDT")
        print(f"  Total potential:     {avg_short_saves * sum(s['num_trades'] for s in short_stats):.2f} USDT")


if __name__ == "__main__":
    import sys
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "paper_trading_simulation_log.csv"

    stats = analyze_optimal_hold_time_per_symbol(csv_file)

    if stats:
        generate_optimal_exit_config(stats)
    else:
        print("❌ No statistics generated")

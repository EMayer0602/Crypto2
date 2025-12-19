"""
Analyze trade peak profit - shows when profit was highest for each trade.

This helps identify if we're exiting too late and giving back gains.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import Supertrend_5Min as st

def analyze_trade_peak_profit(trade, df):
    """
    Analyze a single trade to find when profit was at its peak.

    Returns dict with:
    - entry_price, exit_price, actual_pnl
    - peak_price, peak_pnl (max unrealized profit)
    - profit_left_on_table (how much we gave back)
    - bars_to_peak (when peak occurred)
    - peak_timestamp (when peak occurred)
    """
    symbol = trade.get("symbol") or trade.get("Symbol")
    direction = (trade.get("direction") or trade.get("Direction", "")).lower()
    entry_time = pd.to_datetime(trade.get("entry_time") or trade.get("Zeit"))
    exit_time = pd.to_datetime(trade.get("exit_time") or trade.get("ExitZeit"))
    entry_price = float(trade.get("entry_price") or trade.get("Entry"))
    exit_price = float(trade.get("exit_price") or trade.get("ExitPreis"))
    stake = float(trade.get("stake") or trade.get("Stake", 1000))
    actual_pnl = float(trade.get("pnl") or trade.get("PnL (USD)", 0))

    # Get price data during the trade
    trade_mask = (df.index >= entry_time) & (df.index <= exit_time)
    trade_df = df.loc[trade_mask].copy()

    if trade_df.empty:
        return None

    # Calculate unrealized PnL at each bar
    if direction == "long":
        # Long: profit when price rises
        trade_df["unrealized_pnl"] = ((trade_df["high"] - entry_price) / entry_price) * stake
        peak_price = trade_df["high"].max()
    else:
        # Short: profit when price falls
        trade_df["unrealized_pnl"] = ((entry_price - trade_df["low"]) / entry_price) * stake
        peak_price = trade_df["low"].min()

    # Find peak profit
    peak_pnl = trade_df["unrealized_pnl"].max()
    peak_idx = trade_df["unrealized_pnl"].idxmax()
    bars_to_peak = len(trade_df.loc[:peak_idx])

    # How much profit did we leave on the table?
    profit_left = peak_pnl - actual_pnl
    profit_left_pct = (profit_left / stake * 100) if stake > 0 else 0

    return {
        "symbol": symbol,
        "direction": direction,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "peak_price": peak_price,
        "peak_timestamp": peak_idx,
        "actual_pnl": actual_pnl,
        "peak_pnl": peak_pnl,
        "profit_left_on_table": profit_left,
        "profit_left_pct": profit_left_pct,
        "bars_to_peak": bars_to_peak,
        "total_bars": len(trade_df),
        "peak_at_pct": (bars_to_peak / len(trade_df) * 100) if len(trade_df) > 0 else 0,
    }


def analyze_all_trades(trades_csv="paper_trading_simulation_log.csv"):
    """Analyze all trades from simulation log."""
    if not Path(trades_csv).exists():
        print(f"❌ File not found: {trades_csv}")
        return

    trades_df = pd.read_csv(trades_csv)

    if trades_df.empty:
        print("❌ No trades found in CSV")
        return

    print(f"📊 Analyzing {len(trades_df)} trades...\n")

    results = []

    # Group by symbol to fetch data once per symbol
    for symbol in trades_df["symbol"].unique():
        print(f"Fetching data for {symbol}...")
        try:
            df = st.fetch_data(symbol, st.TIMEFRAME, st.LOOKBACK)
            df = df.copy()
        except Exception as e:
            print(f"  ⚠️  Failed to fetch {symbol}: {e}")
            continue

        symbol_trades = trades_df[trades_df["symbol"] == symbol]

        for _, trade in symbol_trades.iterrows():
            result = analyze_trade_peak_profit(trade, df)
            if result:
                results.append(result)

    if not results:
        print("❌ No trade analysis results")
        return

    results_df = pd.DataFrame(results)

    # Generate summary
    print("\n" + "="*80)
    print("PEAK PROFIT ANALYSIS - SUMMARY")
    print("="*80)

    # Overall statistics
    total_actual_pnl = results_df["actual_pnl"].sum()
    total_peak_pnl = results_df["peak_pnl"].sum()
    total_left = results_df["profit_left_on_table"].sum()

    print(f"\n💰 OVERALL:")
    print(f"  Actual PnL:              {total_actual_pnl:>12.2f} USDT")
    print(f"  Peak PnL (best case):    {total_peak_pnl:>12.2f} USDT")
    print(f"  Left on table:           {total_left:>12.2f} USDT ({total_left/abs(total_actual_pnl)*100:.1f}%)")

    # Separate by direction
    for direction in ["long", "short"]:
        dir_df = results_df[results_df["direction"] == direction]
        if dir_df.empty:
            continue

        dir_actual = dir_df["actual_pnl"].sum()
        dir_peak = dir_df["peak_pnl"].sum()
        dir_left = dir_df["profit_left_on_table"].sum()

        print(f"\n📈 {direction.upper()} TRADES ({len(dir_df)} trades):")
        print(f"  Actual PnL:              {dir_actual:>12.2f} USDT")
        print(f"  Peak PnL:                {dir_peak:>12.2f} USDT")
        print(f"  Left on table:           {dir_left:>12.2f} USDT ({dir_left/abs(dir_actual)*100 if dir_actual != 0 else 0:.1f}%)")

        avg_peak_at = dir_df["peak_at_pct"].mean()
        print(f"  Avg peak at:             {avg_peak_at:.1f}% of trade duration")

    # Worst offenders (trades that gave back most profit)
    print("\n" + "="*80)
    print("TOP 10 TRADES THAT GAVE BACK MOST PROFIT:")
    print("="*80)

    worst = results_df.nlargest(10, "profit_left_on_table")

    for _, row in worst.iterrows():
        print(f"\n{row['symbol']:10s} {row['direction']:5s}  Entry: {row['entry_time']}")
        print(f"  Actual PnL: {row['actual_pnl']:>10.2f} | Peak PnL: {row['peak_pnl']:>10.2f} | Left: {row['profit_left_on_table']:>10.2f} ({row['profit_left_pct']:.1f}%)")
        print(f"  Peak at bar {row['bars_to_peak']}/{row['total_bars']} ({row['peak_at_pct']:.1f}% of trade)")
        print(f"  Peak price: {row['peak_price']:.8f} | Exit price: {row['exit_price']:.8f}")

    # Save detailed results
    output_file = "trade_peak_analysis.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results saved to: {output_file}")

    # Recommendations
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS:")
    print("="*80)

    avg_peak_at = results_df["peak_at_pct"].mean()
    avg_left_pct = results_df["profit_left_pct"].mean()

    if avg_peak_at < 50:
        print(f"\n⚠️  Peak profit occurs early (avg {avg_peak_at:.1f}% into trade)")
        print("   Consider: TRAILING STOP or PARTIAL EXIT after peak")

    if avg_left_pct > 20:
        print(f"\n⚠️  Giving back {avg_left_pct:.1f}% of stake on average")
        print("   Consider: TIGHTER trailing stop or TIME-based exits")

    high_giveback = len(results_df[results_df["profit_left_pct"] > 50])
    if high_giveback > len(results_df) * 0.3:
        print(f"\n⚠️  {high_giveback}/{len(results_df)} trades gave back >50% of stake")
        print("   Consider: PROFIT TARGET or VOLATILITY-based exit")


if __name__ == "__main__":
    import sys
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "paper_trading_simulation_log.csv"
    analyze_all_trades(csv_file)

"""
List all trades from simulation log
"""
import json
from pathlib import Path

log_file = Path("paper_trading_simulation_log.json")

if not log_file.exists():
    print(f"Error: {log_file} not found")
    exit(1)

with open(log_file, 'r') as f:
    data = json.load(f)

print(f"Found {len(data)} trades:\n")
print(f"{'#':<4} {'Symbol':<12} {'Direction':<8} {'Entry Time':<30}")
print("=" * 70)

for i, trade in enumerate(data, 1):
    symbol = trade.get("symbol", "N/A")
    direction = trade.get("direction", "N/A")
    entry_time = trade.get("entry_time", "N/A")

    print(f"{i:<4} {symbol:<12} {direction:<8} {entry_time:<30}")

print("\nTo visualize a trade, copy the exact entry_time and run:")
print('python visualize_trades.py "<symbol>" "<direction>" "<entry_time>"')

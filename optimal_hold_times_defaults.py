"""
Optimal hold times based on peak profit analysis of 115 real trades.

Analysis results:
- Long trades: Avg optimal 5 bars (saves 59.82 USDT/trade, total 1,196 USDT)
- Short trades: Avg optimal 3 bars (saves 43.63 USDT/trade, total 4,145 USDT)
- Total potential savings: 5,340 USDT

Key insight: Peaks occur MUCH earlier than expected (2-10 bars vs 12-15)!
"""

# Optimal hold times from real data analysis (115 trades)
# Format: (symbol, direction) -> bars
OPTIMAL_HOLD_BARS = {
    # Real data from find_optimal_hold_times.py analysis

    # BTC/EUR
    ("BTC/EUR", "short"): 5,   # Peak at 73%, saves 22.94 USDT/trade

    # ETH/EUR
    ("ETH/EUR", "long"): 10,   # Peak at 91%, saves 52.90 USDT/trade
    ("ETH/EUR", "short"): 2,   # Peak at 51%, saves 27.75 USDT/trade

    # LINK/EUR
    ("LINK/EUR", "long"): 3,   # Peak at 53%, saves 35.94 USDT/trade
    ("LINK/EUR", "short"): 2,  # Peak at 53%, saves 24.93 USDT/trade

    # LUNC/USDT
    ("LUNC/USDT", "long"): 7,  # Peak at 70%, saves 72.58 USDT/trade
    ("LUNC/USDT", "short"): 2, # Peak at 41%, saves 70.75 USDT/trade (was 2.5, rounded to 2)

    # SOL/EUR
    ("SOL/EUR", "long"): 3,    # Peak at 51%, saves 39.67 USDT/trade
    ("SOL/EUR", "short"): 2,   # Peak at 37%, saves 44.10 USDT/trade (was 2.5, rounded to 2)

    # SUI/EUR
    ("SUI/EUR", "long"): 9,    # Peak at 53%, saves 65.42 USDT/trade
    ("SUI/EUR", "short"): 2,   # Peak at 61%, saves 26.04 USDT/trade

    # TNSR/USDC
    ("TNSR/USDC", "long"): 2,  # Peak at 46%, saves 98.40 USDT/trade
    ("TNSR/USDC", "short"): 2, # Peak at 36%, saves 51.72 USDT/trade

    # XRP/EUR
    ("XRP/EUR", "short"): 2,   # Peak at 53%, saves 27.10 USDT/trade (was 2.5, rounded to 2)

    # ZEC/USDC
    ("ZEC/USDC", "long"): 2,   # Peak at 29%, saves 53.83 USDT/trade
    ("ZEC/USDC", "short"): 4,  # Peak at 31%, saves 97.30 USDT/trade
}

def get_optimal_hold_bars(symbol: str, direction: str) -> int:
    """
    Get optimal hold time for symbol/direction based on real data.

    Returns conservative defaults if symbol/direction not found:
    - Long: 5 bars (from analysis average)
    - Short: 3 bars (from analysis average)
    """
    direction = direction.lower()
    key = (symbol, direction)

    if key in OPTIMAL_HOLD_BARS:
        return OPTIMAL_HOLD_BARS[key]

    # Fallback to analysis averages
    return 5 if direction == "long" else 3

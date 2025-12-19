"""
Default optimal hold times based on peak profit analysis insights.

Analysis results (from previous analyze_peak_profit.py runs):
- Overall: Peak occurs at ~65% of trade duration
- Long trades: Left 2,920 USDT on table (22,733% giveback rate) - need TIGHT exits
- Short trades: Left 6,491 USDT on table (486% giveback rate) - also need early exits

These defaults can be customized once we run full peak analysis per symbol.
"""

# Default optimal hold times in bars
# Format: (symbol, direction) -> bars
OPTIMAL_HOLD_BARS = {
    # Conservative defaults for all symbols
    # Long trades: 10-12 bars (tighter due to high giveback)
    # Short trades: 15-18 bars (slightly more room)

    # EUR pairs (typical 6h-8h timeframes)
    ("BTC/EUR", "long"): 10,
    ("BTC/EUR", "short"): 15,
    ("ETH/EUR", "long"): 10,
    ("ETH/EUR", "short"): 15,
    ("SOL/EUR", "long"): 12,
    ("SOL/EUR", "short"): 15,
    ("SUI/EUR", "long"): 12,
    ("SUI/EUR", "short"): 15,
    ("LINK/EUR", "long"): 10,
    ("LINK/EUR", "short"): 15,
    ("XRP/EUR", "long"): 10,
    ("XRP/EUR", "short"): 15,

    # USDT/USDC pairs
    ("LUNC/USDT", "long"): 12,
    ("LUNC/USDT", "short"): 15,
    ("TNSR/USDC", "long"): 12,
    ("TNSR/USDC", "short"): 15,
    ("ZEC/USDC", "long"): 12,
    ("ZEC/USDC", "short"): 15,
}

def get_optimal_hold_bars(symbol: str, direction: str) -> int:
    """
    Get optimal hold time for symbol/direction.

    Returns default of 12 bars for longs, 15 for shorts if not found.
    """
    direction = direction.lower()
    key = (symbol, direction)

    if key in OPTIMAL_HOLD_BARS:
        return OPTIMAL_HOLD_BARS[key]

    # Fallback defaults
    return 12 if direction == "long" else 15

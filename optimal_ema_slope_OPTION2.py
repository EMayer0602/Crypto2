"""
OPTION 2: Stricter Slope Threshold (0.1% instead of 0%)

Stronger uptrend requirement for long entries.

Strategy:
- EMA-20 Slope ≥ 0.1% (stricter than baseline 0%)
- Only enter longs in actively rising trends

Expected: Higher win rate, fewer trades (only strong uptrends)
"""

USE_EMA_SLOPE_FILTER = True
USE_PRICE_ABOVE_EMA_FILTER = False

OPTIMAL_EMA_PARAMS = {
    ("ETH/EUR", "long"): (20, 0.1),      # Changed from 0.0 to 0.1
    ("BTC/EUR", "long"): (20, 0.1),      # Changed from 0.0 to 0.1
    ("SOL/EUR", "long"): (20, 0.1),      # Changed from 0.0 to 0.1
    ("XRP/EUR", "long"): (20, 0.1),      # Changed from 0.0 to 0.1
    ("LINK/EUR", "long"): (20, 0.1),     # Changed from 0.0 to 0.1
    ("LUNC/USDT", "long"): (20, 0.1),    # Changed from 0.0 to 0.1
    ("SUI/EUR", "long"): (20, 0.1),      # Changed from 0.0 to 0.1
    ("TNSR/USDC", "long"): (20, 0.1),    # Changed from 0.0 to 0.1
    ("ZEC/USDC", "long"): (20, 0.1),     # Changed from 0.0 to 0.1
}

DEFAULT_EMA_PERIOD = 20
DEFAULT_SLOPE_THRESHOLD = 0.1  # Changed from 0.0

def get_ema_slope_params(symbol: str, direction: str) -> tuple:
    direction = direction.lower()
    if direction != "long":
        return None, None
    key = (symbol, direction)
    if key in OPTIMAL_EMA_PARAMS:
        return OPTIMAL_EMA_PARAMS[key]
    return DEFAULT_EMA_PERIOD, DEFAULT_SLOPE_THRESHOLD

def should_filter_entry_by_ema_slope(symbol: str, direction: str, current_slope: float) -> bool:
    if not USE_EMA_SLOPE_FILTER:
        return False
    ema_period, slope_threshold = get_ema_slope_params(symbol, direction)
    if ema_period is None or slope_threshold is None:
        return False
    if current_slope < slope_threshold:
        return True
    return False

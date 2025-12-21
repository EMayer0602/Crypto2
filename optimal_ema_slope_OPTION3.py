"""
OPTION 3: Multi-Timeframe EMA Confirmation

Dual EMA confirmation: Both EMA-20 and EMA-50 must be rising.

Strategy:
- EMA-20 Slope ≥ 0% (short-term trend)
- EMA-50 Slope ≥ 0% (medium-term trend)
- Both must confirm uptrend

Expected: Strongest trend confirmation, fewer but highest quality trades
"""

USE_EMA_SLOPE_FILTER = True
USE_PRICE_ABOVE_EMA_FILTER = False
USE_DUAL_EMA_FILTER = True  # NEW: Dual EMA confirmation

# Primary EMA (short-term)
OPTIMAL_EMA_PARAMS = {
    ("ETH/EUR", "long"): (20, 0.0),
    ("BTC/EUR", "long"): (20, 0.0),
    ("SOL/EUR", "long"): (20, 0.0),
    ("XRP/EUR", "long"): (20, 0.0),
    ("LINK/EUR", "long"): (20, 0.0),
    ("LUNC/USDT", "long"): (20, 0.0),
    ("SUI/EUR", "long"): (20, 0.0),
    ("TNSR/USDC", "long"): (20, 0.0),
    ("ZEC/USDC", "long"): (20, 0.0),
}

# Secondary EMA (medium-term confirmation)
SECONDARY_EMA_PERIOD = 50
SECONDARY_SLOPE_THRESHOLD = 0.0

DEFAULT_EMA_PERIOD = 20
DEFAULT_SLOPE_THRESHOLD = 0.0

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

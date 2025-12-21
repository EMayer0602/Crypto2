"""
BASELINE Configuration - Current best performing setup.

EMA-20 Slope Filter with 0% threshold.
Results (11.23-12.21):
- Long PnL: +3,956 USDT
- Long Win Rate: 43.69%
- Total PnL: +7,952 USDT
"""

# Default EMA slope filter settings
USE_EMA_SLOPE_FILTER = True
USE_PRICE_ABOVE_EMA_FILTER = False  # Option 1: Price > EMA

# EMA configuration
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

DEFAULT_EMA_PERIOD = 20
DEFAULT_SLOPE_THRESHOLD = 0.0  # 0% = not falling

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

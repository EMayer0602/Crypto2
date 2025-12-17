"""
Local Configuration Override Template

Copy this file to 'config_local.py' and modify the values you want to override.
The config_local.py file is gitignored and won't cause conflicts.

Example usage:
1. Copy: cp config_local.example.py config_local.py
2. Edit config_local.py with your test settings
3. Run your scripts normally - they'll use your local settings

Only include variables you want to override - leave the rest commented out.
"""

# ============================================================================
# SYMBOLS AND TESTING
# ============================================================================
# SYMBOLS = ["BTC/USDT", "ETH/USDT"]
# ACTIVE_INDICATORS = ["jma", "kama"]

# ============================================================================
# TIMEFRAME SETTINGS
# ============================================================================
# TIMEFRAME = "1h"
# BARS_PER_DAY = 24  # Adjust based on timeframe: 5min=288, 15min=96, 1h=24

# ============================================================================
# FILTERS (Enable/Disable)
# ============================================================================
# USE_HIGHER_TIMEFRAME_FILTER = False
# USE_MOMENTUM_FILTER = False
# USE_JMA_TREND_FILTER = False
# USE_MIN_HOLD_FILTER = False
# USE_BREAKOUT_FILTER = False

# ============================================================================
# JMA TREND FILTER SETTINGS
# ============================================================================
# JMA_TREND_LENGTH = 30
# JMA_TREND_PHASE = 0
# JMA_TREND_THRESH_UP = 0.0002
# JMA_TREND_THRESH_DOWN = -0.0002

# ============================================================================
# CAPITAL SETTINGS
# ============================================================================
# START_EQUITY = 10000.0
# STAKE_DIVISOR = 10

# ============================================================================
# BACKTEST PARAMETERS
# ============================================================================
# MIN_HOLD_BAR_VALUES = [0, 12, 24, 48]
# ATR_STOP_MULTS = [None, 1.0, 1.5, 2.0, 2.5]

# ============================================================================
# HIGHER TIMEFRAME SETTINGS
# ============================================================================
# HTF_FACTOR = 3.0
# HTF_LENGTH = 14

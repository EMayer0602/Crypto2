# Local Configuration - READY TO USE
# Settings match overall_best_detailed.html backtest exactly

# ============================================================================
# PAPER TRADING SETTINGS
# ============================================================================
PAPER_TRADING_MODE = True
LIVE_TRADING_SYMBOLS = ["BTC/USDC", "ETH/USDC", "SOL/USDC", "BNB/USDC", "XRP/USDC",
                        "ADA/USDC", "AVAX/USDC", "DOGE/USDC", "DOT/USDC", "MATIC/USDC",
                        "LINK/USDC", "UNI/USDC", "ATOM/USDC", "LTC/USDC", "ETC/USDC",
                        "FIL/USDC", "NEAR/USDC", "APT/USDC", "ARB/USDC", "OP/USDC"]
LIVE_TRADING_TIMEFRAME = "5m"
LIVE_TRADING_INTERVAL_SECONDS = 300

# ============================================================================
# HISTORICAL DATA
# ============================================================================
LOOKBACK = 1000

# ============================================================================
# TRADING DIRECTION - BOTH ENABLED
# ============================================================================
ENABLE_LONGS = True
ENABLE_SHORTS = True

# ============================================================================
# FILTERS - MATCH BACKTEST
# ============================================================================
USE_MIN_HOLD_FILTER = True
USE_HIGHER_TIMEFRAME_FILTER = True
USE_MOMENTUM_FILTER = False
USE_JMA_TREND_FILTER = False
USE_BREAKOUT_FILTER = False

# ============================================================================
# CAPITAL SETTINGS
# ============================================================================
START_EQUITY = 14000.0
RISK_FRACTION = 1
STAKE_DIVISOR = 14
FEE_RATE = 0.001

# ============================================================================
# EXIT STRATEGY - TIME-BASED EXITS
# ============================================================================
USE_TIME_BASED_EXIT = True  # Enable time-based exits based on peak profit analysis

# Local Configuration - EXACT BACKTEST MATCH
# Settings from overall_best_detailed.html (2025-12-19 06:28:11 CET)
#
# IMPORTANT: This backtest uses SAVED PARAMETERS from best_params_overall.csv
# Each symbol has specific settings (timeframe, JMA params, MinHold, ATR stop)

# ============================================================================
# PAPER TRADING SETTINGS - USE SYMBOLS FROM BACKTEST
# ============================================================================
PAPER_TRADING_MODE = True

# EXACT symbols from backtest (Long + Short)
LIVE_TRADING_SYMBOLS = [
    # Long trades (3 symbols)
    "LUNC/USDT",   # JMA 12h, Length=20, Phase=0.0, MinHold=12d
    "TNSR/USDC",   # JMA 6h, Length=30, Phase=50.0, ATRStop=1.0, MinHold=24d
    "ZEC/USDC",    # JMA 4h, Length=30, Phase=-50.0, MinHold=24d

    # Short trades (7 symbols) - ALL EUR PAIRS!
    "BTC/EUR",     # JMA 8h, Length=30, Phase=50.0, MinHold=12d
    "ETH/EUR",     # JMA 8h, Length=20, Phase=0.0, ATRStop=1.5, MinHold=24d
    "LINK/EUR",    # JMA 8h, Length=30, Phase=0.0, MinHold=24d
    "SOL/EUR",     # JMA 8h, Length=50, Phase=-50.0, MinHold=24d
    "SUI/EUR",     # JMA 8h, Length=20, Phase=50.0, MinHold=24d
    "XRP/EUR",     # JMA 12h, Length=20, Phase=0.0, MinHold=24d
]

# Note: Timeframes vary by symbol (4h, 6h, 8h, 12h)
# paper_trader.py will read the correct timeframe from best_params_overall.csv
LIVE_TRADING_TIMEFRAME = "8h"  # Most common timeframe in backtest

# Run more frequently to check for signals (will respect symbol-specific timeframes)
LIVE_TRADING_INTERVAL_SECONDS = 1800  # 30 minutes

# ============================================================================
# HISTORICAL DATA - Use enough data for higher timeframes
# ============================================================================
LOOKBACK = 2000  # Higher timeframes need more data

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
# CAPITAL SETTINGS - MATCH BACKTEST
# ============================================================================
START_EQUITY = 14000.0
RISK_FRACTION = 1
STAKE_DIVISOR = 14
FEE_RATE = 0.001

# ============================================================================
# IMPORTANT NOTES
# ============================================================================
# 1. Short trades use EUR pairs (BTC/EUR, ETH/EUR, etc.)
# 2. Each symbol has different timeframe and JMA parameters
# 3. Parameters are loaded from best_params_overall.csv
# 4. MinHold values: 12d or 24d (in bars of the symbol's timeframe)
# 5. Some symbols use ATR stops, others use trend flip only

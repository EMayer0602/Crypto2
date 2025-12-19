"""
Local Configuration Override Template - BACKTEST MATCHING VERSION

Copy this file to 'config_local.py' to test with identical settings as overall_best_detailed.html

Example usage:
1. Copy: cp config_local.example.py config_local.py
2. Run: python paper_trader.py --test
3. Check: paper_trading_simulation_summary.html

CRITICAL SETTINGS FOR BACKTEST MATCH:
- USE_JMA_TREND_FILTER must be False
- ENABLE_SHORTS must be True
- Use timeframe format: "5m", "1h", "4h" (NOT "5min", "1hour")
"""

# ============================================================================
# PAPER TRADING SETTINGS
# ============================================================================
PAPER_TRADING_MODE = True
LIVE_TRADING_SYMBOLS = ["BTC/USDC", "ETH/USDC", "SOL/USDC", "BNB/USDC", "XRP/USDC",
                        "ADA/USDC", "AVAX/USDC", "DOGE/USDC", "DOT/USDC", "MATIC/USDC",
                        "LINK/USDC", "UNI/USDC", "ATOM/USDC", "LTC/USDC", "ETC/USDC",
                        "FIL/USDC", "NEAR/USDC", "APT/USDC", "ARB/USDC", "OP/USDC"]
LIVE_TRADING_TIMEFRAME = "5m"  # Use "5m", "1h", "4h" format (NOT "5min", "1hour")
LIVE_TRADING_INTERVAL_SECONDS = 300  # 5 minutes

# ============================================================================
# HISTORICAL DATA
# ============================================================================
LOOKBACK = 1000  # Fetch enough historical data for testing

# ============================================================================
# TRADING DIRECTION (BACKTEST MATCH)
# ============================================================================
ENABLE_LONGS = True
ENABLE_SHORTS = True  # CRITICAL: Must be True for both long and short trades

# ============================================================================
# FILTERS (BACKTEST MATCH - overall_best_detailed.html)
# ============================================================================
USE_MIN_HOLD_FILTER = True
USE_HIGHER_TIMEFRAME_FILTER = True
USE_MOMENTUM_FILTER = False
USE_JMA_TREND_FILTER = False  # CRITICAL: Must be False to match backtest
USE_BREAKOUT_FILTER = False

# ============================================================================
# CAPITAL SETTINGS (BACKTEST MATCH)
# ============================================================================
START_EQUITY = 14000.0
RISK_FRACTION = 1
STAKE_DIVISOR = 14
FEE_RATE = 0.001

"""Verify configuration matches backtest settings."""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import settings
import Supertrend_5Min as st

# Check for config_local overrides
try:
    import config_local as cl
    has_local = True
except ImportError:
    cl = None
    has_local = False

print("=" * 70)
print("CONFIGURATION VERIFICATION - Backtest Settings Match")
print("=" * 70)

print("\n📊 TRADING DIRECTION:")
print(f"  ENABLE_LONGS:  {getattr(cl, 'ENABLE_LONGS', st.ENABLE_LONGS) if has_local else st.ENABLE_LONGS}")
print(f"  ENABLE_SHORTS: {getattr(cl, 'ENABLE_SHORTS', st.ENABLE_SHORTS) if has_local else st.ENABLE_SHORTS}")

print("\n🎯 FILTERS:")
print(f"  USE_MIN_HOLD_FILTER:        {getattr(cl, 'USE_MIN_HOLD_FILTER', st.USE_MIN_HOLD_FILTER) if has_local else st.USE_MIN_HOLD_FILTER}")
print(f"  USE_HIGHER_TIMEFRAME_FILTER: {getattr(cl, 'USE_HIGHER_TIMEFRAME_FILTER', st.USE_HIGHER_TIMEFRAME_FILTER) if has_local else st.USE_HIGHER_TIMEFRAME_FILTER}")
print(f"  USE_MOMENTUM_FILTER:        {getattr(cl, 'USE_MOMENTUM_FILTER', st.USE_MOMENTUM_FILTER) if has_local else st.USE_MOMENTUM_FILTER}")
print(f"  USE_JMA_TREND_FILTER:       {getattr(cl, 'USE_JMA_TREND_FILTER', st.USE_JMA_TREND_FILTER) if has_local else st.USE_JMA_TREND_FILTER} ⚠️ SHOULD BE False")
print(f"  USE_BREAKOUT_FILTER:        {getattr(cl, 'USE_BREAKOUT_FILTER', st.USE_BREAKOUT_FILTER) if has_local else st.USE_BREAKOUT_FILTER}")

print("\n💰 CAPITAL & RISK:")
print(f"  START_EQUITY:   {getattr(cl, 'START_EQUITY', st.START_EQUITY) if has_local else st.START_EQUITY}")
print(f"  RISK_FRACTION:  {getattr(cl, 'RISK_FRACTION', st.RISK_FRACTION) if has_local else st.RISK_FRACTION}")
print(f"  STAKE_DIVISOR:  {getattr(cl, 'STAKE_DIVISOR', st.STAKE_DIVISOR) if has_local else st.STAKE_DIVISOR}")
print(f"  FEE_RATE:       {getattr(cl, 'FEE_RATE', st.FEE_RATE) if has_local else st.FEE_RATE}")

if has_local:
    print("\n📝 PAPER TRADING (config_local.py):")
    print(f"  PAPER_TRADING_MODE:    {getattr(cl, 'PAPER_TRADING_MODE', 'Not set')}")
    print(f"  TIMEFRAME:             {getattr(cl, 'LIVE_TRADING_TIMEFRAME', 'Not set')}")
    print(f"  LOOKBACK:              {getattr(cl, 'LOOKBACK', 'Not set')}")
    print(f"  SYMBOLS:               {len(getattr(cl, 'LIVE_TRADING_SYMBOLS', []))} symbols")

print("\n" + "=" * 70)
print("✅ VERIFICATION COMPLETE")
print("=" * 70)

print("\n⚠️  KEY REQUIREMENTS FOR BACKTEST MATCH:")
print("  1. USE_JMA_TREND_FILTER must be False")
print("  2. ENABLE_SHORTS must be True")
print("  3. All other filter settings should match backtest HTML")

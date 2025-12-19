"""Check if settings match backtest configuration."""
import re

def extract_setting(filename, pattern, default="Not found"):
    """Extract a setting value from a Python file."""
    try:
        with open(filename, 'r') as f:
            content = f.read()
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                return match.group(1).strip()
    except FileNotFoundError:
        return "File not found"
    return default

print("=" * 70)
print("BACKTEST SETTINGS VERIFICATION")
print("=" * 70)

# Check main strategy file
st_file = "Supertrend_5Min.py"
config_file = "config_local.py"

print("\n📊 Supertrend_5Min.py Settings:")
print("-" * 70)

settings = {
    "ENABLE_LONGS": r"^ENABLE_LONGS\s*=\s*(.+)",
    "ENABLE_SHORTS": r"^ENABLE_SHORTS\s*=\s*(.+)",
    "USE_MIN_HOLD_FILTER": r"^USE_MIN_HOLD_FILTER\s*=\s*(.+)",
    "USE_HIGHER_TIMEFRAME_FILTER": r"^USE_HIGHER_TIMEFRAME_FILTER\s*=\s*(.+)",
    "USE_MOMENTUM_FILTER": r"^USE_MOMENTUM_FILTER\s*=\s*(.+)",
    "USE_JMA_TREND_FILTER": r"^USE_JMA_TREND_FILTER\s*=\s*(.+)",
    "USE_BREAKOUT_FILTER": r"^USE_BREAKOUT_FILTER\s*=\s*(.+)",
    "START_EQUITY": r"^START_EQUITY\s*=\s*(.+)",
    "STAKE_DIVISOR": r"^STAKE_DIVISOR\s*=\s*(.+)",
}

for name, pattern in settings.items():
    value = extract_setting(st_file, pattern)
    status = ""
    if name == "USE_JMA_TREND_FILTER" and "False" in value:
        status = " ✅ CORRECT (must be False for backtest match)"
    elif name == "ENABLE_SHORTS" and "True" in value:
        status = " ✅ CORRECT (must be True for shorts)"
    print(f"  {name:30s} = {value}{status}")

print("\n📝 config_local.py Settings:")
print("-" * 70)

local_settings = {
    "PAPER_TRADING_MODE": r"^PAPER_TRADING_MODE\s*=\s*(.+)",
    "LIVE_TRADING_TIMEFRAME": r"^LIVE_TRADING_TIMEFRAME\s*=\s*(.+)",
    "LOOKBACK": r"^LOOKBACK\s*=\s*(.+)",
    "ENABLE_LONGS": r"^ENABLE_LONGS\s*=\s*(.+)",
    "ENABLE_SHORTS": r"^ENABLE_SHORTS\s*=\s*(.+)",
}

for name, pattern in local_settings.items():
    value = extract_setting(config_file, pattern)
    print(f"  {name:30s} = {value}")

print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

# Critical checks
jma_filter = extract_setting(st_file, r"^USE_JMA_TREND_FILTER\s*=\s*(.+)")
shorts = extract_setting(st_file, r"^ENABLE_SHORTS\s*=\s*(.+)")

print("\n⚠️  CRITICAL REQUIREMENTS:")
print(f"  1. USE_JMA_TREND_FILTER = False  {'✅' if 'False' in jma_filter else '❌ FIX NEEDED'}")
print(f"  2. ENABLE_SHORTS = True          {'✅' if 'True' in shorts else '❌ FIX NEEDED'}")

print("\n📌 To run test on Windows:")
print("  1. Run: python paper_trader.py --test")
print("  2. Or: run_live_trader.bat")
print("  3. Check: paper_trading_simulation_summary.html")
print("\n" + "=" * 70)

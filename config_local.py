# ============================================================================
# SWEEP SETTINGS - FÜR SIMULATION DEAKTIVIERT
# ============================================================================
RUN_PARAMETER_SWEEP = False
RUN_SAVED_PARAMS = False
RUN_OVERALL_BEST = True

# NUR JMA und KAMA (beste Ergebnisse)
ACTIVE_INDICATORS = ["jma", "kama"]

# ============================================================================
# HTF TIMEFRAMES - NUR DIE BESTEN (4h, 6h, 8h, 12h statt 3h-24h)
# ============================================================================
HTF_CANDIDATES = ["4h", "6h", "8h", "12h"]

# ============================================================================
# FILTERS
# ============================================================================
USE_HIGHER_TIMEFRAME_FILTER = True
USE_MIN_HOLD_FILTER = True
USE_MOMENTUM_FILTER = False
USE_BREAKOUT_FILTER = False

# ============================================================================
# MIN HOLD - FOKUS AUF 2 (optimal)
# ============================================================================
MIN_HOLD_DAY_VALUES = [0, 2]

# ============================================================================
# ATR STOPS - REDUZIERT
# ============================================================================
ATR_STOP_MULTS = [None, 1.5]

# ============================================================================
# TIME-BASED EXIT - FOKUS AUF 2 BARS (optimal)
# ============================================================================
USE_TIME_BASED_EXIT = True
TIME_EXIT_BAR_VALUES = [0, 2, 3, 4, 5]  # 0=trend flip, 2-5=time exit bars

# ============================================================================
# TRADING DIRECTION
# ============================================================================
ENABLE_LONGS = True
ENABLE_SHORTS = True

# ============================================================================
# CAPITAL
# ============================================================================
START_EQUITY = 16000.0
STAKE_DIVISOR = 5  # ~3200 per trade to match Crypto2Profitable stakes
FEE_RATE = 0.001

# ============================================================================
# DATA - Für 1-Jahr Simulation ab 2025-01-01, Warm-up ab 2024-09-01
# ============================================================================
LOOKBACK = 12000  # ~16 Monate bei 1h Timeframe
OHLCV_CACHE_DIR = "ohlcv_cache"  # Verzeichnis mit CSV-Cache-Dateien
CACHE_ONLY = True  # NUR Cache verwenden, keine Binance API-Calls (für Offline-Simulation)

# ============================================================================
# PAPER TRADING
# ============================================================================
MAX_OPEN_POSITIONS = 8

# ============================================================================
# SYMBOLS
# ============================================================================
SYMBOLS = [
    "BTC/EUR",
    "ETH/EUR",
    "XRP/EUR",
    "SOL/EUR",
    "LINK/EUR",
    "SUI/EUR",
    "LUNC/USDT",
    "TNSR/USDC",
    "ZEC/USDC",
]

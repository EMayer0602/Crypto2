"""Optimal hold times defaults for time-based exit strategy.

This module provides default optimal hold bars based on backtesting results.
The time-based exit closes positions after a maximum number of bars to
capture profits before reversal.
"""
from typing import Dict, Optional, Tuple

# Default optimal hold bars by (symbol, indicator, direction, htf)
# Format: {(symbol, indicator, direction): optimal_bars}
# These are derived from backtesting and represent the sweet spot for exits

# Default fallback values
DEFAULT_OPTIMAL_HOLD_BARS = {
    "supertrend": 36,  # ~1.5 days on 1h
    "htf_crossover": 36,
    "jma": 48,  # Slower indicator, longer holds
    "kama": 40,
}

# Specific optimal hold bars from backtesting
OPTIMAL_HOLD_TIMES: Dict[Tuple[str, str, str], int] = {
    # BTC/EUR
    ("BTC/EUR", "supertrend", "long"): 24,
    ("BTC/EUR", "supertrend", "short"): 36,
    ("BTC/EUR", "htf_crossover", "long"): 24,
    ("BTC/EUR", "htf_crossover", "short"): 36,
    ("BTC/EUR", "jma", "long"): 48,
    ("BTC/EUR", "jma", "short"): 36,
    ("BTC/EUR", "kama", "long"): 24,
    ("BTC/EUR", "kama", "short"): 36,

    # ETH/EUR
    ("ETH/EUR", "supertrend", "long"): 24,
    ("ETH/EUR", "supertrend", "short"): 36,
    ("ETH/EUR", "htf_crossover", "long"): 24,
    ("ETH/EUR", "htf_crossover", "short"): 36,
    ("ETH/EUR", "jma", "long"): 48,
    ("ETH/EUR", "jma", "short"): 48,
    ("ETH/EUR", "kama", "long"): 24,
    ("ETH/EUR", "kama", "short"): 40,

    # XRP/EUR
    ("XRP/EUR", "supertrend", "long"): 36,
    ("XRP/EUR", "supertrend", "short"): 24,
    ("XRP/EUR", "htf_crossover", "long"): 36,
    ("XRP/EUR", "htf_crossover", "short"): 24,
    ("XRP/EUR", "jma", "long"): 48,
    ("XRP/EUR", "jma", "short"): 48,
    ("XRP/EUR", "kama", "long"): 24,
    ("XRP/EUR", "kama", "short"): 48,

    # LINK/EUR
    ("LINK/EUR", "supertrend", "long"): 24,
    ("LINK/EUR", "supertrend", "short"): 36,
    ("LINK/EUR", "htf_crossover", "long"): 24,
    ("LINK/EUR", "htf_crossover", "short"): 36,
    ("LINK/EUR", "jma", "long"): 24,
    ("LINK/EUR", "jma", "short"): 48,
    ("LINK/EUR", "kama", "long"): 24,
    ("LINK/EUR", "kama", "short"): 36,

    # LUNC/USDT
    ("LUNC/USDT", "supertrend", "long"): 36,
    ("LUNC/USDT", "supertrend", "short"): 24,
    ("LUNC/USDT", "htf_crossover", "long"): 36,
    ("LUNC/USDT", "htf_crossover", "short"): 24,
    ("LUNC/USDT", "jma", "long"): 60,
    ("LUNC/USDT", "jma", "short"): 48,
    ("LUNC/USDT", "kama", "long"): 24,
    ("LUNC/USDT", "kama", "short"): 40,

    # SOL/EUR
    ("SOL/EUR", "supertrend", "long"): 24,
    ("SOL/EUR", "supertrend", "short"): 48,
    ("SOL/EUR", "htf_crossover", "long"): 24,
    ("SOL/EUR", "htf_crossover", "short"): 48,
    ("SOL/EUR", "jma", "long"): 48,
    ("SOL/EUR", "jma", "short"): 48,
    ("SOL/EUR", "kama", "long"): 24,
    ("SOL/EUR", "kama", "short"): 36,

    # SUI/EUR
    ("SUI/EUR", "supertrend", "long"): 24,
    ("SUI/EUR", "supertrend", "short"): 48,
    ("SUI/EUR", "htf_crossover", "long"): 24,
    ("SUI/EUR", "htf_crossover", "short"): 48,
    ("SUI/EUR", "jma", "long"): 24,
    ("SUI/EUR", "jma", "short"): 48,
    ("SUI/EUR", "kama", "long"): 36,
    ("SUI/EUR", "kama", "short"): 48,

    # TNSR/USDC
    ("TNSR/USDC", "supertrend", "long"): 36,
    ("TNSR/USDC", "supertrend", "short"): 36,
    ("TNSR/USDC", "htf_crossover", "long"): 36,
    ("TNSR/USDC", "htf_crossover", "short"): 36,
    ("TNSR/USDC", "jma", "long"): 36,
    ("TNSR/USDC", "jma", "short"): 48,
    ("TNSR/USDC", "kama", "long"): 48,
    ("TNSR/USDC", "kama", "short"): 48,

    # ZEC/USDC
    ("ZEC/USDC", "supertrend", "long"): 36,
    ("ZEC/USDC", "supertrend", "short"): 36,
    ("ZEC/USDC", "htf_crossover", "long"): 36,
    ("ZEC/USDC", "htf_crossover", "short"): 36,
    ("ZEC/USDC", "jma", "long"): 36,
    ("ZEC/USDC", "jma", "short"): 48,
    ("ZEC/USDC", "kama", "long"): 36,
    ("ZEC/USDC", "kama", "short"): 36,
}


def get_optimal_hold_bars(
    symbol: str,
    indicator: str,
    direction: str,
    htf: Optional[str] = None,
) -> int:
    """Get optimal hold bars for a given strategy configuration.

    Args:
        symbol: Trading pair (e.g., 'BTC/EUR')
        indicator: Indicator name (e.g., 'jma', 'supertrend')
        direction: Trade direction ('long' or 'short')
        htf: Higher timeframe (optional, for future use)

    Returns:
        Optimal number of bars to hold before time-based exit
    """
    key = (symbol.upper(), indicator.lower(), direction.lower())

    # Try exact match first
    if key in OPTIMAL_HOLD_TIMES:
        return OPTIMAL_HOLD_TIMES[key]

    # Fallback to indicator default
    indicator_lower = indicator.lower()
    if indicator_lower in DEFAULT_OPTIMAL_HOLD_BARS:
        return DEFAULT_OPTIMAL_HOLD_BARS[indicator_lower]

    # Ultimate fallback
    return 36

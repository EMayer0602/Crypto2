"""
Simple HTF Crossover Strategy

Entry: Close crosses HTF indicator from below (Long) or above (Short)
Exit: Close crosses HTF indicator in opposite direction

This is MUCH simpler than complex exit strategies with ATR stops,
trailing stops, and market regime detection.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict


def check_htf_crossover_entry(df: pd.DataFrame, direction: str) -> tuple[bool, str]:
    """
    Check if Close has crossed HTF indicator for entry.

    For Long: Close[t-1] < HTF[t-1] AND Close[t] > HTF[t]
    For Short: Close[t-1] > HTF[t-1] AND Close[t] < HTF[t]

    Returns: (signal, reason)
    """
    if len(df) < 2:
        return False, "Not enough bars"

    if "htf_indicator" not in df.columns:
        return False, "HTF indicator not available"

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    close_curr = float(curr["close"])
    close_prev = float(prev["close"])

    htf_curr = float(curr["htf_indicator"]) if pd.notna(curr["htf_indicator"]) else None
    htf_prev = float(prev["htf_indicator"]) if pd.notna(prev["htf_indicator"]) else None

    if htf_curr is None or htf_prev is None:
        return False, "HTF indicator has NaN values"

    long_mode = direction == "long"

    if long_mode:
        # Long entry: Close crosses above HTF
        crossed_above = close_prev < htf_prev and close_curr > htf_curr
        if crossed_above:
            return True, "Close crossed HTF upward"
        else:
            return False, "No HTF crossover"
    else:
        # Short entry: Close crosses below HTF
        crossed_below = close_prev > htf_prev and close_curr < htf_curr
        if crossed_below:
            return True, "Close crossed HTF downward"
        else:
            return False, "No HTF crossover"


def check_htf_crossover_exit(position: Dict, df: pd.DataFrame) -> Optional[Dict]:
    """
    Check if Close has crossed HTF indicator for exit.

    For Long position: Close crosses below HTF → Exit
    For Short position: Close crosses above HTF → Exit

    Returns: Exit dict with exit_price, reason, fees, pnl or None
    """
    if len(df) < 2:
        return None

    if "htf_indicator" not in df.columns:
        return None

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    direction = str(position.get("direction", "long")).lower()
    long_mode = direction == "long"
    entry_price = float(position.get("entry_price", 0.0) or 0.0)
    stake = float(position.get("stake", 0.0) or 0.0)

    close_curr = float(curr["close"])
    close_prev = float(prev["close"])

    htf_curr = float(curr["htf_indicator"]) if pd.notna(curr["htf_indicator"]) else None
    htf_prev = float(prev["htf_indicator"]) if pd.notna(prev["htf_indicator"]) else None

    if htf_curr is None or htf_prev is None:
        return None

    exit_triggered = False

    if long_mode:
        # Long exit: Close crosses below HTF
        crossed_below = close_prev > htf_prev and close_curr < htf_curr
        exit_triggered = crossed_below
    else:
        # Short exit: Close crosses above HTF
        crossed_above = close_prev < htf_prev and close_curr > htf_curr
        exit_triggered = crossed_above

    if not exit_triggered:
        return None

    # Calculate PnL
    import Supertrend_5Min as st
    exit_price = close_curr
    gross_pnl = (exit_price - entry_price) / entry_price * stake if long_mode else (entry_price - exit_price) / entry_price * stake
    fees = stake * st.FEE_RATE * 2.0
    pnl = gross_pnl - fees

    return {
        "exit_price": exit_price,
        "reason": "HTF crossover exit",
        "fees": fees,
        "pnl": pnl,
    }


def evaluate_entry_htf_crossover(df: pd.DataFrame, direction: str) -> tuple[bool, str]:
    """
    Simplified entry evaluation - ONLY HTF crossover, no filters.

    This replaces the old evaluate_entry() which used trend_flag and filters.
    """
    return check_htf_crossover_entry(df, direction)


def evaluate_exit_htf_crossover(position: Dict, df: pd.DataFrame) -> Optional[Dict]:
    """
    Simplified exit evaluation - ONLY HTF crossover, no ATR stops or min hold.

    This replaces the old evaluate_exit() which used ATR stops, trend flips, min hold.
    """
    return check_htf_crossover_exit(position, df)


# Compatibility wrapper for compare script
def evaluate_exit_simple_htf(
    position: Dict,
    df: pd.DataFrame,
    atr_mult: Optional[float] = None,  # Ignored
    min_hold_bars: int = 0  # Ignored
) -> Optional[Dict]:
    """
    Wrapper for comparison script compatibility.
    Ignores atr_mult and min_hold_bars - uses ONLY HTF crossover.
    """
    return evaluate_exit_htf_crossover(position, df)

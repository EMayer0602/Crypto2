"""
HTF Crossover Strategy with Regime-Based Filters

Combines:
- Simple HTF crossover entry/exit logic
- Regime detection for adaptive parameters
- Minimum hold times based on market conditions
- ATR-based stops only in extreme regimes
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


def detect_market_regime(df: pd.DataFrame, lookback: int = 20) -> Dict[str, any]:
    """
    Detect market regime using price action and volatility.

    Returns regime with characteristics:
    - bull_fast, bull_slow, bear_fast, bear_slow, volatile, sideways
    """
    if len(df) < lookback:
        return {"type": "unknown", "min_hold_bars": 12}

    recent = df.tail(lookback)

    close = recent["close"].astype(float)
    returns = close.pct_change().dropna()

    # Calculate metrics
    avg_return = returns.mean()
    volatility = returns.std()
    trend_strength = abs(avg_return) / (volatility + 1e-9)

    # Thresholds
    STRONG_TREND = 0.5
    WEAK_TREND = 0.2
    HIGH_VOL = 0.03

    # Classify regime
    if trend_strength > STRONG_TREND:
        if avg_return > 0:
            regime_type = "bull_fast"
            min_hold = 6  # Exit faster in strong bull
        else:
            regime_type = "bear_fast"
            min_hold = 3  # Exit very fast in strong bear
    elif trend_strength > WEAK_TREND:
        if avg_return > 0:
            regime_type = "bull_slow"
            min_hold = 12  # Normal hold in slow bull
        else:
            regime_type = "bear_slow"
            min_hold = 6  # Moderate hold in slow bear
    elif volatility > HIGH_VOL:
        regime_type = "volatile"
        min_hold = 8  # Moderate hold in volatile
    else:
        regime_type = "sideways"
        min_hold = 15  # Hold longer in sideways

    return {
        "type": regime_type,
        "min_hold_bars": min_hold,
        "avg_return": avg_return,
        "volatility": volatility,
        "trend_strength": trend_strength,
    }


def check_htf_crossover_exit_filtered(position: Dict, df: pd.DataFrame,
                                       min_hold_bars: int = 12,
                                       use_regime_filter: bool = True) -> Optional[Dict]:
    """
    HTF Crossover exit with regime-based filters.

    Args:
        position: Position dict with direction, entry_price, stake, entry_time
        df: DataFrame with price data (must have 'close', 'htf_indicator', 'atr')
        min_hold_bars: Minimum bars to hold (can be overridden by regime)
        use_regime_filter: Whether to use regime-based adjustments

    Returns:
        Exit dict if exit signal found, None otherwise
    """
    if len(df) < 2:
        return None

    direction = position["direction"]
    entry_price = position["entry_price"]
    stake = position.get("stake", 1000)

    # Detect market regime
    if use_regime_filter:
        regime = detect_market_regime(df)
        effective_min_hold = regime["min_hold_bars"]
        regime_type = regime["type"]
    else:
        effective_min_hold = min_hold_bars
        regime_type = "unknown"

    # Check each bar for crossover
    for i in range(1, len(df)):
        bars_held = i

        # Enforce minimum hold based on regime
        if bars_held < effective_min_hold:
            continue

        curr = df.iloc[i]
        prev = df.iloc[i - 1]

        close_curr = float(curr["close"])
        close_prev = float(prev["close"])
        htf_curr = float(curr["htf_indicator"])
        htf_prev = float(prev["htf_indicator"])

        # Check for crossover
        crossover_detected = False

        if direction == "LONG":
            # Exit when close crosses BELOW HTF
            if close_prev >= htf_prev and close_curr < htf_curr:
                crossover_detected = True

        elif direction == "SHORT":
            # Exit when close crosses ABOVE HTF
            if close_prev <= htf_prev and close_curr > htf_curr:
                crossover_detected = True

        if crossover_detected:
            # Calculate PnL
            if direction == "LONG":
                pnl = (close_curr - entry_price) / entry_price * stake
            else:
                pnl = (entry_price - close_curr) / entry_price * stake

            return {
                "exit_price": close_curr,
                "bars_held": bars_held,
                "reason": f"HTF crossover exit (regime={regime_type})",
                "pnl": pnl,
                "regime": regime_type,
            }

        # Additional regime-based stop loss (only in bear markets)
        if use_regime_filter and regime_type in ["bear_fast", "bear_slow"]:
            atr = float(curr.get("atr", 0))

            # Use tighter stops in bear markets
            if regime_type == "bear_fast":
                stop_mult = 1.0  # Very tight
            else:
                stop_mult = 1.5  # Moderate

            if atr > 0:
                if direction == "LONG":
                    stop_price = entry_price - (atr * stop_mult)
                    if close_curr <= stop_price:
                        pnl = (close_curr - entry_price) / entry_price * stake
                        return {
                            "exit_price": close_curr,
                            "bars_held": bars_held,
                            "reason": f"ATR stop in {regime_type} (x{stop_mult:.1f})",
                            "pnl": pnl,
                            "regime": regime_type,
                        }
                else:
                    stop_price = entry_price + (atr * stop_mult)
                    if close_curr >= stop_price:
                        pnl = (entry_price - close_curr) / entry_price * stake
                        return {
                            "exit_price": close_curr,
                            "bars_held": bars_held,
                            "reason": f"ATR stop in {regime_type} (x{stop_mult:.1f})",
                            "pnl": pnl,
                            "regime": regime_type,
                        }

    return None


def check_htf_crossover_entry_filtered(df: pd.DataFrame, direction: str,
                                        use_regime_filter: bool = True) -> Tuple[bool, str]:
    """
    Check for HTF crossover entry with regime filter.

    Only enters if:
    - HTF crossover detected
    - Regime is favorable (not bear_fast for LONG, not bull_fast for SHORT)

    Args:
        df: DataFrame with price data
        direction: "LONG" or "SHORT"
        use_regime_filter: Whether to filter entries by regime

    Returns:
        (should_enter, reason)
    """
    if len(df) < 2:
        return False, "Not enough bars"

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    close_curr = float(curr["close"])
    close_prev = float(prev["close"])
    htf_curr = float(curr["htf_indicator"])
    htf_prev = float(prev["htf_indicator"])

    # Detect regime
    if use_regime_filter:
        regime = detect_market_regime(df)
        regime_type = regime["type"]

        # Filter out unfavorable regimes
        if direction == "LONG" and regime_type == "bear_fast":
            return False, f"Rejected: {regime_type} regime unfavorable for LONG"

        if direction == "SHORT" and regime_type == "bull_fast":
            return False, f"Rejected: {regime_type} regime unfavorable for SHORT"
    else:
        regime_type = "unknown"

    # Check crossover
    if direction == "LONG":
        # Buy when close crosses ABOVE HTF
        if close_prev < htf_prev and close_curr > htf_curr:
            return True, f"Close crossed HTF upward (regime={regime_type})"

    elif direction == "SHORT":
        # Sell when close crosses BELOW HTF
        if close_prev > htf_prev and close_curr < htf_curr:
            return True, f"Close crossed HTF downward (regime={regime_type})"

    return False, "No crossover"


if __name__ == "__main__":
    print("HTF Crossover Strategy with Regime Filters")
    print("=" * 50)
    print("\nThis module provides:")
    print("  - HTF crossover entry/exit logic")
    print("  - Market regime detection")
    print("  - Adaptive minimum hold times")
    print("  - Regime-based ATR stops")
    print("\nImport and use in your comparison scripts.")

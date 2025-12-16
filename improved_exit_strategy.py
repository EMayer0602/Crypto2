"""
Improved Exit Strategy - HTF Crossover Based with Market Regime Detection

This module implements a more sophisticated exit strategy that:
1. Detects market regime (Bull/Bear/Sideways/Volatile)
2. Uses HTF indicator crossover for exits
3. Adapts parameters based on market conditions
4. Implements trailing stops for profitable positions
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


def detect_market_regime(df: pd.DataFrame, lookback: int = 20) -> Dict[str, any]:
    """
    Detect current market regime based on HTF indicator and price action.

    Returns:
        dict with:
            - regime: 'bull_fast', 'bull_slow', 'bear_fast', 'bear_slow', 'sideways', 'volatile'
            - htf_trend: 'up', 'down', 'flat'
            - volatility: 'high', 'medium', 'low'
            - momentum: float (price momentum)
    """
    if len(df) < lookback + 1:
        return {
            "regime": "unknown",
            "htf_trend": "flat",
            "volatility": "medium",
            "momentum": 0.0
        }

    recent = df.iloc[-lookback:]
    curr = df.iloc[-1]

    # HTF Trend detection
    if "htf_indicator" in df.columns:
        htf_values = recent["htf_indicator"].dropna()
        if len(htf_values) >= 2:
            htf_start = htf_values.iloc[0]
            htf_end = htf_values.iloc[-1]
            htf_change_pct = (htf_end - htf_start) / htf_start * 100

            if htf_change_pct > 2.0:
                htf_trend = "up"
            elif htf_change_pct < -2.0:
                htf_trend = "down"
            else:
                htf_trend = "flat"
        else:
            htf_trend = "flat"
    else:
        htf_trend = "flat"

    # Price momentum
    close_values = recent["close"]
    price_start = close_values.iloc[0]
    price_end = close_values.iloc[-1]
    momentum = (price_end - price_start) / price_start * 100

    # Volatility detection (using ATR if available)
    if "atr" in df.columns and "close" in df.columns:
        atr = curr["atr"]
        price = curr["close"]
        atr_pct = (atr / price) * 100

        if atr_pct > 5.0:
            volatility = "high"
        elif atr_pct > 2.5:
            volatility = "medium"
        else:
            volatility = "low"
    else:
        volatility = "medium"

    # Determine regime
    if htf_trend == "up" and momentum > 3.0:
        regime = "bull_fast"
    elif htf_trend == "up" and momentum > 0:
        regime = "bull_slow"
    elif htf_trend == "down" and momentum < -3.0:
        regime = "bear_fast"
    elif htf_trend == "down" and momentum < 0:
        regime = "bear_slow"
    elif volatility == "high":
        regime = "volatile"
    else:
        regime = "sideways"

    return {
        "regime": regime,
        "htf_trend": htf_trend,
        "volatility": volatility,
        "momentum": momentum
    }


def calculate_htf_crossover_exit(
    position: Dict,
    df: pd.DataFrame,
    regime: Dict
) -> Tuple[Optional[float], Optional[str]]:
    """
    Calculate exit based on HTF indicator crossover.

    Args:
        position: Current position dict with direction, entry_price, etc.
        df: DataFrame with close, htf_indicator, atr columns
        regime: Market regime dict from detect_market_regime()

    Returns:
        (exit_price, reason) or (None, None) if no exit
    """
    if len(df) < 2:
        return None, None

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    direction = str(position.get("direction", "long")).lower()
    long_mode = direction == "long"

    close_curr = float(curr["close"])
    close_prev = float(prev["close"])

    # Get HTF indicator value
    if "htf_indicator" not in df.columns:
        return None, None

    htf_curr = float(curr["htf_indicator"]) if pd.notna(curr["htf_indicator"]) else None
    htf_prev = float(prev["htf_indicator"]) if pd.notna(prev["htf_indicator"]) else None

    if htf_curr is None or htf_prev is None:
        return None, None

    # Get ATR for buffer calculation
    atr = float(curr["atr"]) if "atr" in df.columns and pd.notna(curr["atr"]) else 0.0

    # Adaptive buffer based on market regime
    regime_type = regime["regime"]

    if regime_type in ["bull_fast", "bull_slow"]:
        # In bull market, give more room (wider buffer)
        buffer_multiplier = 1.5 if regime_type == "bull_fast" else 1.0
    elif regime_type in ["bear_fast", "bear_slow"]:
        # In bear market, tighter stops
        buffer_multiplier = 0.5 if regime_type == "bear_fast" else 0.75
    elif regime_type == "volatile":
        # In volatile market, wider buffer to avoid whipsaws
        buffer_multiplier = 2.0
    else:  # sideways
        buffer_multiplier = 1.0

    buffer = atr * buffer_multiplier

    # Check for crossover
    if long_mode:
        # Long exit: Close crosses below HTF - buffer
        exit_level = htf_curr - buffer
        prev_exit_level = htf_prev - buffer

        # Crossover happened?
        crossed = close_prev >= prev_exit_level and close_curr < exit_level

        if crossed:
            return close_curr, f"HTF crossover down (regime={regime_type})"

    else:  # short mode
        # Short exit: Close crosses above HTF + buffer
        exit_level = htf_curr + buffer
        prev_exit_level = htf_prev + buffer

        # Crossover happened?
        crossed = close_prev <= prev_exit_level and close_curr > exit_level

        if crossed:
            return close_curr, f"HTF crossover up (regime={regime_type})"

    return None, None


def calculate_trailing_stop(
    position: Dict,
    df: pd.DataFrame,
    regime: Dict
) -> Tuple[Optional[float], Optional[str]]:
    """
    Calculate trailing stop for profitable positions.
    Only activates when position is in profit.
    """
    if len(df) < 1:
        return None, None

    curr = df.iloc[-1]
    direction = str(position.get("direction", "long")).lower()
    long_mode = direction == "long"
    entry_price = float(position.get("entry_price", 0.0) or 0.0)

    close_curr = float(curr["close"])

    # Check if in profit
    if long_mode:
        in_profit = close_curr > entry_price
    else:
        in_profit = close_curr < entry_price

    if not in_profit:
        return None, None  # No trailing stop if not in profit

    # Get ATR
    atr = float(curr["atr"]) if "atr" in df.columns and pd.notna(curr["atr"]) else 0.0
    if atr == 0:
        return None, None

    # Trailing stop multiplier based on regime
    regime_type = regime["regime"]

    if regime_type in ["bull_fast", "bear_fast"]:
        # Fast markets: Wider trailing (let profits run)
        trail_mult = 2.0
    elif regime_type in ["bull_slow", "bear_slow"]:
        # Slow markets: Medium trailing
        trail_mult = 1.5
    else:  # sideways/volatile
        # Choppy markets: Tighter trailing (protect profits)
        trail_mult = 1.0

    # Calculate trailing stop level
    if long_mode:
        # Trail below current price
        trail_level = close_curr - (atr * trail_mult)

        # Check if hit
        if float(curr["low"]) <= trail_level:
            return trail_level, f"Trailing stop (trail={trail_mult:.1f}x ATR)"
    else:
        # Trail above current price
        trail_level = close_curr + (atr * trail_mult)

        # Check if hit
        if float(curr["high"]) >= trail_level:
            return trail_level, f"Trailing stop (trail={trail_mult:.1f}x ATR)"

    return None, None


def evaluate_exit_improved(
    position: Dict,
    df: pd.DataFrame,
    atr_mult: Optional[float],
    min_hold_bars: int
) -> Optional[Dict]:
    """
    Improved exit evaluation combining multiple strategies.

    Priority:
    1. Fixed ATR stop (disaster protection)
    2. Trailing stop (for profitable positions)
    3. HTF crossover (regime-adaptive)
    4. Trend flip (last resort after min_hold)

    Args:
        position: Position dict
        df: OHLC dataframe with indicators
        atr_mult: ATR multiplier for fixed stop
        min_hold_bars: Minimum bars to hold

    Returns:
        Exit dict with exit_price, reason, fees, pnl or None
    """
    if len(df) < 2:
        return None

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    direction = str(position.get("direction", "long")).lower()
    long_mode = direction == "long"
    entry_price = float(position.get("entry_price", 0.0) or 0.0)
    entry_atr = float(position.get("entry_atr", 0.0) or 0.0)
    stake = float(position.get("stake", 0.0) or 0.0)
    entry_time = position.get("entry_time")
    latest_ts = df.index[-1]

    # Calculate bars held
    if entry_time and isinstance(entry_time, pd.Timestamp):
        time_diff = latest_ts - entry_time
        bars_held = int(time_diff.total_seconds() / 300)  # 5-minute bars
    else:
        bars_held = 0

    # Detect market regime
    regime = detect_market_regime(df, lookback=20)

    exit_price = None
    reason = None

    # 1. DISASTER PROTECTION: Fixed ATR stop (always active)
    if atr_mult is not None and entry_atr > 0:
        # Make initial stop wider in bull markets, tighter in bear markets
        if regime["regime"] in ["bull_fast", "bull_slow"]:
            effective_mult = atr_mult * 1.5  # 50% wider
        elif regime["regime"] in ["bear_fast", "bear_slow"]:
            effective_mult = atr_mult * 0.75  # 25% tighter
        else:
            effective_mult = atr_mult

        stop_price = (entry_price - effective_mult * entry_atr) if long_mode else (entry_price + effective_mult * entry_atr)
        hit_stop = (long_mode and float(curr["low"]) <= stop_price) or ((not long_mode) and float(curr["high"]) >= stop_price)

        if hit_stop:
            exit_price = stop_price
            reason = f"ATR stop x{effective_mult:.2f} (regime={regime['regime']})"
            # Don't return yet, check if trailing stop is better

    # 2. TRAILING STOP: Protect profits (only for profitable positions)
    trail_price, trail_reason = calculate_trailing_stop(position, df, regime)
    if trail_price is not None:
        # If we have both ATR stop and trailing stop, choose the better one
        if exit_price is None:
            exit_price = trail_price
            reason = trail_reason
        else:
            # Choose the more favorable stop for the trader
            if long_mode:
                # For long, higher stop is better (less loss/more profit)
                if trail_price > exit_price:
                    exit_price = trail_price
                    reason = trail_reason
            else:
                # For short, lower stop is better
                if trail_price < exit_price:
                    exit_price = trail_price
                    reason = trail_reason

    # 3. HTF CROSSOVER: Regime-adaptive exit
    if exit_price is None and bars_held >= min_hold_bars:
        htf_price, htf_reason = calculate_htf_crossover_exit(position, df, regime)
        if htf_price is not None:
            exit_price = htf_price
            reason = htf_reason

    # 4. TREND FLIP: Last resort after minimum hold period
    if exit_price is None and bars_held >= min_hold_bars:
        trend_curr = int(curr["trend_flag"]) if "trend_flag" in curr else 0
        trend_prev = int(prev["trend_flag"]) if "trend_flag" in prev else 0

        flip_long = long_mode and trend_prev == 1 and trend_curr == -1
        flip_short = (not long_mode) and trend_prev == -1 and trend_curr == 1

        if flip_long or flip_short:
            # In strong trends, be more reluctant to exit on flip
            if regime["regime"] in ["bull_fast", "bear_fast"]:
                # Require 2 consecutive flips in fast markets
                if len(df) >= 3:
                    prev2 = df.iloc[-3]
                    trend_prev2 = int(prev2["trend_flag"]) if "trend_flag" in prev2 else 0
                    confirmed = (flip_long and trend_prev2 == 1) or (flip_short and trend_prev2 == -1)
                    if confirmed:
                        exit_price = float(curr["close"])
                        reason = f"Confirmed trend flip (regime={regime['regime']})"
            else:
                # Normal trend flip exit
                exit_price = float(curr["close"])
                reason = f"Trend flip (regime={regime['regime']})"

    # No exit signal
    if exit_price is None:
        return None

    # Calculate PnL
    import Supertrend_5Min as st
    gross_pnl = (exit_price - entry_price) / entry_price * stake if long_mode else (entry_price - exit_price) / entry_price * stake
    fees = stake * st.FEE_RATE * 2.0
    pnl = gross_pnl - fees

    return {
        "exit_price": exit_price,
        "reason": reason,
        "fees": fees,
        "pnl": pnl,
        "regime": regime["regime"],  # Store regime for analysis
        "bars_held": bars_held,
    }

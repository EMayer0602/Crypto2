# Time-Based Exit Strategy

## Overview

This feature implements data-driven exit timing based on peak profit analysis. Analysis of historical trades showed:

- **Overall**: Peak profit occurs at ~65% of trade duration on average
- **Long trades**: Left 2,920 USDT on table (22,733% giveback rate) - need TIGHT exits
- **Short trades**: Left 6,491 USDT on table (486% giveback rate) - also need early exits
- **Total profit left**: 9,411 USDT (698% of actual profit)

## How It Works

1. **Tracks trade duration** in bars since entry
2. **Compares to optimal hold time** configured per symbol/direction
3. **Exits when**:
   - Trade has been held for >= optimal_hold_bars
   - Current unrealized PnL >= -1% (at or near breakeven)

This prevents giving back profits while avoiding premature exits on losing trades.

## Configuration

### Enable/Disable

In `paper_trader.py` or `config_local.py`:

```python
USE_TIME_BASED_EXIT = True  # Enable time-based exits
```

### Customize Hold Times

Edit `optimal_hold_times_defaults.py`:

```python
OPTIMAL_HOLD_BARS = {
    ("BTC/EUR", "long"): 10,   # Exit BTC longs after 10 bars
    ("BTC/EUR", "short"): 15,  # Exit BTC shorts after 15 bars
    ("ETH/EUR", "long"): 12,
    # ... add more symbols
}
```

**Default fallbacks**:
- Long trades: 12 bars
- Short trades: 15 bars

### Generate Optimal Times from Data

Run the analysis tool (when network access available):

```batch
python find_optimal_hold_times_standalone.py paper_trading_simulation_log.csv
```

This will:
1. Analyze each closed trade
2. Find when profit peaked
3. Calculate median bars to peak per symbol/direction
4. Generate `optimal_hold_times.py` with recommended values

## Priority Order

Exits are checked in this order:

1. **ATR Stop Loss** - Immediate exit if stop hit
2. **Time-Based Exit** - Exit after optimal hold time (if enabled and profitable)
3. **Trend Flip** - Exit on trend reversal (respecting min_hold filter)

## Example Exit Reasons

In trade logs you'll see:

- `"Time-based exit (12 bars, optimal=12)"` - Exited at optimal time
- `"Time-based exit (15 bars, optimal=12)"` - Held longer than optimal but still profitable
- `"ATR stop x1.50"` - Stop loss triggered first
- `"Trend flip"` - Trend reversed before reaching optimal time

## Customization Tips

### Conservative (Exit Earlier)
- Reduce optimal_hold_bars values
- Use lower bound of analysis (e.g., 25th percentile instead of median)

### Aggressive (Hold Longer)
- Increase optimal_hold_bars values
- Use upper bound of analysis (e.g., 75th percentile)

### Symbol-Specific
Different symbols have different optimal times. Analyze separately:
- High volatility coins: Shorter holds (8-10 bars)
- Stable coins: Longer holds (15-20 bars)
- EUR pairs vs USDT pairs: May behave differently

## Monitoring

Check `paper_trading_simulation_log.csv` for exit reasons:

```batch
grep "Time-based exit" paper_trading_simulation_log.csv | wc -l
```

Compare PnL of time-based exits vs other exit types to validate effectiveness.

## Next Steps

1. **Run simulation** with time-based exits enabled
2. **Analyze results** - Did we capture more profit?
3. **Fine-tune hold times** per symbol based on results
4. **Iterate** - Rerun analysis with updated times

## Notes

- Time-based exits complement (don't replace) ATR stops and trend flips
- Works best with symbols that have consistent peak timing patterns
- May need adjustment as market conditions change
- Consider seasonal/market cycle effects when setting long-term holds

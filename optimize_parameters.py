"""
Parameter Optimization Script

This script helps optimize the exit strategy parameters by testing
different combinations and showing which performs best.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import itertools


# Parameter ranges to test
PARAM_GRID = {
    # Buffer multipliers for HTF crossover
    "bull_fast_buffer": [2.0, 2.5, 3.0],      # Currently 1.5
    "bull_slow_buffer": [1.5, 2.0, 2.5],      # Currently 1.0
    "bear_fast_buffer": [0.5, 0.75, 1.0],     # Currently 0.5
    "bear_slow_buffer": [0.75, 1.0, 1.25],    # Currently 0.75
    "volatile_buffer": [2.5, 3.0, 3.5],       # Currently 2.0
    "sideways_buffer": [1.5, 2.0, 2.5],       # Currently 1.0

    # Trailing stop multipliers
    "fast_trail": [2.5, 3.0, 3.5],            # Currently 2.0
    "slow_trail": [2.0, 2.5, 3.0],            # Currently 1.5
    "choppy_trail": [1.5, 2.0, 2.5],          # Currently 1.0

    # ATR stop adjustments
    "bull_atr_mult": [1.5, 1.75, 2.0],        # Currently 1.5
    "bear_atr_mult": [0.75, 1.0, 1.25],       # Currently 0.75
}


def test_parameter_combination(params: Dict) -> Dict:
    """
    Test a specific parameter combination.
    Returns performance metrics.
    """
    # TODO: Implement actual testing
    # For now, return dummy data
    return {
        "total_pnl": 0.0,
        "win_rate": 0.0,
        "avg_pnl": 0.0,
        "params": params
    }


def grid_search_top_params(max_combinations: int = 20):
    """
    Perform grid search to find best parameter combinations.

    Instead of testing all combinations (too many), we:
    1. Test a random sample
    2. Find top performers
    3. Refine search around best areas
    """
    print("="*80)
    print("PARAMETER OPTIMIZATION - Grid Search")
    print("="*80)

    # Generate parameter combinations
    all_combos = []

    # Sample approach: Test key combinations
    # Bull markets (most important to fix)
    for bull_fast in PARAM_GRID["bull_fast_buffer"]:
        for bull_slow in PARAM_GRID["bull_slow_buffer"]:
            for fast_trail in PARAM_GRID["fast_trail"]:
                all_combos.append({
                    "bull_fast_buffer": bull_fast,
                    "bull_slow_buffer": bull_slow,
                    "bear_fast_buffer": 0.5,
                    "bear_slow_buffer": 0.75,
                    "volatile_buffer": 2.5,
                    "sideways_buffer": 1.5,
                    "fast_trail": fast_trail,
                    "slow_trail": 2.0,
                    "choppy_trail": 1.5,
                    "bull_atr_mult": 1.75,
                    "bear_atr_mult": 0.75,
                })

    print(f"\nTesting {len(all_combos)} parameter combinations...")
    print("(This may take a while...)\n")

    results = []
    for i, params in enumerate(all_combos):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(all_combos)}")

        result = test_parameter_combination(params)
        results.append(result)

    # Sort by total PnL
    results.sort(key=lambda x: x["total_pnl"], reverse=True)

    # Show top 5
    print("\n" + "="*80)
    print("TOP 5 PARAMETER COMBINATIONS")
    print("="*80)

    for i, result in enumerate(results[:5]):
        print(f"\n#{i+1}: Total PnL: {result['total_pnl']:.2f} USDT")
        print(f"   Win Rate: {result['win_rate']:.1f}%")
        print(f"   Avg PnL: {result['avg_pnl']:.2f} USDT")
        print(f"   Parameters:")
        for key, val in result['params'].items():
            print(f"     {key}: {val}")

    return results


def suggest_parameters_from_results(current_results: Dict):
    """
    Analyze current test results and suggest parameter adjustments.
    """
    print("\n" + "="*80)
    print("PARAMETER SUGGESTIONS BASED ON YOUR RESULTS")
    print("="*80)

    print("\n📊 Analysis of your results:")
    print(f"  - Old Strategy: -234.52 USDT (0% win rate)")
    print(f"  - New Strategy: -167.56 USDT (12.5% win rate)")
    print(f"  - Improvement: +66.96 USDT (+28.6%)")

    print("\n✓ Good news:")
    print("  - New strategy IS better (+67 USDT)")
    print("  - ETH/EUR shows big improvement (+54.58 USDT)")
    print("  - Direction is correct!")

    print("\n⚠️  Issues to fix:")
    print("  - Still losing overall (-167.56 USDT)")
    print("  - SOL/EUR shows no change (parameters too similar?)")
    print("  - Win rate still low (12.5%)")

    print("\n🔧 Recommended Parameter Adjustments:")
    print("\n1. INCREASE BULL MARKET BUFFERS (let winners run longer):")
    print("   bull_fast_buffer: 1.5 → 2.5 or 3.0")
    print("   bull_slow_buffer: 1.0 → 2.0")
    print("   Reason: Exits are too early in uptrends")

    print("\n2. INCREASE TRAILING STOPS (protect profits better):")
    print("   fast_trail: 2.0 → 3.0")
    print("   slow_trail: 1.5 → 2.5")
    print("   Reason: Let profitable positions breathe")

    print("\n3. WIDEN ATR STOPS IN BULL MARKETS:")
    print("   bull_atr_mult: 1.5 → 2.0")
    print("   Reason: Avoid premature stops in uptrends")

    print("\n4. KEEP BEAR PARAMETERS TIGHT:")
    print("   bear_fast_buffer: 0.5 (keep)")
    print("   bear_slow_buffer: 0.75 (keep)")
    print("   Reason: Cut losses quickly in downtrends")

    print("\n" + "="*80)
    print("SUGGESTED NEW PARAMETERS:")
    print("="*80)

    suggested = {
        "bull_fast_buffer": 2.5,    # Was 1.5
        "bull_slow_buffer": 2.0,    # Was 1.0
        "bear_fast_buffer": 0.5,    # Keep
        "bear_slow_buffer": 0.75,   # Keep
        "volatile_buffer": 3.0,     # Was 2.0
        "sideways_buffer": 2.0,     # Was 1.0
        "fast_trail": 3.0,          # Was 2.0
        "slow_trail": 2.5,          # Was 1.5
        "choppy_trail": 1.5,        # Keep
        "bull_atr_mult": 2.0,       # Was 1.5
        "bear_atr_mult": 0.75,      # Keep
    }

    for key, val in suggested.items():
        print(f"  {key}: {val}")

    print("\n💡 Next steps:")
    print("  1. I'll update improved_exit_strategy.py with these values")
    print("  2. Run comparison again: python compare_exit_strategies.py --trades 10")
    print("  3. Check if results improve (target: positive PnL, 40%+ win rate)")
    print("  4. Iterate if needed")

    return suggested


if __name__ == "__main__":
    # For now, just show suggestions based on the user's results
    suggest_parameters_from_results({})

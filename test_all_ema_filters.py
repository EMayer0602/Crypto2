"""
Automated testing of all EMA filter variants.

Tests 4 configurations:
- BASELINE: EMA-20 Slope ≥ 0%
- OPTION 1: + Price > EMA-20
- OPTION 2: Slope ≥ 0.1% (stricter)
- OPTION 3: + EMA-50 confirmation

Runs 4-week simulation for each and compares results.
"""

import os
import shutil
import subprocess
import pandas as pd
from datetime import datetime

# Test configurations
VARIANTS = {
    "BASELINE": {
        "file": "optimal_ema_slope_BASELINE.py",
        "description": "EMA-20 Slope ≥ 0% (Current Best)"
    },
    "OPTION1": {
        "file": "optimal_ema_slope_OPTION1.py",
        "description": "EMA-20 Slope ≥ 0% + Price > EMA-20"
    },
    "OPTION2": {
        "file": "optimal_ema_slope_OPTION2.py",
        "description": "EMA-20 Slope ≥ 0.1% (Stricter)"
    },
    "OPTION3": {
        "file": "optimal_ema_slope_OPTION3.py",
        "description": "EMA-20 + EMA-50 Dual Confirmation"
    }
}

# Simulation parameters
START_DATE = "2024-11-23"
STAKE = 2000

# Results storage
RESULTS_DIR = "ema_filter_test_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_simulation(variant_name, config_file):
    """Run simulation with specific config file."""
    print(f"\n{'='*80}")
    print(f"Testing {variant_name}: {VARIANTS[variant_name]['description']}")
    print(f"{'='*80}\n")

    # Backup current config
    if os.path.exists("optimal_ema_slope_defaults.py"):
        shutil.copy("optimal_ema_slope_defaults.py", "optimal_ema_slope_defaults.py.backup")

    # Copy test config to active config
    shutil.copy(config_file, "optimal_ema_slope_defaults.py")

    # Run simulation
    cmd = [
        "python", "paper_trader.py",
        "--simulate",
        "--start", START_DATE,
        "--stake", str(STAKE)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        # Save logs
        log_files = [
            "paper_trading_simulation_log.csv",
            "paper_trading_actual_trades.csv",
            "paper_trading_simulation_log.json"
        ]

        for log_file in log_files:
            if os.path.exists(log_file):
                dest = os.path.join(RESULTS_DIR, f"{variant_name}_{log_file}")
                shutil.copy(log_file, dest)

        return True

    except subprocess.TimeoutExpired:
        print(f"ERROR: {variant_name} simulation timed out!")
        return False
    except Exception as e:
        print(f"ERROR running {variant_name}: {e}")
        return False
    finally:
        # Restore original config
        if os.path.exists("optimal_ema_slope_defaults.py.backup"):
            shutil.copy("optimal_ema_slope_defaults.py.backup", "optimal_ema_slope_defaults.py")
            os.remove("optimal_ema_slope_defaults.py.backup")


def analyze_results():
    """Analyze and compare all variant results."""
    print(f"\n{'='*80}")
    print("COMPARATIVE RESULTS ANALYSIS")
    print(f"{'='*80}\n")

    results = []

    for variant_name in VARIANTS.keys():
        csv_file = os.path.join(RESULTS_DIR, f"{variant_name}_paper_trading_simulation_log.csv")

        if not os.path.exists(csv_file):
            print(f"WARNING: No results for {variant_name}")
            continue

        df = pd.read_csv(csv_file)

        # Calculate metrics
        total_trades = len(df)
        long_trades = df[df['direction'].str.lower() == 'long']
        short_trades = df[df['direction'].str.lower() == 'short']

        total_pnl = df['pnl'].sum()
        long_pnl = long_trades['pnl'].sum() if not long_trades.empty else 0
        short_pnl = short_trades['pnl'].sum() if not short_trades.empty else 0

        long_wins = len(long_trades[long_trades['pnl'] > 0]) if not long_trades.empty else 0
        long_total = len(long_trades) if not long_trades.empty else 0
        long_win_rate = (long_wins / long_total * 100) if long_total > 0 else 0

        avg_long_pnl = long_pnl / long_total if long_total > 0 else 0

        results.append({
            'Variant': variant_name,
            'Description': VARIANTS[variant_name]['description'],
            'Total Trades': total_trades,
            'Long Trades': long_total,
            'Total PnL': total_pnl,
            'Long PnL': long_pnl,
            'Short PnL': short_pnl,
            'Long Win Rate': long_win_rate,
            'Avg Long PnL': avg_long_pnl
        })

    # Create comparison DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Long PnL', ascending=False)

    # Display results
    print(results_df.to_string(index=False))

    # Save to CSV
    results_file = os.path.join(RESULTS_DIR, "comparison_summary.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\n✅ Results saved to: {results_file}")

    # Find best variant
    best = results_df.iloc[0]
    print(f"\n🏆 BEST PERFORMING: {best['Variant']}")
    print(f"   {best['Description']}")
    print(f"   Long PnL: {best['Long PnL']:.2f} USDT")
    print(f"   Long Win Rate: {best['Long Win Rate']:.2f}%")
    print(f"   Total PnL: {best['Total PnL']:.2f} USDT")

    return results_df


def main():
    print("="*80)
    print("EMA FILTER VARIANT TESTING")
    print("="*80)
    print(f"Start Date: {START_DATE}")
    print(f"Stake: {STAKE} USDT")
    print(f"Variants to test: {len(VARIANTS)}")
    print("="*80)

    # Run all simulations
    for variant_name, config in VARIANTS.items():
        success = run_simulation(variant_name, config['file'])
        if not success:
            print(f"⚠️  {variant_name} failed, continuing with next...")

    # Analyze results
    analyze_results()

    print(f"\n{'='*80}")
    print("✅ ALL TESTS COMPLETE!")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

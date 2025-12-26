#!/usr/bin/env python3
"""Parameter testing script for ATR multiplier and HTF optimization."""

import os
import shutil
import subprocess
import re
from datetime import datetime, timedelta

CSV_PATH = os.path.join("report_html", "best_params_overall.csv")
BACKUP_PATH = os.path.join("report_html", "best_params_overall_backup.csv")

# Global simulation period (set via --months)
SIM_START_DATE = None  # None = use default (2025-01-01)


def backup_csv():
    shutil.copy(CSV_PATH, BACKUP_PATH)
    print(f"[Backup] Saved to {BACKUP_PATH}")


def restore_csv():
    if os.path.exists(BACKUP_PATH):
        shutil.copy(BACKUP_PATH, CSV_PATH)
        print(f"[Restore] Restored from {BACKUP_PATH}")


def modify_atr(atr_value: float):
    """Set all ATRStopMult and ATRStopMultValue to the same value."""
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = [lines[0]]  # Keep header
    atr_str = str(atr_value).replace(".", ",")

    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.strip().split(";")
        if len(parts) >= 10:
            parts[8] = atr_str   # ATRStopMult
            parts[9] = atr_str   # ATRStopMultValue
        new_lines.append(";".join(parts) + "\n")

    with open(CSV_PATH, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"[Modify] ATR set to {atr_value}")


def modify_htf(htf_value: str):
    """Set all HTF to the same value."""
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = [lines[0]]  # Keep header

    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.strip().split(";")
        if len(parts) >= 13:
            parts[12] = htf_value  # HTF column
        new_lines.append(";".join(parts) + "\n")

    with open(CSV_PATH, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"[Modify] HTF set to {htf_value}")


def run_simulation():
    """Run simulation and extract key metrics from JSON output."""
    import json

    cmd = ["python", "paper_trader.py", "--simulate", "--clear-outputs", "--reset-state"]
    if SIM_START_DATE:
        cmd.extend(["--start", SIM_START_DATE])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    # Read results from JSON file
    json_path = "paper_trading_simulation_summary.json"
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        pnl = float(data.get("closed_pnl", 0))
        trades = int(data.get("closed_trades", 0))
        winrate = float(data.get("win_rate", 0))

        return {"pnl": pnl, "trades": trades, "winrate": winrate}
    except Exception as e:
        print(f"[Error] Could not read results: {e}")
        return {"pnl": 0, "trades": 0, "winrate": 0}


def test_atr_values():
    """Test different ATR multiplier values."""
    print("\n" + "="*60)
    print("ATR MULTIPLIER TEST")
    print("="*60)

    backup_csv()
    results = []

    for atr in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        restore_csv()
        modify_atr(atr)
        print(f"\n[Test] Running with ATR = {atr}...")
        metrics = run_simulation()
        results.append({"atr": atr, **metrics})
        print(f"[Result] ATR={atr}: PnL={metrics['pnl']:.2f}, Trades={metrics['trades']}, WinRate={metrics['winrate']:.1f}%")

    restore_csv()

    print("\n" + "-"*60)
    print("ATR TEST SUMMARY")
    print("-"*60)
    print(f"{'ATR':<8} {'PnL':>12} {'Trades':>8} {'WinRate':>10}")
    for r in results:
        print(f"{r['atr']:<8} {r['pnl']:>12.2f} {r['trades']:>8} {r['winrate']:>9.1f}%")

    best = max(results, key=lambda x: x['pnl'])
    print(f"\nBest ATR: {best['atr']} with PnL: {best['pnl']:.2f}")
    return results


def test_htf_values():
    """Test different HTF values."""
    print("\n" + "="*60)
    print("HTF (Higher Timeframe) TEST")
    print("="*60)

    backup_csv()
    results = []

    for htf in ["1h", "4h", "6h", "8h", "12h"]:
        restore_csv()
        modify_htf(htf)
        print(f"\n[Test] Running with HTF = {htf}...")
        metrics = run_simulation()
        results.append({"htf": htf, **metrics})
        print(f"[Result] HTF={htf}: PnL={metrics['pnl']:.2f}, Trades={metrics['trades']}, WinRate={metrics['winrate']:.1f}%")

    restore_csv()

    print("\n" + "-"*60)
    print("HTF TEST SUMMARY")
    print("-"*60)
    print(f"{'HTF':<8} {'PnL':>12} {'Trades':>8} {'WinRate':>10}")
    for r in results:
        print(f"{r['htf']:<8} {r['pnl']:>12.2f} {r['trades']:>8} {r['winrate']:>9.1f}%")

    best = max(results, key=lambda x: x['pnl'])
    print(f"\nBest HTF: {best['htf']} with PnL: {best['pnl']:.2f}")
    return results


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Parameter testing for ATR and HTF optimization")
    parser.add_argument("test_type", choices=["atr", "htf", "both"], help="Test type to run")
    parser.add_argument("--months", type=int, default=None, help="Simulate last N months (default: full year from Jan 1)")
    args = parser.parse_args()

    # Set start date based on --months
    sim_start = None
    if args.months:
        start_date = datetime.now() - timedelta(days=args.months * 30)
        sim_start = start_date.strftime("%Y-%m-%d")
        print(f"[Config] Simulation period: last {args.months} months (from {sim_start})")
    else:
        print("[Config] Simulation period: full year (from 2025-01-01)")

    # Update module-level variable
    import param_test
    param_test.SIM_START_DATE = sim_start

    if args.test_type == "atr":
        test_atr_values()
    elif args.test_type == "htf":
        test_htf_values()
    elif args.test_type == "both":
        test_atr_values()
        test_htf_values()

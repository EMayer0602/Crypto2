#!/usr/bin/env python3
"""Parameter testing script for ATR multiplier and HTF optimization."""

import os
import shutil
import subprocess
import re

CSV_PATH = os.path.join("report_html", "best_params_overall.csv")
BACKUP_PATH = os.path.join("report_html", "best_params_overall_backup.csv")


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
    """Run simulation and extract key metrics."""
    result = subprocess.run(
        ["python", "paper_trader.py", "--simulate", "--clear-outputs", "--reset-state"],
        capture_output=True,
        text=True
    )
    output = result.stdout + result.stderr

    # Extract metrics from output
    pnl_match = re.search(r"Closed PnL.*?(-?[\d,.]+)", output)
    trades_match = re.search(r"Closed trades.*?(\d+)", output)
    winrate_match = re.search(r"Win rate.*?([\d.]+)", output)

    pnl = float(pnl_match.group(1).replace(",", "")) if pnl_match else 0
    trades = int(trades_match.group(1)) if trades_match else 0
    winrate = float(winrate_match.group(1)) if winrate_match else 0

    return {"pnl": pnl, "trades": trades, "winrate": winrate}


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

    if len(sys.argv) < 2:
        print("Usage: python param_test.py [atr|htf|both]")
        sys.exit(1)

    test_type = sys.argv[1].lower()

    if test_type == "atr":
        test_atr_values()
    elif test_type == "htf":
        test_htf_values()
    elif test_type == "both":
        test_atr_values()
        test_htf_values()
    else:
        print(f"Unknown test type: {test_type}")
        print("Use: atr, htf, or both")

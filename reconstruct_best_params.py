#!/usr/bin/env python3
"""
Reconstruct best_params_overall.csv from individual best_params_*.csv files.
"""

import os
import glob
import pandas as pd
from pathlib import Path

REPORT_DIR = "report_html"
OUTPUT_FILE = os.path.join(REPORT_DIR, "best_params_overall.csv")

def find_best_params_files():
    """Find all best_params CSV files in subdirectories."""
    patterns = [
        os.path.join(REPORT_DIR, "*", "best_params*.csv"),
        os.path.join(REPORT_DIR, "best_params_*.csv"),
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))

    # Exclude the overall file itself
    files = [f for f in files if "overall" not in f.lower()]

    return files

def main():
    print("Reconstructing best_params_overall.csv...")

    files = find_best_params_files()
    print(f"Found {len(files)} parameter files")

    if not files:
        print("No files found! Check report_html directory.")
        return

    all_dfs = []

    for f in files:
        try:
            # Try different separators
            try:
                df = pd.read_csv(f, sep=";", decimal=",")
            except:
                df = pd.read_csv(f)

            if not df.empty:
                print(f"  + {f}: {len(df)} rows")
                all_dfs.append(df)
        except Exception as e:
            print(f"  ! Error reading {f}: {e}")

    if not all_dfs:
        print("No data loaded!")
        return

    # Combine all dataframes
    combined = pd.concat(all_dfs, ignore_index=True)

    # Remove duplicates based on Symbol + Direction + Indicator
    if "Symbol" in combined.columns and "Direction" in combined.columns:
        combined = combined.drop_duplicates(subset=["Symbol", "Direction", "Indicator"], keep="first")

    # Save
    combined.to_csv(OUTPUT_FILE, sep=";", decimal=",", index=False)

    # Count Long/Short
    if "Direction" in combined.columns:
        longs = len(combined[combined["Direction"] == "Long"])
        shorts = len(combined[combined["Direction"] == "Short"])
        print(f"\nResult: {len(combined)} total ({longs} Long, {shorts} Short)")
    else:
        print(f"\nResult: {len(combined)} total rows")

    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

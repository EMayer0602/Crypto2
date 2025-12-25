#!/usr/bin/env python3
"""Reconstruct best_params_overall.csv from individual files."""

import os
import glob
import pandas as pd

REPORT_DIR = "report_html"
OUTPUT_FILE = os.path.join(REPORT_DIR, "best_params_overall.csv")

def main():
    print("Reconstructing best_params_overall.csv...")

    patterns = [
        os.path.join(REPORT_DIR, "*", "best_params*.csv"),
        os.path.join(REPORT_DIR, "best_params_*.csv"),
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    files = [f for f in files if "overall" not in f.lower()]

    print(f"Found {len(files)} parameter files")
    if not files:
        print("No files found!")
        return

    all_dfs = []
    for f in files:
        try:
            try:
                df = pd.read_csv(f, sep=";", decimal=",")
            except:
                df = pd.read_csv(f)
            if not df.empty:
                print(f"  + {f}: {len(df)} rows")
                all_dfs.append(df)
        except Exception as e:
            print(f"  ! Error: {f}: {e}")

    if not all_dfs:
        print("No data loaded!")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    if "Symbol" in combined.columns and "Direction" in combined.columns:
        combined = combined.drop_duplicates(subset=["Symbol", "Direction", "Indicator"], keep="first")

    combined.to_csv(OUTPUT_FILE, sep=";", decimal=",", index=False)

    if "Direction" in combined.columns:
        longs = len(combined[combined["Direction"] == "Long"])
        shorts = len(combined[combined["Direction"] == "Short"])
        print(f"\nResult: {len(combined)} total ({longs} Long, {shorts} Short)")
    else:
        print(f"\nResult: {len(combined)} rows")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

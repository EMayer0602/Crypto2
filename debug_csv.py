"""Debug script to check CSV formatting and data types"""
import pandas as pd
import os

files_to_check = [
    "paper_trading_log.csv",
    "report_html/initial_trades.csv",
    "paper_trading_simulation_log.csv",
]

for file_path in files_to_check:
    if not os.path.exists(file_path):
        print(f"\n{'='*60}")
        print(f"[X] FILE NOT FOUND: {file_path}")
        continue

    print(f"\n{'='*60}")
    print(f"FILE: {file_path}")
    print(f"{'='*60}")

    # Read CSV
    df = pd.read_csv(file_path, quotechar='"')

    print(f"\nRows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    if df.empty:
        print("WARNING: DataFrame is EMPTY")
        continue

    # Check data types
    print(f"\nData types:")
    for col in df.columns:
        if col in ["EntryPrice", "ExitPrice", "Stake", "PnL", "entry_price", "exit_price", "stake", "pnl"]:
            print(f"  {col}: {df[col].dtype}")

    # Show first row raw values
    print(f"\nFirst row (raw values):")
    first_row = df.iloc[0]
    for col in df.columns:
        if col in ["Symbol", "EntryPrice", "ExitPrice", "Stake", "PnL", "symbol", "entry_price", "exit_price", "stake", "pnl"]:
            val = first_row[col]
            print(f"  {col}: {repr(val)} (type: {type(val).__name__})")

    # Show first row after to_numeric conversion
    print(f"\nFirst row (after pd.to_numeric):")
    for col in ["EntryPrice", "ExitPrice", "Stake", "PnL"]:
        if col in df.columns:
            numeric_val = pd.to_numeric(df[col].iloc[0], errors="coerce")
            print(f"  {col}: {numeric_val} → formatted: {numeric_val:.8f if pd.notna(numeric_val) else 'NaN'}")

    # Show formatted string
    print(f"\nFormatted strings (how they appear in HTML):")
    for col in ["EntryPrice", "ExitPrice", "Stake", "PnL"]:
        if col in df.columns:
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            formatted_series = numeric_series.apply(lambda x: f"{x:.8f}" if pd.notna(x) else "")
            print(f"  {col}: {formatted_series.iloc[0]}")

print(f"\n{'='*60}")
print("[OK] Debug complete")

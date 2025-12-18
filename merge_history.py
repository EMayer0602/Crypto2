"""Merge initial_trades.csv into paper_trading_log.csv to build full history"""
import pandas as pd
import os

log_file = "paper_trading_log.csv"
initial_file = "report_html/initial_trades.csv"

# Load existing log (if any)
if os.path.exists(log_file):
    log_df = pd.read_csv(log_file, quotechar='"')
    print(f"Existing log: {len(log_df)} trades")
else:
    log_df = pd.DataFrame()
    print("No existing log")

# Load initial trades from last simulation
if os.path.exists(initial_file):
    initial_df = pd.read_csv(initial_file)
    print(f"Initial trades: {len(initial_df)} trades")
    
    # Merge - keep all unique trades
    if not log_df.empty and not initial_df.empty:
        # Combine and deduplicate
        combined = pd.concat([initial_df, log_df], ignore_index=True)

        # Remove duplicates based on key fields (use lowercase column names)
        # Try lowercase first, fallback to Uppercase for legacy CSVs
        dedup_cols = []
        if "symbol" in combined.columns:
            dedup_cols = ["symbol", "entry_time", "exit_time", "exit_price"]
        elif "Symbol" in combined.columns:
            dedup_cols = ["Symbol", "EntryTime", "ExitTime", "ExitPrice"]

        if dedup_cols and all(col in combined.columns for col in dedup_cols):
            combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
            print(f"Combined: {len(combined)} unique trades")
        else:
            print(f"[Warning] Could not deduplicate - missing columns. Combined: {len(combined)} trades (may have duplicates)")

        # Sort by exit time (try lowercase first, then Uppercase)
        sort_col = "exit_time" if "exit_time" in combined.columns else ("ExitTime" if "ExitTime" in combined.columns else None)
        if sort_col:
            combined[f"{sort_col}_sort"] = pd.to_datetime(combined[sort_col], errors="coerce")
            combined = combined.sort_values(f"{sort_col}_sort").drop(columns=[f"{sort_col}_sort"])
        
        # Write back with proper CSV format (quotes for ParamDesc/Reason)
        combined.to_csv(log_file, index=False, quoting=1)  # QUOTE_ALL
        print(f"✓ Wrote {len(combined)} trades to {log_file}")
    elif not initial_df.empty:
        initial_df.to_csv(log_file, index=False, quoting=1)
        print(f"✓ Created new log with {len(initial_df)} trades")
else:
    print(f"No initial trades file found at {initial_file}")

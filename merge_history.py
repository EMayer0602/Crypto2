"""Copy initial_trades.csv to paper_trading_log.csv (overwrite, don't merge)"""
import pandas as pd
import os
import shutil

log_file = "paper_trading_log.csv"
initial_file = "report_html/initial_trades.csv"

# Load initial trades from last simulation
if os.path.exists(initial_file):
    # Simply copy/overwrite - the --simulate already contains the full history we want
    initial_df = pd.read_csv(initial_file)
    print(f"Initial trades: {len(initial_df)} trades")

    # Write to log file without merging (overwrite)
    # Use quoting=1 (QUOTE_ALL) to handle commas in ParamDesc/Reason
    initial_df.to_csv(log_file, index=False, quoting=1)
    print(f"✓ Overwrote {log_file} with {len(initial_df)} trades")
else:
    print(f"No initial trades file found at {initial_file}")

# Supertrend_5Min Helper Guide

## Overview
Quick reference to reset state, run simulations, place forced testnet orders, and inspect Binance testnet order history for the Supertrend/overall-best strategies.

## Prerequisites
- Python 3.10+ on PATH
- Install deps: `pip install -r requirements.txt`
- `.env` with testnet keys:
  - `BINANCE_API_KEY_TEST`
  - `BINANCE_API_SECRET_TEST`

## Common Tasks

### 1) Full cleanup (state + outputs)
PowerShell:
```
cd "C:\Users\Edgar.000\Documents\____Trading strategies\Supertrend_5Min"
python paper_trader.py --clear-all
```

### 2) Run a 48h paper simulation
```
cd "C:\Users\Edgar.000\Documents\____Trading strategies\Supertrend_5Min"
$start=(Get-Date).AddHours(-48).ToString('s')
$end=(Get-Date).ToString('s')
$syms="BTC/EUR,ETH/EUR,XRP/EUR,LINK/EUR,LUNC/USDT,SOL/EUR,SUI/EUR,TNSR/USDC,ZEC/USDC"
python paper_trader.py --simulate --start $start --end $end --symbols $syms --summary-html report_html/last48_summary.html --summary-json report_html/last48_summary.json --sim-log report_html/last48_trades.csv --sim-json report_html/last48_trades.json --open-log report_html/last48_open.csv --open-json report_html/last48_open.json
```
Outputs: `report_html/last48_*` plus summary files.

### 3) Force fresh testnet buys (small stake to avoid balance errors)
Testnet has no EUR balance; USDT/USDC pairs will place, EUR pairs will fail unless funded. Use a small stake like 50:
```
cd "C:\Users\Edgar.000\Documents\____Trading strategies\Supertrend_5Min"
powershell -NoLogo -Command "$ErrorActionPreference='Stop'; $symbols = @('BTC/EUR','ETH/EUR','XRP/EUR','LINK/EUR','LUNC/USDT','SOL/EUR','SUI/EUR','TNSR/USDC','ZEC/USDC'); foreach ($s in $symbols) { Write-Host \"=== Forcing $s ===\"; python paper_trader.py --testnet --stake 50 --place-orders --force-entry \"$s:long\" --force-lookback-hours 24 --max-open-positions 50 }"
```

### 4) Fetch Binance testnet order history
```
cd "C:\Users\Edgar.000\Documents\____Trading strategies\Supertrend_5Min"
python BinTestnetOrderHistory.py
```
Shows side/type/status, average price, executed qty, and timestamps per symbol.

### 5) Clear positions only (paper state)
```
cd "C:\Users\Edgar.000\Documents\____Trading strategies\Supertrend_5Min"
python paper_trader.py --clear-positions
```

## Notes
- Use `--max-open-positions` when forcing many entries (e.g., 50).
- Keep stakes low on testnet to avoid insufficient balance.
- `--clear-all` deletes generated reports; rerun simulations afterward.

## Git
Stage and commit when ready:
```
cd "C:\Users\Edgar.000\Documents\____Trading strategies\Supertrend_5Min"
git status
git add README.md
git commit -m "Add helper README for simulations and testnet ops"
```

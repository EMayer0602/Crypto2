# Configuration Guide - Avoid Git Conflicts

## Problem
When testing, you need to modify config variables in `Supertrend_5Min.py`, which causes git conflicts when pulling updates.

## Solution: Local Configuration Override

We use a **local configuration file** that is gitignored, so your personal settings never conflict with server updates.

---

## How to Use Local Configuration

### Step 1: Create Your Local Config File

Copy the example template:

```bash
cp config_local.example.py config_local.py
```

### Step 2: Edit Your Local Settings

Open `config_local.py` and uncomment/modify only the variables you want to override:

```python
# Example: Test with only 2 symbols and JMA filter disabled
SYMBOLS = ["ETH/USDT", "BTC/USDT"]
USE_JMA_TREND_FILTER = False
JMA_TREND_LENGTH = 30
ATR_STOP_MULTS = [None, 1.0, 2.0]
```

### Step 3: Run Your Scripts Normally

Your local settings will automatically be loaded:

```bash
python Supertrend_5Min.py
```

You'll see:
```
[Config] Loaded local configuration overrides from config_local.py
```

---

## Benefits

✅ **No Git Conflicts** - `config_local.py` is gitignored
✅ **Safe Updates** - Pull server changes without losing your settings
✅ **Easy Testing** - Quickly switch between test configurations
✅ **Clean Commits** - Never accidentally commit test settings

---

## What's Ignored

The following files are gitignored and won't cause conflicts:

### Generated Files
- `*.csv` (backtest results)
- `*.html` (reports)
- `*.png` (charts)
- `*.json` (trading state, simulation logs)

### Local Configuration
- `config_local.py` (your personal settings)

### Runtime Data
- `paper_trading_*.csv`
- `paper_trading_*.json`
- `merge_history.json`

---

## Workflow Example

### Testing New Parameters

1. **Edit local config:**
   ```python
   # config_local.py
   USE_JMA_TREND_FILTER = False  # Test without filter
   SYMBOLS = ["ETH/USDT"]        # Quick test with 1 symbol
   MIN_HOLD_BAR_VALUES = [0]     # Fast backtest
   ```

2. **Run backtest:**
   ```bash
   .\run_backtest.bat
   ```

3. **Pull server updates anytime:**
   ```bash
   git pull origin claude/testing-mj7s6ckof32fqz1y-EtB57
   ```
   No conflicts! 🎉

### Production Settings

When you find optimal parameters, you can:
- Keep them in `config_local.py` for personal use
- Or request me to update the defaults in `Supertrend_5Min.py`

---

## Troubleshooting

### "I still get conflicts"

Check if you have uncommitted changes to `Supertrend_5Min.py`:

```bash
git status
```

If yes, reset to server version (your local config is safe!):

```bash
git reset --hard origin/claude/testing-mj7s6ckof32fqz1y-EtB57
```

### "My settings aren't loading"

Make sure `config_local.py` exists and has valid Python syntax:

```bash
python -c "import config_local; print('Config OK')"
```

---

## Questions?

Just ask! This system keeps your testing flexible while avoiding git conflicts.

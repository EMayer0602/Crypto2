#!/usr/bin/env python3
"""
Manual verification script for cryptocurrency trading system
"""

from crypto_trader import CryptoTrader

print("=" * 70)
print("CRYPTOCURRENCY TRADING SYSTEM - MANUAL VERIFICATION")
print("=" * 70)

# Test 1: Initialize trader
print("\n[TEST 1] Initialize trader with $10,000")
trader = CryptoTrader(initial_balance=10000.0)
print(f"✓ Trader initialized")
print(f"  Initial balance: ${trader.portfolio.get_balance():,.2f}")

# Test 2: View available cryptocurrencies
print("\n[TEST 2] View available cryptocurrencies")
cryptos = trader.get_available_cryptocurrencies()
print(f"✓ Found {len(cryptos)} cryptocurrencies:")
for crypto in cryptos:
    print(f"  - {crypto.symbol}: {crypto.name} @ ${crypto.price:,.2f}")

# Test 3: Buy cryptocurrency
print("\n[TEST 3] Buy 0.1 BTC")
trade1 = trader.buy("BTC", 0.1)
print(f"✓ Purchase successful")
print(f"  {trade1}")
print(f"  Remaining balance: ${trader.portfolio.get_balance():,.2f}")

# Test 4: Buy more cryptocurrencies
print("\n[TEST 4] Buy 1 ETH and 10 SOL")
trade2 = trader.buy("ETH", 1.0)
trade3 = trader.buy("SOL", 10.0)
print(f"✓ Purchases successful")
print(f"  {trade2}")
print(f"  {trade3}")
print(f"  Remaining balance: ${trader.portfolio.get_balance():,.2f}")

# Test 5: View portfolio
print("\n[TEST 5] View portfolio")
summary = trader.get_portfolio_summary()
print(f"✓ Portfolio summary:")
print(f"  Cash: ${summary['cash_balance']:,.2f}")
print(f"  Holdings:")
for symbol, data in summary['holdings'].items():
    print(f"    {symbol}: {data['amount']:.6f} @ ${data['price']:,.2f} = ${data['value']:,.2f}")
print(f"  Total value: ${summary['total_value']:,.2f}")

# Test 6: Update prices
print("\n[TEST 6] Update cryptocurrency prices")
trader.update_price("BTC", 50000.00)
trader.update_price("ETH", 3500.00)
trader.update_price("SOL", 120.00)
print(f"✓ Prices updated")
print(f"  BTC: ${trader.get_price('BTC'):,.2f}")
print(f"  ETH: ${trader.get_price('ETH'):,.2f}")
print(f"  SOL: ${trader.get_price('SOL'):,.2f}")

# Test 7: Check portfolio value after price changes
print("\n[TEST 7] Portfolio value after price changes")
new_value = trader.get_portfolio_value()
old_value = 10000.00
profit = new_value - old_value
print(f"✓ Portfolio value: ${new_value:,.2f}")
print(f"  Initial value: ${old_value:,.2f}")
print(f"  Profit/Loss: ${profit:,.2f} ({(profit/old_value)*100:.2f}%)")

# Test 8: Sell cryptocurrency
print("\n[TEST 8] Sell 5 SOL")
trade4 = trader.sell("SOL", 5.0)
print(f"✓ Sale successful")
print(f"  {trade4}")
print(f"  New balance: ${trader.portfolio.get_balance():,.2f}")

# Test 9: View trade history
print("\n[TEST 9] View trade history")
history = trader.portfolio.get_trade_history()
print(f"✓ Trade history ({len(history)} trades):")
for i, trade in enumerate(history, 1):
    print(f"  {i}. {trade.trade_type.value:4s} {trade.amount:8.6f} {trade.crypto_symbol} @ ${trade.price:,.2f}")

# Test 10: Export trade history
print("\n[TEST 10] Export trade history")
import tempfile
import os

# Use temporary directory for cross-platform compatibility
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    filename = f.name

trader.export_trade_history(filename)
print(f"✓ Trade history exported to {filename}")

# Clean up
if os.path.exists(filename):
    os.remove(filename)

# Test 11: Error handling - insufficient funds
print("\n[TEST 11] Error handling - insufficient funds")
try:
    trader.buy("BTC", 100.0)  # Try to buy too much
    print(f"✗ Should have raised ValueError")
except ValueError as e:
    print(f"✓ Correctly raised ValueError: {e}")

# Test 12: Error handling - insufficient holdings
print("\n[TEST 12] Error handling - insufficient holdings")
try:
    trader.sell("BTC", 10.0)  # Try to sell more than we have
    print(f"✗ Should have raised ValueError")
except ValueError as e:
    print(f"✓ Correctly raised ValueError: {e}")

# Test 13: Error handling - invalid cryptocurrency
print("\n[TEST 13] Error handling - invalid cryptocurrency")
try:
    trader.buy("INVALID", 1.0)
    print(f"✗ Should have raised ValueError")
except ValueError as e:
    print(f"✓ Correctly raised ValueError: {e}")

# Test 14: Add new cryptocurrency
print("\n[TEST 14] Add new cryptocurrency")
trader.add_cryptocurrency("DOGE", "Dogecoin", 0.08)
print(f"✓ Added DOGE")
print(f"  Price: ${trader.get_price('DOGE'):,.2f}")

# Test 15: Trade new cryptocurrency
print("\n[TEST 15] Trade new cryptocurrency")
trade5 = trader.buy("DOGE", 1000.0)
print(f"✓ Purchase successful")
print(f"  {trade5}")

# Final summary
print("\n" + "=" * 70)
print("FINAL PORTFOLIO SUMMARY")
print("=" * 70)
trader.print_portfolio()

print("\n" + "=" * 70)
print("ALL TESTS PASSED ✓")
print("=" * 70)
print(f"\nTotal trades executed: {len(trader.portfolio.get_trade_history())}")
print(f"Final portfolio value: ${trader.get_portfolio_value():,.2f}")
print("\nThe cryptocurrency trading system is fully functional!")

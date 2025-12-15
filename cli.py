#!/usr/bin/env python3
"""
Interactive CLI for Cryptocurrency Trading System
"""

import sys
from crypto_trader import CryptoTrader


def print_menu():
    """Print the main menu"""
    print("\n" + "="*60)
    print("CRYPTOCURRENCY TRADING SYSTEM")
    print("="*60)
    print("1. View available cryptocurrencies")
    print("2. View portfolio")
    print("3. Buy cryptocurrency")
    print("4. Sell cryptocurrency")
    print("5. Update cryptocurrency price")
    print("6. View trade history")
    print("7. Export trade history")
    print("8. Exit")
    print("="*60)


def view_cryptocurrencies(trader):
    """Display available cryptocurrencies"""
    print("\nAvailable Cryptocurrencies:")
    print("-" * 60)
    cryptos = trader.get_available_cryptocurrencies()
    for crypto in cryptos:
        print(f"  {crypto.symbol:6s} - {crypto.name:20s} ${crypto.price:>12,.2f}")
    print("-" * 60)


def view_portfolio(trader):
    """Display portfolio information"""
    trader.print_portfolio()


def buy_cryptocurrency(trader):
    """Buy cryptocurrency"""
    try:
        symbol = input("Enter cryptocurrency symbol (e.g., BTC): ").strip().upper()
        amount = float(input(f"Enter amount of {symbol} to buy: ").strip())
        
        trade = trader.buy(symbol, amount)
        print(f"\n✓ Success! {trade}")
        print(f"  Total cost: ${trade.total_value:.2f}")
        print(f"  Remaining balance: ${trader.portfolio.get_balance():.2f}")
        
    except ValueError as e:
        print(f"\n✗ Error: {e}")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")


def sell_cryptocurrency(trader):
    """Sell cryptocurrency"""
    try:
        symbol = input("Enter cryptocurrency symbol (e.g., BTC): ").strip().upper()
        amount = float(input(f"Enter amount of {symbol} to sell: ").strip())
        
        trade = trader.sell(symbol, amount)
        print(f"\n✓ Success! {trade}")
        print(f"  Total value: ${trade.total_value:.2f}")
        print(f"  New balance: ${trader.portfolio.get_balance():.2f}")
        
    except ValueError as e:
        print(f"\n✗ Error: {e}")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")


def update_price(trader):
    """Update cryptocurrency price"""
    try:
        symbol = input("Enter cryptocurrency symbol (e.g., BTC): ").strip().upper()
        new_price = float(input(f"Enter new price for {symbol}: $").strip())
        
        trader.update_price(symbol, new_price)
        print(f"\n✓ Success! {symbol} price updated to ${new_price:.2f}")
        
    except ValueError as e:
        print(f"\n✗ Error: {e}")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")


def view_trade_history(trader):
    """Display trade history"""
    history = trader.portfolio.get_trade_history()
    
    if not history:
        print("\nNo trades yet.")
        return
    
    print("\nTrade History:")
    print("-" * 60)
    for i, trade in enumerate(history, 1):
        print(f"{i:3d}. {trade.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | {trade}")
    print("-" * 60)
    print(f"Total trades: {len(history)}")


def export_history(trader):
    """Export trade history to file"""
    try:
        filename = input("Enter filename (default: trade_history.json): ").strip()
        if not filename:
            filename = "trade_history.json"
        
        trader.export_trade_history(filename)
        print(f"\n✓ Trade history exported to {filename}")
        
    except Exception as e:
        print(f"\n✗ Error exporting trade history: {e}")


def main():
    """Main CLI loop"""
    print("Welcome to the Cryptocurrency Trading System!")
    
    # Get initial balance
    while True:
        try:
            balance_input = input("\nEnter initial balance (default: $10,000): $").strip()
            initial_balance = float(balance_input) if balance_input else 10000.0
            if initial_balance <= 0:
                print("Balance must be positive. Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Initialize trader
    trader = CryptoTrader(initial_balance=initial_balance)
    print(f"\n✓ Trading account created with ${initial_balance:,.2f}")
    
    # Main menu loop
    while True:
        print_menu()
        choice = input("\nEnter your choice (1-8): ").strip()
        
        if choice == '1':
            view_cryptocurrencies(trader)
        elif choice == '2':
            view_portfolio(trader)
        elif choice == '3':
            buy_cryptocurrency(trader)
        elif choice == '4':
            sell_cryptocurrency(trader)
        elif choice == '5':
            update_price(trader)
        elif choice == '6':
            view_trade_history(trader)
        elif choice == '7':
            export_history(trader)
        elif choice == '8':
            print("\nThank you for using the Cryptocurrency Trading System!")
            print("Goodbye!\n")
            sys.exit(0)
        else:
            print("\n✗ Invalid choice. Please enter a number between 1 and 8.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTrading session interrupted. Goodbye!")
        sys.exit(0)

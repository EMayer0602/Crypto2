#!/usr/bin/env python3
"""
Cryptocurrency Trading System
A simple but functional cryptocurrency trading implementation.
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum


class TradeType(Enum):
    """Types of trades"""
    BUY = "BUY"
    SELL = "SELL"


class CryptoCurrency:
    """Represents a cryptocurrency"""
    
    def __init__(self, symbol: str, name: str, price: float):
        self.symbol = symbol.upper()
        self.name = name
        self.price = price
    
    def update_price(self, new_price: float):
        """Update the current price of the cryptocurrency"""
        if new_price <= 0:
            raise ValueError("Price must be positive")
        self.price = new_price
    
    def __repr__(self):
        return f"CryptoCurrency({self.symbol}, ${self.price:.2f})"


class Trade:
    """Represents a single trade transaction"""
    
    def __init__(self, trade_type: TradeType, crypto: CryptoCurrency, 
                 amount: float, price: float, timestamp: Optional[datetime] = None):
        self.trade_type = trade_type
        self.crypto_symbol = crypto.symbol
        self.crypto_name = crypto.name
        self.amount = amount
        self.price = price
        self.timestamp = timestamp or datetime.now()
        self.total_value = amount * price
    
    def to_dict(self) -> dict:
        """Convert trade to dictionary"""
        return {
            'type': self.trade_type.value,
            'symbol': self.crypto_symbol,
            'name': self.crypto_name,
            'amount': self.amount,
            'price': self.price,
            'total_value': self.total_value,
            'timestamp': self.timestamp.isoformat()
        }
    
    def __repr__(self):
        return (f"Trade({self.trade_type.value} {self.amount:.6f} {self.crypto_symbol} "
                f"@ ${self.price:.2f} = ${self.total_value:.2f})")


class Portfolio:
    """Manages a cryptocurrency portfolio"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.balance = initial_balance
        self.holdings: Dict[str, float] = {}
        self.trade_history: List[Trade] = []
    
    def get_holdings(self) -> Dict[str, float]:
        """Get current holdings"""
        return self.holdings.copy()
    
    def get_balance(self) -> float:
        """Get current cash balance"""
        return self.balance
    
    def add_trade(self, trade: Trade):
        """Add a trade to history"""
        self.trade_history.append(trade)
    
    def get_trade_history(self) -> List[Trade]:
        """Get all trades"""
        return self.trade_history.copy()
    
    def __repr__(self):
        return f"Portfolio(balance=${self.balance:.2f}, holdings={self.holdings})"


class CryptoTrader:
    """Main cryptocurrency trading system"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.portfolio = Portfolio(initial_balance)
        self.cryptocurrencies: Dict[str, CryptoCurrency] = {}
        self._initialize_default_cryptos()
    
    def _initialize_default_cryptos(self):
        """Initialize with some default cryptocurrencies"""
        default_cryptos = [
            CryptoCurrency("BTC", "Bitcoin", 45000.00),
            CryptoCurrency("ETH", "Ethereum", 3000.00),
            CryptoCurrency("USDT", "Tether", 1.00),
            CryptoCurrency("BNB", "Binance Coin", 350.00),
            CryptoCurrency("SOL", "Solana", 100.00),
        ]
        for crypto in default_cryptos:
            self.cryptocurrencies[crypto.symbol] = crypto
    
    def add_cryptocurrency(self, symbol: str, name: str, price: float):
        """Add a new cryptocurrency to the trading system"""
        crypto = CryptoCurrency(symbol, name, price)
        self.cryptocurrencies[crypto.symbol] = crypto
        return crypto
    
    def update_price(self, symbol: str, new_price: float):
        """Update the price of a cryptocurrency"""
        symbol = symbol.upper()
        if symbol not in self.cryptocurrencies:
            raise ValueError(f"Cryptocurrency {symbol} not found")
        self.cryptocurrencies[symbol].update_price(new_price)
    
    def get_price(self, symbol: str) -> float:
        """Get current price of a cryptocurrency"""
        symbol = symbol.upper()
        if symbol not in self.cryptocurrencies:
            raise ValueError(f"Cryptocurrency {symbol} not found")
        return self.cryptocurrencies[symbol].price
    
    def buy(self, symbol: str, amount: float) -> Trade:
        """
        Buy cryptocurrency
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC')
            amount: Amount of cryptocurrency to buy
            
        Returns:
            Trade object representing the transaction
            
        Raises:
            ValueError: If insufficient funds or invalid parameters
        """
        symbol = symbol.upper()
        
        if symbol not in self.cryptocurrencies:
            raise ValueError(f"Cryptocurrency {symbol} not found")
        
        if amount <= 0:
            raise ValueError("Amount must be positive")
        
        crypto = self.cryptocurrencies[symbol]
        total_cost = amount * crypto.price
        
        if total_cost > self.portfolio.balance:
            raise ValueError(
                f"Insufficient funds: need ${total_cost:.2f}, have ${self.portfolio.balance:.2f}"
            )
        
        # Execute trade
        self.portfolio.balance -= total_cost
        if symbol in self.portfolio.holdings:
            self.portfolio.holdings[symbol] += amount
        else:
            self.portfolio.holdings[symbol] = amount
        
        # Record trade
        trade = Trade(TradeType.BUY, crypto, amount, crypto.price)
        self.portfolio.add_trade(trade)
        
        return trade
    
    def sell(self, symbol: str, amount: float) -> Trade:
        """
        Sell cryptocurrency
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC')
            amount: Amount of cryptocurrency to sell
            
        Returns:
            Trade object representing the transaction
            
        Raises:
            ValueError: If insufficient holdings or invalid parameters
        """
        symbol = symbol.upper()
        
        if symbol not in self.cryptocurrencies:
            raise ValueError(f"Cryptocurrency {symbol} not found")
        
        if amount <= 0:
            raise ValueError("Amount must be positive")
        
        if symbol not in self.portfolio.holdings or self.portfolio.holdings[symbol] < amount:
            current_amount = self.portfolio.holdings.get(symbol, 0)
            raise ValueError(
                f"Insufficient holdings: trying to sell {amount}, have {current_amount}"
            )
        
        crypto = self.cryptocurrencies[symbol]
        total_value = amount * crypto.price
        
        # Execute trade
        self.portfolio.balance += total_value
        self.portfolio.holdings[symbol] -= amount
        
        # Remove from holdings if amount is zero
        if self.portfolio.holdings[symbol] == 0:
            del self.portfolio.holdings[symbol]
        
        # Record trade
        trade = Trade(TradeType.SELL, crypto, amount, crypto.price)
        self.portfolio.add_trade(trade)
        
        return trade
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value (cash + holdings)"""
        total = self.portfolio.balance
        
        for symbol, amount in self.portfolio.holdings.items():
            if symbol in self.cryptocurrencies:
                total += amount * self.cryptocurrencies[symbol].price
        
        return total
    
    def get_portfolio_summary(self) -> dict:
        """Get a summary of the portfolio"""
        holdings_value = {}
        for symbol, amount in self.portfolio.holdings.items():
            if symbol in self.cryptocurrencies:
                price = self.cryptocurrencies[symbol].price
                value = amount * price
                holdings_value[symbol] = {
                    'amount': amount,
                    'price': price,
                    'value': value
                }
        
        return {
            'cash_balance': self.portfolio.balance,
            'holdings': holdings_value,
            'total_value': self.get_portfolio_value()
        }
    
    def get_available_cryptocurrencies(self) -> List[CryptoCurrency]:
        """Get list of all available cryptocurrencies"""
        return list(self.cryptocurrencies.values())
    
    def export_trade_history(self, filename: str):
        """Export trade history to JSON file"""
        trades_data = [trade.to_dict() for trade in self.portfolio.trade_history]
        with open(filename, 'w') as f:
            json.dump(trades_data, f, indent=2)
    
    def print_portfolio(self):
        """Print portfolio summary to console"""
        summary = self.get_portfolio_summary()
        
        print("\n" + "="*60)
        print("PORTFOLIO SUMMARY")
        print("="*60)
        print(f"Cash Balance: ${summary['cash_balance']:.2f}")
        print("\nHoldings:")
        
        if summary['holdings']:
            for symbol, data in summary['holdings'].items():
                print(f"  {symbol}: {data['amount']:.6f} @ ${data['price']:.2f} = ${data['value']:.2f}")
        else:
            print("  (none)")
        
        print(f"\nTotal Portfolio Value: ${summary['total_value']:.2f}")
        print("="*60 + "\n")


def main():
    """Example usage of the cryptocurrency trading system"""
    print("Cryptocurrency Trading System")
    print("="*60)
    
    # Initialize trader with $10,000
    trader = CryptoTrader(initial_balance=10000.0)
    
    # Show available cryptocurrencies
    print("\nAvailable Cryptocurrencies:")
    for crypto in trader.get_available_cryptocurrencies():
        print(f"  {crypto.symbol}: {crypto.name} - ${crypto.price:.2f}")
    
    # Initial portfolio
    trader.print_portfolio()
    
    # Example trades
    print("Executing trades...")
    
    try:
        # Buy 0.1 BTC
        trade1 = trader.buy("BTC", 0.1)
        print(f"✓ {trade1}")
        
        # Buy 1 ETH (reduced from 2.0)
        trade2 = trader.buy("ETH", 1.0)
        print(f"✓ {trade2}")
        
        # Buy 10 SOL
        trade3 = trader.buy("SOL", 10.0)
        print(f"✓ {trade3}")
        
        # Show portfolio after purchases
        trader.print_portfolio()
        
        # Simulate price changes
        print("Simulating price changes...")
        trader.update_price("BTC", 47000.00)  # BTC goes up
        trader.update_price("ETH", 2900.00)   # ETH goes down
        trader.update_price("SOL", 110.00)    # SOL goes up
        print("✓ Prices updated")
        
        # Show portfolio after price changes
        trader.print_portfolio()
        
        # Sell some holdings
        print("Selling some holdings...")
        trade4 = trader.sell("SOL", 5.0)
        print(f"✓ {trade4}")
        
        # Final portfolio
        trader.print_portfolio()
        
        # Show trade history
        print("Trade History:")
        for i, trade in enumerate(trader.portfolio.get_trade_history(), 1):
            print(f"  {i}. {trade}")
        
        # Export trade history
        trader.export_trade_history("trade_history.json")
        print("\n✓ Trade history exported to trade_history.json")
        
    except ValueError as e:
        print(f"✗ Error: {e}")
    
    print("\nTrading session complete!")


if __name__ == "__main__":
    main()

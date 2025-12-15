#!/usr/bin/env python3
"""
Unit tests for the cryptocurrency trading system
"""

import unittest
import os
from datetime import datetime
from crypto_trader import (
    CryptoCurrency, Trade, TradeType, Portfolio, 
    CryptoTrader
)


class TestCryptoCurrency(unittest.TestCase):
    """Test CryptoCurrency class"""
    
    def test_create_cryptocurrency(self):
        """Test creating a cryptocurrency"""
        crypto = CryptoCurrency("BTC", "Bitcoin", 45000.00)
        self.assertEqual(crypto.symbol, "BTC")
        self.assertEqual(crypto.name, "Bitcoin")
        self.assertEqual(crypto.price, 45000.00)
    
    def test_symbol_uppercase(self):
        """Test that symbol is converted to uppercase"""
        crypto = CryptoCurrency("btc", "Bitcoin", 45000.00)
        self.assertEqual(crypto.symbol, "BTC")
    
    def test_update_price(self):
        """Test updating cryptocurrency price"""
        crypto = CryptoCurrency("BTC", "Bitcoin", 45000.00)
        crypto.update_price(50000.00)
        self.assertEqual(crypto.price, 50000.00)
    
    def test_update_price_invalid(self):
        """Test that negative prices raise error"""
        crypto = CryptoCurrency("BTC", "Bitcoin", 45000.00)
        with self.assertRaises(ValueError):
            crypto.update_price(-100.00)
        with self.assertRaises(ValueError):
            crypto.update_price(0)


class TestTrade(unittest.TestCase):
    """Test Trade class"""
    
    def test_create_buy_trade(self):
        """Test creating a buy trade"""
        crypto = CryptoCurrency("BTC", "Bitcoin", 45000.00)
        trade = Trade(TradeType.BUY, crypto, 0.1, 45000.00)
        
        self.assertEqual(trade.trade_type, TradeType.BUY)
        self.assertEqual(trade.crypto_symbol, "BTC")
        self.assertEqual(trade.amount, 0.1)
        self.assertEqual(trade.price, 45000.00)
        self.assertEqual(trade.total_value, 4500.00)
    
    def test_create_sell_trade(self):
        """Test creating a sell trade"""
        crypto = CryptoCurrency("ETH", "Ethereum", 3000.00)
        trade = Trade(TradeType.SELL, crypto, 2.0, 3000.00)
        
        self.assertEqual(trade.trade_type, TradeType.SELL)
        self.assertEqual(trade.crypto_symbol, "ETH")
        self.assertEqual(trade.amount, 2.0)
        self.assertEqual(trade.total_value, 6000.00)
    
    def test_trade_to_dict(self):
        """Test converting trade to dictionary"""
        crypto = CryptoCurrency("BTC", "Bitcoin", 45000.00)
        trade = Trade(TradeType.BUY, crypto, 0.1, 45000.00)
        trade_dict = trade.to_dict()
        
        self.assertEqual(trade_dict['type'], 'BUY')
        self.assertEqual(trade_dict['symbol'], 'BTC')
        self.assertEqual(trade_dict['amount'], 0.1)
        self.assertEqual(trade_dict['price'], 45000.00)
        self.assertEqual(trade_dict['total_value'], 4500.00)


class TestPortfolio(unittest.TestCase):
    """Test Portfolio class"""
    
    def test_create_portfolio(self):
        """Test creating a portfolio"""
        portfolio = Portfolio(10000.0)
        self.assertEqual(portfolio.balance, 10000.0)
        self.assertEqual(len(portfolio.holdings), 0)
        self.assertEqual(len(portfolio.trade_history), 0)
    
    def test_get_balance(self):
        """Test getting balance"""
        portfolio = Portfolio(5000.0)
        self.assertEqual(portfolio.get_balance(), 5000.0)
    
    def test_get_holdings(self):
        """Test getting holdings"""
        portfolio = Portfolio(10000.0)
        portfolio.holdings["BTC"] = 0.5
        holdings = portfolio.get_holdings()
        self.assertEqual(holdings["BTC"], 0.5)


class TestCryptoTrader(unittest.TestCase):
    """Test CryptoTrader class"""
    
    def setUp(self):
        """Set up test trader"""
        self.trader = CryptoTrader(initial_balance=10000.0)
    
    def test_create_trader(self):
        """Test creating a trader"""
        self.assertEqual(self.trader.portfolio.balance, 10000.0)
        self.assertGreater(len(self.trader.cryptocurrencies), 0)
    
    def test_add_cryptocurrency(self):
        """Test adding a new cryptocurrency"""
        crypto = self.trader.add_cryptocurrency("DOGE", "Dogecoin", 0.10)
        self.assertEqual(crypto.symbol, "DOGE")
        self.assertIn("DOGE", self.trader.cryptocurrencies)
    
    def test_get_price(self):
        """Test getting cryptocurrency price"""
        price = self.trader.get_price("BTC")
        self.assertEqual(price, 45000.00)
    
    def test_get_price_not_found(self):
        """Test getting price of non-existent cryptocurrency"""
        with self.assertRaises(ValueError):
            self.trader.get_price("NOTFOUND")
    
    def test_update_price(self):
        """Test updating cryptocurrency price"""
        self.trader.update_price("BTC", 50000.00)
        self.assertEqual(self.trader.get_price("BTC"), 50000.00)
    
    def test_buy_cryptocurrency(self):
        """Test buying cryptocurrency"""
        trade = self.trader.buy("BTC", 0.1)
        
        self.assertEqual(trade.trade_type, TradeType.BUY)
        self.assertEqual(trade.amount, 0.1)
        self.assertEqual(self.trader.portfolio.holdings["BTC"], 0.1)
        self.assertEqual(self.trader.portfolio.balance, 10000.0 - 4500.0)
    
    def test_buy_multiple_times(self):
        """Test buying the same cryptocurrency multiple times"""
        self.trader.buy("BTC", 0.1)
        self.trader.buy("BTC", 0.05)
        
        self.assertAlmostEqual(self.trader.portfolio.holdings["BTC"], 0.15, places=6)
    
    def test_buy_insufficient_funds(self):
        """Test buying with insufficient funds"""
        with self.assertRaises(ValueError) as context:
            self.trader.buy("BTC", 1.0)  # Costs $45,000
        self.assertIn("Insufficient funds", str(context.exception))
    
    def test_buy_invalid_amount(self):
        """Test buying with invalid amount"""
        with self.assertRaises(ValueError):
            self.trader.buy("BTC", -0.1)
        with self.assertRaises(ValueError):
            self.trader.buy("BTC", 0)
    
    def test_buy_not_found(self):
        """Test buying non-existent cryptocurrency"""
        with self.assertRaises(ValueError):
            self.trader.buy("NOTFOUND", 1.0)
    
    def test_sell_cryptocurrency(self):
        """Test selling cryptocurrency"""
        # First buy some
        self.trader.buy("BTC", 0.1)
        initial_balance = self.trader.portfolio.balance
        
        # Then sell
        trade = self.trader.sell("BTC", 0.05)
        
        self.assertEqual(trade.trade_type, TradeType.SELL)
        self.assertEqual(trade.amount, 0.05)
        self.assertEqual(self.trader.portfolio.holdings["BTC"], 0.05)
        self.assertEqual(self.trader.portfolio.balance, initial_balance + 2250.0)
    
    def test_sell_all_holdings(self):
        """Test selling all holdings removes from portfolio"""
        self.trader.buy("BTC", 0.1)
        self.trader.sell("BTC", 0.1)
        
        self.assertNotIn("BTC", self.trader.portfolio.holdings)
    
    def test_sell_insufficient_holdings(self):
        """Test selling with insufficient holdings"""
        with self.assertRaises(ValueError) as context:
            self.trader.sell("BTC", 0.1)
        self.assertIn("Insufficient holdings", str(context.exception))
    
    def test_sell_invalid_amount(self):
        """Test selling with invalid amount"""
        self.trader.buy("BTC", 0.1)
        
        with self.assertRaises(ValueError):
            self.trader.sell("BTC", -0.05)
        with self.assertRaises(ValueError):
            self.trader.sell("BTC", 0)
    
    def test_get_portfolio_value(self):
        """Test calculating portfolio value"""
        # Initial value should be just the cash balance
        self.assertEqual(self.trader.get_portfolio_value(), 10000.0)
        
        # Buy some crypto
        self.trader.buy("BTC", 0.1)  # Costs $4,500
        self.trader.buy("ETH", 1.0)  # Costs $3,000
        
        # Portfolio value should be cash + holdings value
        expected_value = 2500.0 + (0.1 * 45000.0) + (1.0 * 3000.0)
        self.assertEqual(self.trader.get_portfolio_value(), expected_value)
    
    def test_get_portfolio_summary(self):
        """Test getting portfolio summary"""
        self.trader.buy("BTC", 0.1)
        summary = self.trader.get_portfolio_summary()
        
        self.assertIn('cash_balance', summary)
        self.assertIn('holdings', summary)
        self.assertIn('total_value', summary)
        self.assertIn('BTC', summary['holdings'])
    
    def test_get_available_cryptocurrencies(self):
        """Test getting available cryptocurrencies"""
        cryptos = self.trader.get_available_cryptocurrencies()
        self.assertGreater(len(cryptos), 0)
        self.assertIsInstance(cryptos[0], CryptoCurrency)
    
    def test_trade_history(self):
        """Test trade history tracking"""
        self.trader.buy("BTC", 0.1)
        self.trader.buy("ETH", 1.0)
        self.trader.sell("BTC", 0.05)
        
        history = self.trader.portfolio.get_trade_history()
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0].trade_type, TradeType.BUY)
        self.assertEqual(history[2].trade_type, TradeType.SELL)
    
    def test_export_trade_history(self):
        """Test exporting trade history"""
        self.trader.buy("BTC", 0.1)
        self.trader.sell("BTC", 0.05)
        
        filename = "/tmp/test_trade_history.json"
        self.trader.export_trade_history(filename)
        
        self.assertTrue(os.path.exists(filename))
        
        # Clean up
        if os.path.exists(filename):
            os.remove(filename)
    
    def test_price_changes_affect_portfolio_value(self):
        """Test that price changes affect portfolio value"""
        # Buy cryptocurrency
        self.trader.buy("BTC", 0.1)
        initial_value = self.trader.get_portfolio_value()
        
        # Update price
        self.trader.update_price("BTC", 50000.00)
        new_value = self.trader.get_portfolio_value()
        
        # Value should increase
        self.assertGreater(new_value, initial_value)
        self.assertEqual(new_value - initial_value, 0.1 * (50000.00 - 45000.00))


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows"""
    
    def test_complete_trading_workflow(self):
        """Test a complete trading workflow"""
        trader = CryptoTrader(initial_balance=20000.0)  # Increased to afford all purchases
        
        # Buy multiple cryptocurrencies
        trader.buy("BTC", 0.1)  # Costs $4,500
        trader.buy("ETH", 2.0)  # Costs $6,000
        trader.buy("SOL", 10.0)  # Costs $1,000
        
        # Check portfolio
        summary = trader.get_portfolio_summary()
        self.assertEqual(len(summary['holdings']), 3)
        
        # Simulate price changes
        trader.update_price("BTC", 47000.00)
        trader.update_price("ETH", 2900.00)
        
        # Sell some holdings
        trader.sell("SOL", 5.0)
        
        # Verify final state
        self.assertIn("BTC", trader.portfolio.holdings)
        self.assertIn("ETH", trader.portfolio.holdings)
        self.assertIn("SOL", trader.portfolio.holdings)
        self.assertEqual(len(trader.portfolio.trade_history), 4)


if __name__ == '__main__':
    unittest.main()

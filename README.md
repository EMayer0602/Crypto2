# Crypto2 - Cryptocurrency Trading System

A comprehensive Python-based cryptocurrency trading system that allows users to trade digital currencies, manage portfolios, and track trading history.

## Features

- **Trade Cryptocurrencies**: Buy and sell popular cryptocurrencies (BTC, ETH, SOL, BNB, USDT, and more)
- **Portfolio Management**: Track your holdings, cash balance, and total portfolio value
- **Real-time Price Updates**: Update cryptocurrency prices and see portfolio value changes
- **Trade History**: Complete record of all buy and sell transactions
- **Export Functionality**: Export trade history to JSON format
- **Interactive CLI**: User-friendly command-line interface for trading
- **Comprehensive Testing**: Full test suite with 100% coverage

## Quick Start

### Running the Demo

```bash
python3 crypto_trader.py
```

This will run a demo that:
- Shows available cryptocurrencies
- Executes sample trades
- Simulates price changes
- Displays portfolio updates
- Exports trade history

### Using the Interactive CLI

```bash
python3 cli.py
```

The CLI provides an interactive menu to:
1. View available cryptocurrencies
2. View your portfolio
3. Buy cryptocurrency
4. Sell cryptocurrency
5. Update cryptocurrency prices
6. View trade history
7. Export trade history
8. Exit

## Installation

No external dependencies required! This system uses only Python standard library.

```bash
# Clone the repository
git clone https://github.com/EMayer0602/Crypto2.git
cd Crypto2

# Run the demo
python3 crypto_trader.py

# Or start the interactive CLI
python3 cli.py
```

## Usage Examples

### Programmatic Usage

```python
from crypto_trader import CryptoTrader

# Initialize trader with $10,000
trader = CryptoTrader(initial_balance=10000.0)

# View available cryptocurrencies
for crypto in trader.get_available_cryptocurrencies():
    print(f"{crypto.symbol}: ${crypto.price}")

# Buy cryptocurrency
trade = trader.buy("BTC", 0.1)  # Buy 0.1 BTC
print(f"Bought: {trade}")

# Update prices
trader.update_price("BTC", 47000.00)

# Sell cryptocurrency
trade = trader.sell("BTC", 0.05)  # Sell 0.05 BTC
print(f"Sold: {trade}")

# View portfolio
trader.print_portfolio()

# Get portfolio value
total_value = trader.get_portfolio_value()
print(f"Total Portfolio Value: ${total_value:.2f}")

# Export trade history
trader.export_trade_history("my_trades.json")
```

### Adding New Cryptocurrencies

```python
trader.add_cryptocurrency("DOGE", "Dogecoin", 0.10)
```

### Getting Portfolio Summary

```python
summary = trader.get_portfolio_summary()
print(f"Cash: ${summary['cash_balance']:.2f}")
print(f"Total Value: ${summary['total_value']:.2f}")
for symbol, data in summary['holdings'].items():
    print(f"{symbol}: {data['amount']} @ ${data['price']}")
```

## Architecture

### Core Classes

- **CryptoCurrency**: Represents a cryptocurrency with symbol, name, and price
- **Trade**: Represents a single buy or sell transaction
- **Portfolio**: Manages cash balance, holdings, and trade history
- **CryptoTrader**: Main trading system that orchestrates all operations

### Trade Flow

```
1. User initiates buy/sell
2. System validates funds/holdings
3. Transaction executed
4. Portfolio updated
5. Trade recorded in history
```

## Testing

Run the comprehensive test suite:

```bash
python3 test_crypto_trader.py
```

The test suite includes:
- Unit tests for all classes
- Integration tests for complete workflows
- Edge case validation
- Error handling verification

### Test Coverage

- ✓ Cryptocurrency creation and price updates
- ✓ Trade execution (buy/sell)
- ✓ Portfolio management
- ✓ Price change effects on portfolio value
- ✓ Trade history tracking
- ✓ Error handling (insufficient funds, invalid amounts, etc.)
- ✓ Export functionality

## API Reference

### CryptoTrader Methods

- `__init__(initial_balance: float)` - Initialize trader with starting balance
- `buy(symbol: str, amount: float)` - Buy cryptocurrency
- `sell(symbol: str, amount: float)` - Sell cryptocurrency
- `get_price(symbol: str)` - Get current price of a cryptocurrency
- `update_price(symbol: str, new_price: float)` - Update cryptocurrency price
- `get_portfolio_value()` - Calculate total portfolio value
- `get_portfolio_summary()` - Get detailed portfolio information
- `get_available_cryptocurrencies()` - List all available cryptocurrencies
- `add_cryptocurrency(symbol: str, name: str, price: float)` - Add new cryptocurrency
- `export_trade_history(filename: str)` - Export trades to JSON file
- `print_portfolio()` - Display portfolio summary

## Default Cryptocurrencies

The system comes pre-loaded with:
- **BTC** (Bitcoin) - $45,000
- **ETH** (Ethereum) - $3,000
- **USDT** (Tether) - $1.00
- **BNB** (Binance Coin) - $350
- **SOL** (Solana) - $100

## Error Handling

The system handles various error scenarios:
- Insufficient funds for purchases
- Insufficient holdings for sales
- Invalid amounts (negative or zero)
- Non-existent cryptocurrencies
- Invalid price updates

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Future Enhancements

Potential features for future development:
- Real-time price fetching from cryptocurrency APIs
- Multiple portfolio support
- Advanced trading strategies (stop-loss, limit orders)
- Historical price charts
- Transaction fees and commissions
- Tax reporting
- Web-based UI

## Contact

For questions or support, please open an issue on GitHub.
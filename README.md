# 📈 Trading Signal System

A machine learning-powered trading signal generator that predicts when to buy stocks and manages your risk automatically.

---

## What Does This Do?

This system looks at historical stock data (prices, volume, etc.) and uses machine learning to predict when a stock is likely to go up. It tells you:

- **When to buy** (with a confidence score)
- **How many shares to buy** (based on your risk tolerance)
- **Where to set your stop-loss** (to limit losses)
- **When to exit** (take profit or cut losses)

Think of it like having a smart assistant that watches the market and tells you when the odds are in your favor.

---

## How It Works (Simple Version)

1. **Collects Data**: Downloads stock prices from Yahoo Finance
2. **Creates Signals**: Calculates 40+ technical indicators (like RSI, MACD, etc.)
3. **Learns Patterns**: Trains a machine learning model on what worked before
4. **Makes Predictions**: Gives you a probability that the stock will go up
5. **Manages Risk**: Tells you how much to buy and where to exit

---

## Key Features

### 🧠 Two-Model System

The system uses two separate models working together:

| Model | Purpose |
|-------|---------|
| **Alpha Model** | Predicts if price will go UP |
| **Risk Model** | Predicts if the trade will have a big drawdown |

A trade only triggers when:
- Alpha model says "likely to go up"
- Risk model says "probably won't drop much"

### 📊 Technical Indicators Used

The system calculates these indicators automatically:

- **RSI** (7, 14, 21 periods) - measures if stock is overbought/oversold
- **MACD** - shows momentum direction
- **Bollinger Bands** - shows if price is stretched
- **Stochastic Oscillator** - another overbought/oversold measure
- **ADX** - tells if market is trending or choppy
- **ATR** - measures volatility (used for stop-loss)
- **Volume Ratio** - unusual trading activity
- **VIX** - fear gauge for the overall market


## Technical Details

### Machine Learning Model

Uses **LightGBM** (Light Gradient Boosting Machine):
- Fast training on large datasets
- Handles missing values automatically
- Good with imbalanced data (more "hold" than "buy" signals)

### Feature Engineering

All features are **level-invariant**, meaning they work the same whether a stock is at $50 or $500:
- Returns instead of raw prices
- Ratios instead of absolute values
- Normalized oscillators (0-100 scale)

### Train/Validation/Test Split

Data is split chronologically (never randomly):
- **70%** for training
- **15%** for validation (tune threshold)
- **15%** for testing (final results)

---

## Installation

### 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
# or
venv\Scripts\activate  # On Windows
```

### 2. Install dependencies

```bash
pip install lightgbm pandas numpy scikit-learn yfinance
```

---

## Quick Start

### Train the Model

```python
from trading_system import TradingSignalSystem, DataLoader

# Download SPY data (S&P 500 ETF)
df = DataLoader.from_yfinance(symbol='SPY', period='max', interval='1d')

# Create and train the system
system = TradingSignalSystem(
    account_size=100000,    # Your account size
    max_loss_pct=2.0,       # Max 2% loss per trade
    cooldown_hours=4,       # Wait 4 hours between trades
    position_size_pct=10    # Max 10% of account per position
)

# Train on historical data
# Predicts 0.5%+ gain in next 3 days
system.train(df, forward_periods=3, threshold=0.005)
```

### Get Live Signals

```python
# Get the latest 60+ bars of data
recent_data = DataLoader.from_yfinance('SPY', period='1mo', interval='1h')

# Generate a signal
signal = system.generate_signal(recent_data)

print(signal)
# Output:
# {
#     'signal': 'BUY',
#     'probability': 0.72,
#     'reason': 'Model confidence 72%, RSI=45',
#     'current_price': 450.25,
#     'stop_loss': 445.10,
#     'shares': 150,
#     'position_value': 67537.50,
#     'risk_dollars': 2000.00
# }
```
---

## Understanding the Output

### Training Output

When you train the model, you'll see something like:

```
Prediction: SPY will rise 0.5%+ in next 3 bars
Test period: 2024-01-15 to 2024-06-15

==================================================
ALPHA MODEL ONLY
==================================================
BUY signals: 145 | Wins: 82 | Losses: 63
Win rate: 56.6%
Min R/R: 0.8:1

==================================================
ALPHA + RISK FILTER (reject high drawdown)
==================================================
BUY signals: 98 | Wins: 61 | Losses: 37
Win rate: 62.2%
Min R/R: 0.6:1

==================================================
ALPHA + RISK + REGIME (only trending markets)
==================================================
BUY signals: 52 | Wins: 35 | Losses: 17
Win rate: 67.3%
Min R/R: 0.5:1
```

**What this means:**
- Alpha only: Takes 145 trades, wins 56.6% of the time
- With risk filter: Fewer trades (98), but wins more often (62.2%)
- With trend filter: Even fewer trades (52), but highest win rate (67.3%)


## Configuration Options

### Training Parameters

```python
system.train(
    df,                       # Your price data
    forward_periods=3,        # Predict N bars ahead (1-10)
    threshold=0.005,          # Min return to count as "win" (0.5% = 0.005)
    val_size=0.15,           # 15% of data for validation
    test_size=0.15           # 15% of data for testing
)
```

---

## How the Indicators Work

### RSI (Relative Strength Index)
- **Range**: 0 to 100
- **Below 30**: Stock might be oversold (could bounce up)
- **Above 70**: Stock might be overbought (could drop)

### MACD (Moving Average Convergence Divergence)
- Compares short-term vs long-term momentum
- When MACD crosses above signal line: bullish
- When MACD crosses below signal line: bearish

### Bollinger Bands
- Shows if price is stretched above or below normal
- **At lower band**: might bounce up
- **At upper band**: might pull back

### ADX (Average Directional Index)
- **Above 25**: Market is trending (good for this strategy)
- **Below 25**: Market is choppy (strategy stays out)

### VIX (Volatility Index)
- **Below 15**: Market is calm
- **15-25**: Normal volatility
- **Above 25**: Market is fearful (high risk)

---

## File Structure

```
XGBoost_trade_signal/
├── trading_system.py      # Main code
├── README.md              # This file
├── venv/                  # Python environment
│   └── lib/
│       └── python3.12/
│           └── site-packages/
│               ├── lightgbm/    # ML library
│               ├── pandas/      # Data handling
│               ├── numpy/       # Math operations
│               ├── sklearn/     # ML utilities
│               └── yfinance/    # Stock data
```

---


## Tips for Best Results

### ✅ Do This

- **Use enough data**: At least 2 years of daily data for training
- **Match timeframes**: Train on hourly data if trading hourly
- **Start small**: Test with paper trading first
- **Check multiple stocks**: What works for SPY might not work for other stocks

### ❌ Avoid This

- **Overfitting**: Don't tweak parameters until backtest looks perfect
- **Ignoring costs**: Real trading has commissions and slippage
- **Trading every signal**: Higher threshold = fewer but better trades
- **Large positions**: Keep position size under 10% of account

---

## Backtesting Results Explained

The walk-forward backtest splits your data into chunks and tests like this:

```
|------ Train 1 ------|Test 1|
|-------- Train 2 ---------|Test 2|
|---------- Train 3 -----------|Test 3|
```

This prevents "peeking" at future data and gives realistic results.

### Key Metrics

| Metric | Good Value | Great Value |
|--------|------------|-------------|
| Win Rate | 55%+ | 60%+ |
| Sharpe Ratio | 1.0+ | 2.0+ |
| Avg Trade P&L | Positive | >0.5% |

---

## Troubleshooting

### "No positive samples in training data"
- Your threshold is too high
- Try lowering `threshold` from 0.005 to 0.003

### "Insufficient data"
- Need at least 60 bars of data
- Load more history with `period='3mo'`

---


## License

This project is for educational purposes. Use at your own risk. Past performance doesn't guarantee future results.

---

## Contributing

Feel free to:
- Report bugs
- Suggest new features
- Add new indicators
- Improve documentation

---

*Built with Python, LightGBM, and a lot of backtesting* 📊


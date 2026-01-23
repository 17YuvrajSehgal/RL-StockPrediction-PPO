# Swing Trading Package

A professional, production-ready reinforcement learning environment for swing trading with continuous action spaces, realistic margin constraints, and comprehensive feature engineering.

## Features

- **Gymnasium-Compatible**: Full integration with modern RL frameworks (Stable Baselines3, RLlib, etc.)
- **Continuous Actions**: Target portfolio weights from -100% (short) to +100% (long)
- **Realistic Execution**: No look-ahead bias - actions execute at next bar's open price
- **Margin Constraints**: Cash-only mode with short-sale haircuts prevents unrealistic leverage
- **Transaction Costs**: Configurable fees and slippage
- **Feature Engineering**: Technical indicators computed from OHLCV data
- **Type-Safe**: Comprehensive type hints throughout
- **Well-Tested**: Unit tests for all core components
- **Flexible Configuration**: Factory presets for different trading strategies

## Installation

```bash
# From the project root
pip install -e .
```

## Quick Start

```python
import pandas as pd
from swing_trading import SwingTradingEnv, SwingTradingConfig

# Load your OHLCV data
df = pd.read_csv("AAPL.csv", parse_dates=['Date'], index_col='Date')

# Create environment with default configuration
config = SwingTradingConfig()
env = SwingTradingEnv(df, config)

# Use with RL frameworks
obs, info = env.reset()
action = 0.5  # Target 50% long position
obs, reward, terminated, truncated, info = env.step(action)
```

## Configuration Presets

The package includes three factory presets for common use cases:

### Conservative Trading
```python
from swing_trading import SwingTradingConfig

# Risk-averse settings: no shorts, higher costs, longer lookback
config = SwingTradingConfig.conservative()
env = SwingTradingEnv(df, config)
```

### Aggressive Trading
```python
# Active trading: shorts enabled, lower costs, shorter lookback
config = SwingTradingConfig.aggressive()
env = SwingTradingEnv(df, config)
```

### Backtesting
```python
# Minimal costs, deterministic episodes, long horizons
config = SwingTradingConfig.backtest()
env = SwingTradingEnv(df, config)
```

## Training with Stable Baselines3

```python
from stable_baselines3 import PPO
from swing_trading import SwingTradingEnv, SwingTradingConfig

# Load data
df = pd.read_csv("AAPL.csv", parse_dates=['Date'], index_col='Date')

# Create environment
config = SwingTradingConfig()
env = SwingTradingEnv(df, config)

# Train PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

# Evaluate
obs, info = env.reset()
for _ in range(252):  # One trading year
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
    env.render()
```

## Custom Configuration

```python
from swing_trading import (
    SwingTradingConfig,
    TradingConfig,
    EnvironmentConfig,
    MarginConfig,
)

# Customize specific settings
config = SwingTradingConfig(
    trading=TradingConfig(
        fee_rate=0.0001,  # 1 bp
        slippage_rate=0.0001,  # 1 bp
        allow_short=False,  # Long-only
        max_abs_weight=0.8,  # Max 80% exposure
    ),
    environment=EnvironmentConfig(
        lookback=60,  # 60-day window
        episode_length=126,  # ~6 months
        random_start=True,  # Randomize episode starts
    ),
    margin=MarginConfig(
        short_proceeds_haircut=0.5,  # 50% haircut on short proceeds
    ),
)

env = SwingTradingEnv(df, config)
```

## Package Structure

```
swing_trading/
├── __init__.py          # Main exports
├── config.py            # Configuration dataclasses
├── margin.py            # Margin model and constraints
├── portfolio.py         # Portfolio state management
├── features.py          # Feature engineering
├── execution.py         # Trade execution engine
├── environment.py       # Gymnasium environment
└── utils.py             # Utility functions
```

## Core Components

### Configuration (`config.py`)
- `MarginConfig`: Margin model parameters
- `TradingConfig`: Trading costs and constraints
- `EnvironmentConfig`: Episode and observation settings
- `SwingTradingConfig`: Master configuration

### Margin Model (`margin.py`)
- `MarginModel`: Enforces cash-only and short haircut constraints
- Uses bisection search for optimal feasible position sizing
- Prevents unrealistic "infinite cash" from shorting

### Portfolio State (`portfolio.py`)
- `PortfolioState`: Immutable portfolio snapshot
- Tracks equity, cash, shares, entry price, realized/unrealized PnL
- Built-in validation for accounting invariants

### Feature Engineering (`features.py`)
- `FeatureEngineer`: Transforms OHLCV → ML features
- Returns, volatility, moving averages, volume metrics
- Extensible: easy to add custom indicators

### Execution Engine (`execution.py`)
- `ExecutionEngine`: Handles trade execution and costs
- `TradeResult`: Detailed trade records
- Integrates with margin model for constraint enforcement

### Environment (`environment.py`)
- `SwingTradingEnv`: Main Gymnasium environment
- Continuous action space: [-1, 1]
- Observation: (lookback, n_features)
- Reward: log return - turnover penalty

### Utilities (`utils.py`)
- Data validation
- Performance metrics (Sharpe, Sortino, Calmar, max drawdown)
- Return calculations

## Observation Space

The environment provides a 2D observation of shape `(lookback, n_features)`:

**Market Features** (always included):
- `ret_1`: 1-day log return
- `range_pct`: (high - low) / close
- `oc_ret`: Open-to-close return
- `ma{N}_dist`: Distance to N-day moving average
- `vol{N}`: N-day return volatility
- `vol_z{N}`: N-day volume z-score

**Portfolio Features** (optional, enabled by default):
- Normalized cash position
- Share count
- Entry price
- Unrealized PnL (%)
- Realized PnL (%)
- Previous portfolio weight

## Action Space

Continuous action space: `Box(low=-1.0, high=1.0, shape=(1,))`

- `action = 0.0`: Flat (no position)
- `action = 1.0`: 100% long
- `action = -1.0`: 100% short (if allowed)
- `action = 0.5`: 50% long

## Reward Function

```python
reward = log_return - turnover_penalty * turnover
```

Where:
- `log_return = log(equity_t / equity_{t-1})`
- `turnover = traded_notional / equity_{t-1}`

## Design Principles

### No Look-Ahead Bias
- Observation at time `t` uses data up to time `t`
- Action executes at `open(t+1)`, not `close(t)`
- Prevents unrealistic "future knowledge" trading

### Realistic Constraints
- Cash-only mode enforces no margin borrowing for longs
- Short proceeds are haircutted (only portion is usable)
- Transaction costs applied to all trades

### Immutable State
- Portfolio state is immutable for easier debugging
- State transitions return new state objects
- Validation ensures accounting invariants

### Professional Code Quality
- Comprehensive type hints
- Detailed docstrings with examples
- Input validation throughout
- Separation of concerns (OOP design)

## Migration from Notebook

If you're migrating from the `Swing-Trade.ipynb` notebook:

```python
# Old notebook code
config = EnvConfig(...)
env = SingleStockSwingEnv(df, config, FeatureEngineer(df))

# New package code
from swing_trading import SwingTradingEnv, SwingTradingConfig

config = SwingTradingConfig(...)
env = SwingTradingEnv(df, config)
```

Key differences:
- Configuration is now split into logical components
- Margin model is automatically integrated
- Feature engineering is handled internally
- Cleaner API with fewer required arguments

## Examples

See the `examples/` directory for:
- Basic usage
- Training with PPO
- Backtesting
- Custom feature engineering
- Performance analysis

## Testing

```bash
# Run all tests
pytest swing_trading/tests/ -v

# With coverage
pytest swing_trading/tests/ --cov=swing_trading --cov-report=html
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Add tests for new features
2. Update documentation
3. Follow existing code style
4. Add type hints

## Citation

If you use this package in your research, please cite:

```bibtex
@software{swing_trading,
  title = {Swing Trading: A Professional RL Environment for Stock Trading},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/RL-StockPrediction-PPO}
}
```

# Daily Trading System

A production-grade daily model retraining and signal generation system for RL-based stock trading.

## Overview

This package provides:
- **Daily Trainer**: Supports both full retraining and fine-tuning modes
- **Signal Generator**: Generates actionable BUY/SELL/HOLD signals with confidence scores
- **Walk-Forward Validator**: Rigorous strategy validation using temporal cross-validation
- **CLI Pipeline**: Easy-to-use command line interface

## Quick Start

### Generate Trading Signal

```bash
# For next trading day
python daily_pipeline.py signal --ticker AAPL

# For specific date
python daily_pipeline.py signal --ticker AAPL --for-date 2026-01-27
```

Example output:
```
============================================================
TRADING SIGNAL: AAPL
============================================================
Signal Date:    2026-01-27 (next trading day)
Generated:      2026-01-24 12:30:00
------------------------------------------------------------
Signal:         BUY
Confidence:     72%
Target Weight:  +65%
Current Price:  $185.50
------------------------------------------------------------
Reasoning:      Model predicts strong bullish signal with 65% 
                long target. Confidence: 72%.
============================================================
```

### Walk-Forward Validation

Test your strategy by training on past data and validating on the next day:

```bash
# Last 30 trading days
python daily_pipeline.py validate --ticker AAPL --days 30

# Specific date range
python daily_pipeline.py validate --ticker AAPL --start 2025-10-01 --end 2025-12-31

# With fine-tuning mode
python daily_pipeline.py validate --ticker AAPL --days 30 --mode fine_tune
```

### Compare Training Modes

Research which mode works better for your use case:

```bash
python daily_pipeline.py compare --ticker AAPL --days 10
```

Output:
```
======================================================================
TRAINING MODE COMPARISON
======================================================================
Metric                          Full Train            Fine-Tune
----------------------------------------------------------------------
Accuracy                              55.0%                60.0%
Cumulative Return                    +3.45%               +4.12%
Sharpe Ratio                          1.23                 1.45
Max Drawdown                         -2.10%               -1.85%
Training Time (hrs)                   5.2                  1.1
======================================================================
```

### Train Model

Train a model for a specific date:

```bash
# Full training
python daily_pipeline.py train --ticker AAPL --date 2026-01-24 --mode full

# Fine-tuning
python daily_pipeline.py train --ticker AAPL --date 2026-01-24 --mode fine_tune --base-model models/AAPL/2026-01-23/model.zip
```

## Package Structure

```
daily_trading/
├── __init__.py          # Package exports (lazy loading for ML modules)
├── config.py            # Configuration dataclasses
│   ├── TrainingMode     # FULL or FINE_TUNE
│   ├── WindowType       # ROLLING or EXPANDING
│   ├── TrainingWindowConfig
│   ├── DailyTrainingConfig
│   ├── SignalConfig     # BUY/SELL thresholds
│   ├── WalkForwardConfig
│   └── PathConfig
├── utils.py             # Utilities
│   ├── TradingCalendar  # Market day handling
│   └── DataLoader       # OHLCV loading with date filtering
├── trainer.py           # Model training
│   ├── DailyTrainer     # Main trainer class
│   └── TrainResult      # Training outcome
├── signals.py           # Signal generation
│   ├── TradingSignal    # Signal dataclass
│   └── SignalGenerator  # Signal generation class
└── walk_forward.py      # Validation
    ├── DayResult        # Single day result
    ├── WalkForwardResult # Aggregate results
    └── WalkForwardValidator
```

## Configuration

### Training Window (Adjustable for Market Conditions)

```python
from daily_trading.config import TrainingWindowConfig

# Standard 2-year rolling window
window = TrainingWindowConfig.rolling(504)

# Volatile market - shorter 6-month window
window = TrainingWindowConfig.volatile_market()

# Stable market - longer 3-year window
window = TrainingWindowConfig.stable_market()

# Expanding window (all history)
window = TrainingWindowConfig.expanding()
```

### Signal Thresholds

```python
from daily_trading.config import SignalConfig

config = SignalConfig(
    buy_threshold=0.3,   # Weight > 0.3 → BUY
    sell_threshold=-0.3, # Weight < -0.3 → SELL
    # -0.3 ≤ Weight ≤ 0.3 → HOLD
)
```

### Training Mode

```python
from daily_trading.config import DailyTrainingConfig, TrainingMode

# Full training (30-60 min, more robust)
config = DailyTrainingConfig.full_train()

# Fine-tuning (5-15 min, faster updates)
config = DailyTrainingConfig.fine_tune()

# Custom
config = DailyTrainingConfig(
    mode=TrainingMode.FINE_TUNE,
    timesteps_fine_tune=100_000,
    learning_rate_fine_tune=3e-5,
    window=TrainingWindowConfig.volatile_market(),
)
```

## Python API

### Generate Signal

```python
from datetime import date
from daily_trading import SignalGenerator
from daily_trading.config import SignalConfig, PathConfig

# Initialize
config = SignalConfig()
paths = PathConfig()
generator = SignalGenerator(config, paths)

# Generate signal for Monday
signal = generator.generate(
    ticker="AAPL",
    for_date=date(2026, 1, 27),
)

print(signal.signal)      # "BUY"
print(signal.confidence)  # 0.72
print(signal.reasoning)   # Human-readable explanation
```

### Walk-Forward Validation

```python
from datetime import date
from daily_trading import WalkForwardValidator
from daily_trading.config import WalkForwardConfig, PathConfig

# Initialize
config = WalkForwardConfig()
paths = PathConfig()
validator = WalkForwardValidator(config, paths)

# Run validation
result = validator.validate(
    ticker="AAPL",
    start_date=date(2025, 10, 1),
    end_date=date(2025, 12, 31),
)

# Analyze results
print(f"Accuracy: {result.accuracy:.1%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Strategy Return: {result.cumulative_return:+.2%}")
print(f"Buy-and-Hold: {result.buy_hold_return:+.2%}")
print(f"Excess Return: {result.excess_return:+.2%}")

# Export to DataFrame for analysis
df = result.to_dataframe()
```

### Daily Training

```python
from datetime import date
from daily_trading import DailyTrainer
from daily_trading.config import DailyTrainingConfig, TrainingMode, PathConfig

# Initialize
config = DailyTrainingConfig.full_train()
paths = PathConfig()
trainer = DailyTrainer(config, paths)

# Train model
result = trainer.train(
    ticker="AAPL",
    as_of_date=date(2026, 1, 24),
    mode=TrainingMode.FULL,
)

if result.success:
    print(f"Model saved: {result.model_path}")
    print(f"Training time: {result.training_time_seconds:.1f}s")
else:
    print(f"Training failed: {result.error}")
```

## Output Directories

After running the pipeline, outputs are organized as:

```
RL-StockPrediction-PPO/
├── daily_models/           # Date-versioned models
│   └── AAPL/
│       ├── 2026-01-24/
│       │   ├── model.zip
│       │   ├── vecnormalize.pkl
│       │   ├── metadata.json
│       │   └── train_result.json
│       └── ...
├── signals/                # Generated signals
│   └── AAPL/
│       ├── 2026-01-27.json
│       └── ...
├── walk_forward_results/   # Validation results
│   └── AAPL_2025-10-01_to_2025-12-31.json
└── daily_trading_logs/     # TensorBoard logs
```

## Key Design Principles

### 1. Temporal Integrity
No future data leakage - training data is strictly filtered by `as_of_date`.

### 2. Train-Test Consistency
Same environment, margin constraints, and execution logic in both training and validation.

### 3. Configurable Windows
Adjust training window based on market conditions:
- Volatile markets → shorter window (6 months)
- Stable markets → longer window (3 years)

### 4. Mode Comparison
Research which training mode works better for your use case before production deployment.

## Requirements

- Python 3.9+
- stable-baselines3
- gymnasium
- pandas
- numpy
- torch

## CLI Reference

```
daily_pipeline.py <command> [options]

Commands:
  signal      Generate trading signal for next trading day
  validate    Run walk-forward validation
  compare     Compare full training vs fine-tuning
  train       Train model for specific date
  quick-test  Quick 5-day validation test

Global Options:
  --data-dir  Directory containing market data (default: yf_data)
  --verbose   Verbose output
  --debug     Debug logging

Use 'python daily_pipeline.py <command> --help' for command-specific options.
```

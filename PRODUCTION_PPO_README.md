# Production PPO Trading System - Quick Start Guide

## Overview

Enterprise-grade PPO training system for financial institutions with GPU acceleration, comprehensive monitoring, and risk management.

## Installation

```bash
# Install in development mode
cd c:\workplace\RL-StockPrediction-PPO
pip install -e .
```

## Quick Start

### Basic Training

```bash
# Train with default settings (production mode, institutional risk limits)
python train_production_ppo.py --ticker AAPL

# Research mode (faster iteration, permissive risk limits)
python train_production_ppo.py --ticker MSFT --mode research --timesteps 100000

# Specify GPU
python train_production_ppo.py --ticker AAPL --gpu 0
```

### Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir ./runs

# Open browser to http://localhost:6006
```

## Key Features

✅ **GPU Acceleration** - Automatic detection with CPU fallback  
✅ **TensorBoard Monitoring** - Real-time training metrics  
✅ **Financial Metrics** - Sharpe ratio, max drawdown, win rate  
✅ **Risk Management** - Position limits, circuit breakers  
✅ **Model Checkpointing** - Automatic saving with versioning  
✅ **Production Ready** - Enterprise-grade code quality  

## Configuration Modes

### Production Mode (Conservative)
- Institutional risk limits
- Lower learning rate
- Larger batch sizes
- Strict drawdown limits (15%)

### Research Mode (Aggressive)
- Permissive risk limits
- Higher learning rate
- Faster iteration
- Relaxed constraints

## Example Usage

```python
from swing_trading import SwingTradingEnv, SwingTradingConfig
from trading_system import PPOTrainer, TrainingConfig, RiskConfig

# Load data
df = pd.read_csv("data/AAPL.csv")

# Create environment
env = SwingTradingEnv(df, SwingTradingConfig.conservative())

# Configure training
config = TrainingConfig.production("aapl_v1")
risk_config = RiskConfig.institutional()

# Train
trainer = PPOTrainer(env, config, risk_config)
trainer.train()

# Save
trainer.save("models/aapl_final")
```

## Directory Structure

```
trading_system/
├── config/              # Configuration management
├── training/            # Training infrastructure
├── evaluation/          # Backtesting and metrics
└── deployment/          # Production inference
```

## Next Steps

1. Run training: `python train_production_ppo.py --ticker AAPL`
2. Monitor with TensorBoard
3. Evaluate results
4. Deploy to production

For detailed documentation, see implementation_plan.md

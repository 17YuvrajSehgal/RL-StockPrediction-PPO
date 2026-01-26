"""
Daily Trading System for RL Stock Prediction.

A production-grade daily model retraining and signal generation system with:
- Configurable training modes (full retrain vs fine-tuning)
- Walk-forward validation framework
- Professional signal generation (BUY/SELL/HOLD)
- Adjustable training windows for different market conditions

Package Structure:
    - config: Configuration management
    - trainer: Daily model training
    - signals: Trading signal generation
    - walk_forward: Walk-forward validation
    - utils: Shared utilities

Example:
    >>> from daily_trading import DailyTrainer, SignalGenerator, WalkForwardValidator
    >>> from daily_trading.config import DailyTradingConfig
    >>> 
    >>> # Generate trading signal for Monday
    >>> config = DailyTradingConfig.production()
    >>> generator = SignalGenerator(config)
    >>> signal = generator.generate("AAPL", for_date=date(2026, 1, 27))
    >>> print(signal)
    TradingSignal(ticker='AAPL', signal='BUY', confidence=0.72, ...)

Requirements:
    - stable_baselines3
    - gymnasium
    - pandas
    - numpy
    - torch
"""

# Configuration - always available (no ML dependencies)
from daily_trading.config import (
    # Enums
    TrainingMode,
    WindowType,
    # Configuration dataclasses
    TrainingWindowConfig,
    DailyTrainingConfig,
    SignalConfig,
    WalkForwardConfig,
    PathConfig,
    DailyTradingConfig,
)

# Utilities - always available
from daily_trading.utils import (
    TradingCalendar,
    DataLoader,
)

__version__ = "1.0.0"
__author__ = "RL Stock Prediction Team"

# Lazy imports for ML-dependent modules to avoid ImportError
# when running in environments without full dependencies
def __getattr__(name):
    """Lazy import for ML-dependent modules."""
    if name in ("DailyTrainer", "TrainResult"):
        from daily_trading.trainer import DailyTrainer, TrainResult
        return DailyTrainer if name == "DailyTrainer" else TrainResult
    elif name in ("TradingSignal", "SignalGenerator"):
        from daily_trading.signals import TradingSignal, SignalGenerator
        return TradingSignal if name == "TradingSignal" else SignalGenerator
    elif name in ("DayResult", "WalkForwardResult", "WalkForwardValidator"):
        from daily_trading.walk_forward import DayResult, WalkForwardResult, WalkForwardValidator
        if name == "DayResult":
            return DayResult
        elif name == "WalkForwardResult":
            return WalkForwardResult
        else:
            return WalkForwardValidator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Enums
    "TrainingMode",
    "WindowType",
    # Configuration
    "TrainingWindowConfig",
    "DailyTrainingConfig",
    "SignalConfig",
    "WalkForwardConfig",
    "PathConfig",
    "DailyTradingConfig",
    # Trainer (lazy)
    "DailyTrainer",
    "TrainResult",
    # Signals (lazy)
    "TradingSignal",
    "SignalGenerator",
    # Walk-forward (lazy)
    "DayResult",
    "WalkForwardResult",
    "WalkForwardValidator",
    # Utilities
    "TradingCalendar",
    "DataLoader",
]

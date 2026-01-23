"""
Swing Trading Package

A professional, production-ready reinforcement learning environment for swing trading
with continuous action spaces, realistic margin constraints, and comprehensive feature engineering.

Key Components:
    - SwingTradingEnv: Gymnasium-compatible environment
    - MarginModel: Realistic margin and position constraints
    - FeatureEngineer: Technical indicator computation
    - ExecutionEngine: Trade execution and cost modeling
    - Portfolio state tracking and PnL accounting

Example:
    >>> from swing_trading import SwingTradingEnv, SwingTradingConfig
    >>> import pandas as pd
    >>> 
    >>> # Load your OHLCV data
    >>> df = pd.read_csv("AAPL.csv")
    >>> 
    >>> # Create environment with default config
    >>> config = SwingTradingConfig()
    >>> env = SwingTradingEnv(df, config)
    >>> 
    >>> # Use with RL frameworks
    >>> obs, info = env.reset()
    >>> action = env.action_space.sample()
    >>> obs, reward, terminated, truncated, info = env.step(action)
"""

from swing_trading.config import (
    MarginConfig,
    TradingConfig,
    EnvironmentConfig,
    SwingTradingConfig,
)

from swing_trading.margin import MarginModel

from swing_trading.portfolio import PortfolioState

from swing_trading.features import FeatureEngineer

from swing_trading.execution import ExecutionEngine, TradeResult

from swing_trading.environment import SwingTradingEnv

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "MarginConfig",
    "TradingConfig",
    "EnvironmentConfig",
    "SwingTradingConfig",
    # Core components
    "MarginModel",
    "PortfolioState",
    "FeatureEngineer",
    "ExecutionEngine",
    "TradeResult",
    # Environment
    "SwingTradingEnv",
]

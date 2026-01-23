"""
Production-Ready PPO Trading System

Enterprise-grade reinforcement learning system for financial trading with:
- GPU-accelerated training
- Comprehensive monitoring and logging
- Risk management and compliance
- Production deployment infrastructure
"""

from trading_system.config import (
    PPOConfig,
    TrainingConfig,
    DeviceConfig,
    LoggingConfig,
    RiskConfig,
)

from trading_system.training import PPOTrainer

from trading_system.evaluation import Backtester, FinancialMetrics

from trading_system.deployment import InferenceEngine

__version__ = "1.0.0"

__all__ = [
    # Configuration
    "PPOConfig",
    "TrainingConfig",
    "DeviceConfig",
    "LoggingConfig",
    "RiskConfig",
    # Training
    "PPOTrainer",
    # Evaluation
    "Backtester",
    "FinancialMetrics",
    # Deployment
    "InferenceEngine",
]

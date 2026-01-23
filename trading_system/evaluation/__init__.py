"""Evaluation package for backtesting and metrics."""

from trading_system.evaluation.metrics import FinancialMetrics
from trading_system.evaluation.backtester import Backtester

__all__ = ["FinancialMetrics", "Backtester"]

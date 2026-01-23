"""Financial metrics computation."""

import numpy as np


class FinancialMetrics:
    """Compute financial performance metrics."""
    
    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Compute annualized Sharpe ratio."""
        if len(returns) == 0:
            return 0.0
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 0.0
        return (mean_return - risk_free_rate) / std_return * np.sqrt(252)
    
    @staticmethod
    def max_drawdown(equity_curve: np.ndarray) -> float:
        """Compute maximum drawdown."""
        if len(equity_curve) == 0:
            return 0.0
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        return float(np.min(drawdown))

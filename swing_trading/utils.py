"""
Utility functions for swing trading package.

This module provides helper functions for data validation, return calculations,
and performance metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_ohlcv_data(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Validate OHLCV DataFrame for required columns and data quality.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    
    Example:
        >>> import pandas as pd
        >>> df = pd.read_csv("data.csv")
        >>> valid, errors = validate_ohlcv_data(df)
        >>> if not valid:
        ...     for error in errors:
        ...         print(f"Error: {error}")
    """
    errors = []
    
    # Check required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required if col not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")
        return False, errors
    
    # Check for NaN values
    for col in required:
        if df[col].isna().any():
            errors.append(f"Column '{col}' contains NaN values")
    
    # Check for non-positive prices
    for col in ['open', 'high', 'low', 'close']:
        if (df[col] <= 0).any():
            errors.append(f"Column '{col}' contains non-positive values")
    
    # Check high >= low
    if (df['high'] < df['low']).any():
        errors.append("Some rows have high < low")
    
    # Check if index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append("Index is not DatetimeIndex")
    
    # Check for duplicate dates
    if df.index.duplicated().any():
        errors.append("Index contains duplicate dates")
    
    return len(errors) == 0, errors


def compute_returns(
    prices: pd.Series | np.ndarray,
    log: bool = True,
) -> pd.Series | np.ndarray:
    """
    Compute returns from price series.
    
    Args:
        prices: Price series or array
        log: If True, compute log returns; else simple returns
    
    Returns:
        Return series or array
    
    Example:
        >>> prices = pd.Series([100, 105, 103, 108])
        >>> returns = compute_returns(prices)
        >>> print(returns)
    """
    if isinstance(prices, pd.Series):
        if log:
            return np.log(prices).diff()
        else:
            return prices.pct_change()
    else:
        if log:
            return np.diff(np.log(prices))
        else:
            return np.diff(prices) / prices[:-1]


def compute_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute annualized Sharpe ratio.
    
    Args:
        returns: Return series or array
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year for annualization
    
    Returns:
        Annualized Sharpe ratio
    
    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02])
        >>> sharpe = compute_sharpe_ratio(returns)
        >>> print(f"Sharpe ratio: {sharpe:.2f}")
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna().values
    
    if len(returns) == 0:
        return 0.0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    
    if std_return == 0:
        return 0.0
    
    # Annualize
    annual_return = mean_return * periods_per_year
    annual_std = std_return * np.sqrt(periods_per_year)
    
    sharpe = (annual_return - risk_free_rate) / annual_std
    return float(sharpe)


def compute_max_drawdown(
    equity_curve: pd.Series | np.ndarray,
) -> tuple[float, int, int]:
    """
    Compute maximum drawdown and its location.
    
    Args:
        equity_curve: Equity time series
    
    Returns:
        Tuple of (max_drawdown, peak_idx, trough_idx)
        max_drawdown is negative (e.g., -0.15 for 15% drawdown)
    
    Example:
        >>> equity = pd.Series([100, 110, 105, 95, 100, 120])
        >>> mdd, peak, trough = compute_max_drawdown(equity)
        >>> print(f"Max drawdown: {mdd:.2%} from idx {peak} to {trough}")
    """
    if isinstance(equity_curve, pd.Series):
        equity = equity_curve.values
    else:
        equity = equity_curve
    
    if len(equity) == 0:
        return 0.0, 0, 0
    
    # Compute running maximum
    running_max = np.maximum.accumulate(equity)
    
    # Compute drawdown
    drawdown = (equity - running_max) / running_max
    
    # Find maximum drawdown
    max_dd_idx = np.argmin(drawdown)
    max_dd = drawdown[max_dd_idx]
    
    # Find peak (last time we were at running max before trough)
    peak_idx = np.where(running_max[:max_dd_idx + 1] == running_max[max_dd_idx])[0][-1]
    
    return float(max_dd), int(peak_idx), int(max_dd_idx)


def compute_calmar_ratio(
    returns: pd.Series | np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """
    Compute Calmar ratio (annualized return / max drawdown).
    
    Args:
        returns: Return series
        periods_per_year: Periods per year for annualization
    
    Returns:
        Calmar ratio
    
    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02])
        >>> calmar = compute_calmar_ratio(returns)
        >>> print(f"Calmar ratio: {calmar:.2f}")
    """
    if isinstance(returns, pd.Series):
        returns_arr = returns.dropna().values
    else:
        returns_arr = returns
    
    if len(returns_arr) == 0:
        return 0.0
    
    # Compute equity curve
    equity = np.cumprod(1 + returns_arr)
    
    # Compute max drawdown
    max_dd, _, _ = compute_max_drawdown(equity)
    
    if max_dd == 0:
        return 0.0
    
    # Compute annualized return
    total_return = equity[-1] - 1
    n_periods = len(returns_arr)
    annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
    
    calmar = annual_return / abs(max_dd)
    return float(calmar)


def compute_sortino_ratio(
    returns: pd.Series | np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute annualized Sortino ratio (uses downside deviation).
    
    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Periods per year for annualization
    
    Returns:
        Annualized Sortino ratio
    
    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02])
        >>> sortino = compute_sortino_ratio(returns)
        >>> print(f"Sortino ratio: {sortino:.2f}")
    """
    if isinstance(returns, pd.Series):
        returns_arr = returns.dropna().values
    else:
        returns_arr = returns
    
    if len(returns_arr) == 0:
        return 0.0
    
    mean_return = np.mean(returns_arr)
    
    # Compute downside deviation (only negative returns)
    downside_returns = returns_arr[returns_arr < 0]
    if len(downside_returns) == 0:
        downside_std = 0.0
    else:
        downside_std = np.std(downside_returns, ddof=1)
    
    if downside_std == 0:
        return 0.0
    
    # Annualize
    annual_return = mean_return * periods_per_year
    annual_downside_std = downside_std * np.sqrt(periods_per_year)
    
    sortino = (annual_return - risk_free_rate) / annual_downside_std
    return float(sortino)

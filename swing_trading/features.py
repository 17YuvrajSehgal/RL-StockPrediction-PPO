"""
Feature engineering for market data.

This module provides technical indicator computation and feature transformation
for machine learning models.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Transforms OHLCV data into ML-ready features.
    
    Computes technical indicators including returns, moving averages,
    volatility, and volume metrics. All features are designed to be
    stationary and normalized for better learning stability.
    
    Attributes:
        df: DataFrame with OHLCV data
        features: Computed feature DataFrame
    
    Example:
        >>> import pandas as pd
        >>> df = pd.read_csv("AAPL.csv", parse_dates=['Date'], index_col='Date')
        >>> engineer = FeatureEngineer(df)
        >>> features = engineer.build()
        >>> print(features.columns.tolist())
    """
    
    def __init__(self, df_daily: pd.DataFrame) -> None:
        """
        Initialize feature engineer.
        
        Args:
            df_daily: DataFrame with columns: open, high, low, close, volume
                     Index should be datetime for proper time-series handling
        
        Raises:
            ValueError: If required columns are missing
        """
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in df_daily.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        self.df = df_daily.copy()
        self.features: Optional[pd.DataFrame] = None
    
    def build(
        self,
        ma_windows: tuple[int, ...] = (20, 50),
        vol_window: int = 20,
        vol_z_window: int = 20,
    ) -> pd.DataFrame:
        """
        Build all features from OHLCV data.
        
        Features computed:
        - ret_1: 1-day log return
        - range_pct: (high - low) / close
        - oc_ret: open-to-close return
        - ma{N}_dist: distance to N-day moving average
        - vol{N}: N-day return volatility
        - vol_z{N}: N-day volume z-score
        
        Args:
            ma_windows: Tuple of moving average window sizes
            vol_window: Window for volatility calculation
            vol_z_window: Window for volume z-score
        
        Returns:
            DataFrame with computed features (NaN rows dropped)
        
        Example:
            >>> engineer = FeatureEngineer(df)
            >>> features = engineer.build(ma_windows=(10, 20, 50))
            >>> print(f"Generated {len(features.columns)} features")
        """
        df = self.df.copy()
        
        # Basic stationary transforms
        df['ret_1'] = np.log(df['close']).diff()
        df['range_pct'] = (df['high'] - df['low']) / df['close']
        df['oc_ret'] = np.log(df['close'] / df['open'])
        
        # Moving averages and distances
        for window in ma_windows:
            ma_col = f'ma{window}'
            dist_col = f'ma{window}_dist'
            df[ma_col] = df['close'].rolling(window).mean()
            df[dist_col] = (df['close'] - df[ma_col]) / df['close']
        
        # Volatility proxy
        df[f'vol{vol_window}'] = df['ret_1'].rolling(vol_window).std()
        
        # Volume z-score
        v = df['volume']
        v_mean = v.rolling(vol_z_window).mean()
        v_std = v.rolling(vol_z_window).std()
        df[f'vol_z{vol_z_window}'] = (v - v_mean) / (v_std + 1e-12)
        
        # Select feature columns (exclude raw OHLCV and intermediate MAs)
        feature_cols = [
            'ret_1', 'range_pct', 'oc_ret',
            *[f'ma{w}_dist' for w in ma_windows],
            f'vol{vol_window}',
            f'vol_z{vol_z_window}',
        ]
        
        features = df[feature_cols].dropna()
        self.features = features
        
        return features
    
    @staticmethod
    def compute_returns(prices: pd.Series, log: bool = True) -> pd.Series:
        """
        Compute returns from price series.
        
        Args:
            prices: Price series
            log: If True, compute log returns; else simple returns
        
        Returns:
            Return series
        
        Example:
            >>> prices = pd.Series([100, 105, 103, 108])
            >>> returns = FeatureEngineer.compute_returns(prices)
            >>> print(returns)
        """
        if log:
            return np.log(prices).diff()
        else:
            return prices.pct_change()
    
    @staticmethod
    def compute_volatility(
        returns: pd.Series,
        window: int,
        annualize: bool = False,
        periods_per_year: int = 252,
    ) -> pd.Series:
        """
        Compute rolling volatility.
        
        Args:
            returns: Return series
            window: Rolling window size
            annualize: If True, annualize volatility
            periods_per_year: Periods per year for annualization
        
        Returns:
            Volatility series
        
        Example:
            >>> returns = pd.Series([0.01, -0.02, 0.015, -0.01])
            >>> vol = FeatureEngineer.compute_volatility(returns, window=20)
        """
        vol = returns.rolling(window).std()
        if annualize:
            vol = vol * np.sqrt(periods_per_year)
        return vol
    
    @staticmethod
    def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Compute Relative Strength Index (RSI).
        
        Args:
            prices: Price series
            window: RSI window (typically 14)
        
        Returns:
            RSI series (0-100)
        
        Example:
            >>> prices = pd.Series([100, 105, 103, 108, 110])
            >>> rsi = FeatureEngineer.compute_rsi(prices)
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        
        rs = gain / (loss + 1e-12)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def compute_bollinger_bands(
        prices: pd.Series,
        window: int = 20,
        num_std: float = 2.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute Bollinger Bands.
        
        Args:
            prices: Price series
            window: Moving average window
            num_std: Number of standard deviations for bands
        
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        
        Example:
            >>> prices = pd.Series([100, 105, 103, 108, 110])
            >>> upper, middle, lower = FeatureEngineer.compute_bollinger_bands(prices)
        """
        middle = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)
        return upper, middle, lower
    
    @staticmethod
    def compute_macd(
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
        
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        
        Example:
            >>> prices = pd.Series([100, 105, 103, 108, 110])
            >>> macd, signal, hist = FeatureEngineer.compute_macd(prices)
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

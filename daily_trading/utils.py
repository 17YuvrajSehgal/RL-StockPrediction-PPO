"""
Utility functions and classes for daily trading system.

This module provides shared utilities including:
- Trading calendar for date handling
- Data loading with proper date filtering
- Logging utilities
- Common helper functions
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


class TradingCalendar:
    """
    Trading calendar utilities for market date handling.
    
    Handles weekends, holidays, and market schedule for US equities.
    
    Attributes:
        holidays: Set of known market holidays
    
    Example:
        >>> calendar = TradingCalendar()
        >>> next_day = calendar.next_trading_day(date(2026, 1, 24))  # Saturday
        >>> print(next_day)  # 2026-01-26 (Monday)
    """
    
    # Major US market holidays (simplified - production would use full calendar)
    US_HOLIDAYS_2026 = {
        date(2026, 1, 1),   # New Year's Day
        date(2026, 1, 19),  # MLK Day
        date(2026, 2, 16),  # Presidents Day
        date(2026, 4, 3),   # Good Friday
        date(2026, 5, 25),  # Memorial Day
        date(2026, 7, 3),   # Independence Day observed
        date(2026, 9, 7),   # Labor Day
        date(2026, 11, 26), # Thanksgiving
        date(2026, 12, 25), # Christmas
    }
    
    US_HOLIDAYS_2025 = {
        date(2025, 1, 1),   # New Year's Day
        date(2025, 1, 20),  # MLK Day
        date(2025, 2, 17),  # Presidents Day
        date(2025, 4, 18),  # Good Friday
        date(2025, 5, 26),  # Memorial Day
        date(2025, 7, 4),   # Independence Day
        date(2025, 9, 1),   # Labor Day
        date(2025, 11, 27), # Thanksgiving
        date(2025, 12, 25), # Christmas
    }
    
    def __init__(self, additional_holidays: Optional[set[date]] = None):
        """
        Initialize trading calendar.
        
        Args:
            additional_holidays: Extra holidays to include
        """
        self.holidays = self.US_HOLIDAYS_2025 | self.US_HOLIDAYS_2026
        if additional_holidays:
            self.holidays |= additional_holidays
    
    def is_trading_day(self, d: date) -> bool:
        """
        Check if date is a trading day.
        
        Args:
            d: Date to check
        
        Returns:
            True if trading day (weekday and not holiday)
        """
        # Weekend check
        if d.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Holiday check
        if d in self.holidays:
            return False
        
        return True
    
    def next_trading_day(self, d: date, include_current: bool = False) -> date:
        """
        Get the next trading day on or after given date.
        
        Args:
            d: Starting date
            include_current: If True and d is trading day, return d
        
        Returns:
            Next trading day
        """
        if include_current and self.is_trading_day(d):
            return d
        
        current = d + timedelta(days=1)
        while not self.is_trading_day(current):
            current += timedelta(days=1)
            # Safety limit
            if (current - d).days > 14:
                raise ValueError(f"Could not find trading day within 14 days of {d}")
        
        return current
    
    def previous_trading_day(self, d: date, include_current: bool = False) -> date:
        """
        Get the previous trading day on or before given date.
        
        Args:
            d: Starting date
            include_current: If True and d is trading day, return d
        
        Returns:
            Previous trading day
        """
        if include_current and self.is_trading_day(d):
            return d
        
        current = d - timedelta(days=1)
        while not self.is_trading_day(current):
            current -= timedelta(days=1)
            if (d - current).days > 14:
                raise ValueError(f"Could not find trading day within 14 days before {d}")
        
        return current
    
    def trading_days_between(
        self, 
        start_date: date, 
        end_date: date,
        include_start: bool = True,
        include_end: bool = True,
    ) -> List[date]:
        """
        Get list of trading days in date range.
        
        Args:
            start_date: Start of range
            end_date: End of range
            include_start: Include start_date if it's a trading day
            include_end: Include end_date if it's a trading day
        
        Returns:
            List of trading days
        """
        if start_date > end_date:
            return []
        
        trading_days = []
        current = start_date
        
        while current <= end_date:
            if self.is_trading_day(current):
                if current == start_date and not include_start:
                    pass
                elif current == end_date and not include_end:
                    pass
                else:
                    trading_days.append(current)
            current += timedelta(days=1)
        
        return trading_days
    
    def count_trading_days(self, start_date: date, end_date: date) -> int:
        """
        Count trading days between dates (inclusive).
        
        Args:
            start_date: Start of range
            end_date: End of range
        
        Returns:
            Number of trading days
        """
        return len(self.trading_days_between(start_date, end_date))


class DataLoader:
    """
    Data loading utilities with proper date filtering for temporal integrity.
    
    Ensures no future data leakage by strictly filtering by as_of_date.
    
    Attributes:
        data_dir: Directory containing data files
        cache: In-memory cache of loaded data
    
    Example:
        >>> loader = DataLoader(Path("yf_data"))
        >>> df = loader.load_ticker("AAPL", as_of_date=date(2026, 1, 24))
        >>> print(df.index[-1])  # Never after as_of_date
    """
    
    def __init__(self, data_dir: Path):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing CSV/Parquet data files
        """
        self.data_dir = Path(data_dir)
        self._cache: dict[str, pd.DataFrame] = {}
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def load_ticker(
        self,
        ticker: str,
        as_of_date: Optional[date] = None,
        start_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV data for ticker with date filtering.
        
        CRITICAL: When as_of_date is specified, no data after this date
        is included to prevent future data leakage.
        
        Args:
            ticker: Stock ticker symbol
            as_of_date: Cutoff date (no data after this, inclusive)
            start_date: Optional start date filter
        
        Returns:
            DataFrame with columns: open, high, low, close, volume
            Index: DatetimeIndex
        
        Raises:
            FileNotFoundError: If no data file found for ticker
            ValueError: If insufficient data after filtering
        """
        # Load from cache or disk
        if ticker not in self._cache:
            df = self._load_from_disk(ticker)
            self._cache[ticker] = df
        else:
            df = self._cache[ticker].copy()
        
        # Apply date filters
        if start_date is not None:
            df = df[df.index >= pd.Timestamp(start_date)]
        
        if as_of_date is not None:
            # CRITICAL: Include as_of_date (<=) to avoid off-by-one errors
            df = df[df.index <= pd.Timestamp(as_of_date)]
        
        if len(df) == 0:
            raise ValueError(
                f"No data for {ticker} in date range "
                f"{start_date} to {as_of_date}"
            )
        
        self._logger.debug(
            f"Loaded {ticker}: {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}"
        )
        
        return df
    
    def _load_from_disk(self, ticker: str) -> pd.DataFrame:
        """Load data file from disk."""
        # Try CSV files
        csv_files = list(self.data_dir.glob(f"{ticker}*.csv"))
        if csv_files:
            df = pd.read_csv(csv_files[0])
        else:
            # Try Parquet
            parquet_files = list(self.data_dir.glob(f"{ticker}*.parquet"))
            if parquet_files:
                df = pd.read_parquet(parquet_files[0])
            else:
                raise FileNotFoundError(
                    f"No data file found for {ticker} in {self.data_dir}"
                )
        
        # Normalize columns
        df = self._normalize_columns(df, ticker)
        
        return df
    
    def _normalize_columns(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Normalize column names and set index."""
        # Handle yfinance multi-level column format first
        # Columns may look like: ("Date", ""), ("Open", "AAPL"), etc.
        # Or tuples stored as strings: "('Date', '')", "('Open', 'AAPL')"
        
        new_cols = {}
        date_col = None
        
        for col in df.columns:
            # Convert column to string and lowercase for matching
            col_str = str(col).lower()
            
            # Check for yfinance-style tuple columns
            if 'date' in col_str:
                new_cols[col] = 'date'
                date_col = col
            elif 'open' in col_str and 'close' not in col_str:
                new_cols[col] = 'open'
            elif 'high' in col_str:
                new_cols[col] = 'high'
            elif 'low' in col_str:
                new_cols[col] = 'low'
            elif 'close' in col_str and 'adj' not in col_str:
                new_cols[col] = 'close'
            elif 'volume' in col_str:
                new_cols[col] = 'volume'
        
        df = df.rename(columns=new_cols)
        
        # Set date index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        elif df.index.name and 'date' in str(df.index.name).lower():
            df.index = pd.to_datetime(df.index)
        
        # Ensure lowercase columns
        df.columns = [str(c).lower() for c in df.columns]
        
        # Sort by date
        df = df.sort_index()
        
        # Validate required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            # Try to find columns that partially match
            available = list(df.columns)
            raise ValueError(
                f"Missing required columns for {ticker}: {missing}. "
                f"Available columns: {available}"
            )
        
        return df[required]
    
    def get_latest_date(self, ticker: str) -> date:
        """Get the most recent data date for ticker."""
        df = self.load_ticker(ticker)
        return df.index[-1].date()
    
    def get_earliest_date(self, ticker: str) -> date:
        """Get the earliest data date for ticker."""
        df = self.load_ticker(ticker)
        return df.index[0].date()
    
    def clear_cache(self) -> None:
        """Clear in-memory data cache."""
        self._cache.clear()


def setup_logging(
    name: str = "daily_trading",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Setup logging for daily trading system.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file to write logs to
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def calculate_return_direction(price_before: float, price_after: float) -> Literal["UP", "DOWN", "FLAT"]:
    """
    Calculate direction of price movement.
    
    Args:
        price_before: Price at start
        price_after: Price at end
    
    Returns:
        "UP", "DOWN", or "FLAT" (within 0.1%)
    """
    pct_change = (price_after - price_before) / price_before
    
    if pct_change > 0.001:  # > 0.1%
        return "UP"
    elif pct_change < -0.001:  # < -0.1%
        return "DOWN"
    else:
        return "FLAT"


def signal_matches_direction(
    signal: Literal["BUY", "SELL", "HOLD"],
    direction: Literal["UP", "DOWN", "FLAT"],
) -> bool:
    """
    Check if signal correctly predicted direction.
    
    Args:
        signal: Generated trading signal
        direction: Actual price direction
    
    Returns:
        True if signal was correct
    """
    if signal == "BUY":
        return direction == "UP"
    elif signal == "SELL":
        return direction == "DOWN"
    else:  # HOLD
        return direction == "FLAT"


def calculate_signal_profit(
    signal: Literal["BUY", "SELL", "HOLD"],
    price_before: float,
    price_after: float,
    position_size: float = 1.0,
) -> float:
    """
    Calculate profit/loss from following a signal.
    
    Args:
        signal: Trading signal
        price_before: Entry price
        price_after: Exit price
        position_size: Size of position (1.0 = full position)
    
    Returns:
        Profit as percentage of capital
    """
    pct_change = (price_after - price_before) / price_before
    
    if signal == "BUY":
        return pct_change * position_size
    elif signal == "SELL":
        return -pct_change * position_size
    else:  # HOLD
        return 0.0

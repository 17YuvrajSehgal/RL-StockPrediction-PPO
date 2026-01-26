"""
Walk-forward validation framework for trading strategy assessment.

This module implements rigorous walk-forward testing:
- Train on data up to day N
- Generate signal for day N+1
- Record actual outcome on day N+1
- Repeat for entire test period

This is the gold standard for validating trading strategies as it
simulates real-world conditions where we only have past data.

Design Principles:
    - Strict temporal ordering (no lookahead bias)
    - Comprehensive metrics aggregation
    - Comparison against baselines (buy-and-hold)
    - Detailed per-day breakdowns for analysis
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import concurrent.futures

import numpy as np
import pandas as pd

from daily_trading.config import (
    WalkForwardConfig,
    DailyTrainingConfig,
    TrainingMode,
    SignalConfig,
    PathConfig,
)
from daily_trading.trainer import DailyTrainer, TrainResult
from daily_trading.signals import SignalGenerator, TradingSignal
from daily_trading.utils import (
    TradingCalendar,
    DataLoader,
    calculate_return_direction,
    signal_matches_direction,
    calculate_signal_profit,
    setup_logging,
)


@dataclass
class DayResult:
    """
    Result for a single day in walk-forward validation.
    
    Attributes:
        test_date: The date being tested
        train_as_of: Training data cutoff
        signal: Generated trading signal
        actual_open: Actual open price on test_date
        actual_close: Actual close price on test_date
        actual_return: Actual return (close/open - 1)
        actual_direction: UP, DOWN, or FLAT
        signal_correct: Whether signal matched direction
        signal_profit: Profit from following signal
        train_result: Training result (if saved)
    """
    
    test_date: date
    train_as_of: date
    signal: TradingSignal
    actual_open: float
    actual_close: float
    actual_return: float
    actual_direction: Literal["UP", "DOWN", "FLAT"]
    signal_correct: bool
    signal_profit: float
    train_result: Optional[TrainResult] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_date": self.test_date.isoformat(),
            "train_as_of": self.train_as_of.isoformat(),
            "signal": self.signal.to_dict(),
            "actual_open": round(self.actual_open, 2),
            "actual_close": round(self.actual_close, 2),
            "actual_return": round(self.actual_return, 6),
            "actual_direction": self.actual_direction,
            "signal_correct": self.signal_correct,
            "signal_profit": round(self.signal_profit, 6),
        }


@dataclass
class WalkForwardResult:
    """
    Complete walk-forward validation results.
    
    Contains aggregate metrics and per-day breakdowns.
    
    Attributes:
        ticker: Stock ticker tested
        start_date: First test date
        end_date: Last test date
        config: Configuration used
        daily_results: List of per-day results
        
    Metrics:
        total_signals: Total signals generated
        correct_signals: Signals matching actual direction
        accuracy: Correct / Total
        cumulative_return: Total return from following signals
        sharpe_ratio: Risk-adjusted return
        max_drawdown: Worst peak-to-trough decline
        signal_distribution: Count by signal type
        buy_hold_return: Buy-and-hold baseline return
        excess_return: Strategy return - buy-and-hold return
    """
    
    ticker: str
    start_date: date
    end_date: date
    config: WalkForwardConfig
    daily_results: List[DayResult] = field(default_factory=list)
    
    # Computed metrics
    total_signals: int = 0
    correct_signals: int = 0
    accuracy: float = 0.0
    cumulative_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    signal_distribution: Dict[str, int] = field(default_factory=dict)
    buy_hold_return: float = 0.0
    excess_return: float = 0.0
    total_training_time: float = 0.0
    
    def compute_metrics(self) -> None:
        """Compute aggregate metrics from daily results."""
        if not self.daily_results:
            return
        
        self.total_signals = len(self.daily_results)
        self.correct_signals = sum(1 for r in self.daily_results if r.signal_correct)
        self.accuracy = self.correct_signals / self.total_signals if self.total_signals > 0 else 0.0
        
        # Signal distribution
        self.signal_distribution = {"BUY": 0, "SELL": 0, "HOLD": 0}
        for r in self.daily_results:
            self.signal_distribution[r.signal.signal] += 1
        
        # Returns
        returns = [r.signal_profit for r in self.daily_results]
        self.cumulative_return = np.prod([1 + r for r in returns]) - 1
        
        # Sharpe ratio (annualized)
        if len(returns) > 1 and np.std(returns) > 0:
            self.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            self.sharpe_ratio = 0.0
        
        # Max drawdown
        equity_curve = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        self.max_drawdown = float(np.min(drawdowns))
        
        # Buy-and-hold comparison
        first_open = self.daily_results[0].actual_open
        last_close = self.daily_results[-1].actual_close
        self.buy_hold_return = (last_close - first_open) / first_open
        
        # Excess return
        self.excess_return = self.cumulative_return - self.buy_hold_return
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ticker": self.ticker,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "total_signals": self.total_signals,
            "correct_signals": self.correct_signals,
            "accuracy": round(self.accuracy, 4),
            "cumulative_return": round(self.cumulative_return, 6),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 6),
            "signal_distribution": self.signal_distribution,
            "buy_hold_return": round(self.buy_hold_return, 6),
            "excess_return": round(self.excess_return, 6),
            "total_training_time": round(self.total_training_time, 2),
            "daily_results": [r.to_dict() for r in self.daily_results],
        }
    
    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> WalkForwardResult:
        """Load results from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        result = cls(
            ticker=data["ticker"],
            start_date=date.fromisoformat(data["start_date"]),
            end_date=date.fromisoformat(data["end_date"]),
            config=WalkForwardConfig(),  # Would need to serialize config too
        )
        result.total_signals = data["total_signals"]
        result.correct_signals = data["correct_signals"]
        result.accuracy = data["accuracy"]
        result.cumulative_return = data["cumulative_return"]
        result.sharpe_ratio = data["sharpe_ratio"]
        result.max_drawdown = data["max_drawdown"]
        result.signal_distribution = data["signal_distribution"]
        result.buy_hold_return = data["buy_hold_return"]
        result.excess_return = data["excess_return"]
        
        return result
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            f"WALK-FORWARD VALIDATION RESULTS: {self.ticker}",
            "=" * 70,
            f"Period: {self.start_date} to {self.end_date}",
            f"Total Trading Days: {self.total_signals}",
            "-" * 70,
            "SIGNAL PERFORMANCE:",
            f"  Accuracy:           {self.accuracy:.1%} ({self.correct_signals}/{self.total_signals})",
            f"  Signal Distribution: BUY={self.signal_distribution.get('BUY', 0)}, "
            f"SELL={self.signal_distribution.get('SELL', 0)}, "
            f"HOLD={self.signal_distribution.get('HOLD', 0)}",
            "-" * 70,
            "RETURNS:",
            f"  Strategy Return:    {self.cumulative_return:+.2%}",
            f"  Buy-and-Hold:       {self.buy_hold_return:+.2%}",
            f"  Excess Return:      {self.excess_return:+.2%}",
            "-" * 70,
            "RISK METRICS:",
            f"  Sharpe Ratio:       {self.sharpe_ratio:.2f}",
            f"  Max Drawdown:       {self.max_drawdown:.2%}",
            "-" * 70,
            f"Total Training Time:  {self.total_training_time/3600:.1f} hours",
            "=" * 70,
        ]
        return "\n".join(lines)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert daily results to DataFrame for analysis."""
        records = []
        for r in self.daily_results:
            records.append({
                "date": r.test_date,
                "signal": r.signal.signal,
                "confidence": r.signal.confidence,
                "target_weight": r.signal.target_weight,
                "actual_open": r.actual_open,
                "actual_close": r.actual_close,
                "actual_return": r.actual_return,
                "actual_direction": r.actual_direction,
                "signal_correct": r.signal_correct,
                "signal_profit": r.signal_profit,
            })
        return pd.DataFrame(records)


class WalkForwardValidator:
    """
    Walk-forward validation framework.
    
    This class orchestrates the complete walk-forward testing process:
    1. For each test day in the range
    2. Train model on data up to previous day
    3. Generate signal for test day
    4. Record actual outcome
    5. Aggregate results
    
    Attributes:
        config: Walk-forward configuration
        paths: Path configuration
        calendar: Trading calendar
    
    Example:
        >>> validator = WalkForwardValidator(config, paths)
        >>> 
        >>> # Run validation for Q4 2025
        >>> result = validator.validate(
        ...     ticker="AAPL",
        ...     start_date=date(2025, 10, 1),
        ...     end_date=date(2025, 12, 31),
        ... )
        >>> 
        >>> print(result.summary())
    """
    
    def __init__(
        self,
        config: WalkForwardConfig,
        paths: PathConfig,
        env_config: Optional[Any] = None,
    ):
        """
        Initialize walk-forward validator.
        
        Args:
            config: Walk-forward configuration
            paths: Path configuration
            env_config: Trading environment configuration
        """
        self.config = config
        self.paths = paths
        self.env_config = env_config
        
        self.calendar = TradingCalendar()
        self.data_loader = DataLoader(paths.data_dir)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        paths.ensure_directories()
    
    def validate(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        training_mode: Optional[TrainingMode] = None,
        timesteps: Optional[int] = None,
        resume_from: Optional[date] = None,
        verbose: bool = True,
    ) -> WalkForwardResult:
        """
        Run walk-forward validation.
        
        Args:
            ticker: Stock ticker to validate
            start_date: First test date
            end_date: Last test date
            training_mode: Override training mode
            timesteps: Override training timesteps
            resume_from: Resume from specific date (for interrupted runs)
            verbose: Print progress
        
        Returns:
            WalkForwardResult with aggregate metrics and daily breakdown
        """
        self.logger.info(f"Starting walk-forward validation for {ticker}")
        self.logger.info(f"Period: {start_date} to {end_date}")
        
        # Get trading days in range
        test_days = self.calendar.trading_days_between(start_date, end_date)
        
        if not test_days:
            raise ValueError(f"No trading days in range {start_date} to {end_date}")
        
        self.logger.info(f"Testing {len(test_days)} trading days")
        
        # Initialize result
        result = WalkForwardResult(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            config=self.config,
        )
        
        # Create trainer and signal generator
        training_config = self.config.training_config
        if training_mode:
            from dataclasses import replace
            training_config = replace(training_config, mode=training_mode)
        
        trainer = DailyTrainer(
            config=training_config,
            paths=self.paths,
            env_config=self.env_config,
        )
        
        signal_generator = SignalGenerator(
            config=self.config.signal_config,
            paths=self.paths,
            env_config=self.env_config,
        )
        
        # Load actual price data for validation
        df = self.data_loader.load_ticker(ticker)
        
        # Base model for fine-tuning (will be updated each iteration)
        base_model_path = None
        
        # Walk forward
        total_training_time = 0.0
        
        for i, test_day in enumerate(test_days):
            # Skip if resuming
            if resume_from and test_day < resume_from:
                continue
            
            # Training cutoff is previous trading day
            train_as_of = self.calendar.previous_trading_day(test_day)
            
            if verbose:
                print(f"\n[{i+1}/{len(test_days)}] Testing {test_day} (train up to {train_as_of})")
            
            try:
                # 1. Train model
                train_result = trainer.train(
                    ticker=ticker,
                    as_of_date=train_as_of,
                    mode=training_mode or training_config.mode,
                    base_model_path=base_model_path,
                    timesteps=timesteps,
                )
                
                total_training_time += train_result.training_time_seconds
                
                if not train_result.success:
                    self.logger.warning(f"Training failed for {test_day}: {train_result.error}")
                    continue
                
                # Update base model for next iteration
                if training_config.mode == TrainingMode.FINE_TUNE:
                    base_model_path = train_result.model_path
                
                # 2. Generate signal
                signal = signal_generator.generate(
                    ticker=ticker,
                    for_date=test_day,
                    model_path=train_result.model_path,
                    data_as_of=train_as_of,
                )
                
                # 3. Get actual outcome
                if test_day.isoformat() not in df.index.strftime('%Y-%m-%d').values:
                    # Test day not in data yet (future date)
                    self.logger.warning(f"No data for test day {test_day}")
                    continue
                
                day_data = df.loc[df.index.date == test_day]
                if day_data.empty:
                    continue
                
                actual_open = float(day_data['open'].iloc[0])
                actual_close = float(day_data['close'].iloc[0])
                actual_return = (actual_close - actual_open) / actual_open
                actual_direction = calculate_return_direction(actual_open, actual_close)
                
                # 4. Evaluate signal
                signal_correct = signal_matches_direction(signal.signal, actual_direction)
                signal_profit = calculate_signal_profit(
                    signal.signal, actual_open, actual_close
                )
                
                # 5. Record result
                day_result = DayResult(
                    test_date=test_day,
                    train_as_of=train_as_of,
                    signal=signal,
                    actual_open=actual_open,
                    actual_close=actual_close,
                    actual_return=actual_return,
                    actual_direction=actual_direction,
                    signal_correct=signal_correct,
                    signal_profit=signal_profit,
                    train_result=train_result if self.config.save_intermediate_models else None,
                )
                
                result.daily_results.append(day_result)
                
                if verbose:
                    correct_str = "✓" if signal_correct else "✗"
                    print(f"  Signal: {signal.signal} | Actual: {actual_direction} | "
                          f"{correct_str} | Return: {signal_profit:+.2%}")
                
            except Exception as e:
                self.logger.error(f"Error processing {test_day}: {e}")
                if verbose:
                    print(f"  ERROR: {e}")
                continue
        
        # Compute aggregate metrics
        result.total_training_time = total_training_time
        result.compute_metrics()
        
        # Save results
        result_path = self.paths.result_path(ticker, start_date, end_date)
        result.save(result_path)
        self.logger.info(f"Results saved to {result_path}")
        
        if verbose:
            print(result.summary())
        
        return result
    
    def validate_comparison(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> Dict[str, WalkForwardResult]:
        """
        Compare full training vs fine-tuning in walk-forward setting.
        
        Args:
            ticker: Stock ticker
            start_date: First test date
            end_date: Last test date
        
        Returns:
            Dictionary with "full" and "fine_tune" results
        """
        results = {}
        
        # Full training
        self.logger.info("Running walk-forward with FULL training mode...")
        results["full"] = self.validate(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            training_mode=TrainingMode.FULL,
        )
        
        # Fine-tuning
        self.logger.info("Running walk-forward with FINE_TUNE training mode...")
        results["fine_tune"] = self.validate(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            training_mode=TrainingMode.FINE_TUNE,
        )
        
        # Print comparison
        print("\n" + "=" * 70)
        print("TRAINING MODE COMPARISON")
        print("=" * 70)
        print(f"{'Metric':<25} {'Full Train':>20} {'Fine-Tune':>20}")
        print("-" * 70)
        print(f"{'Accuracy':<25} {results['full'].accuracy:>19.1%} {results['fine_tune'].accuracy:>19.1%}")
        print(f"{'Cumulative Return':<25} {results['full'].cumulative_return:>+19.2%} {results['fine_tune'].cumulative_return:>+19.2%}")
        print(f"{'Sharpe Ratio':<25} {results['full'].sharpe_ratio:>20.2f} {results['fine_tune'].sharpe_ratio:>20.2f}")
        print(f"{'Max Drawdown':<25} {results['full'].max_drawdown:>19.2%} {results['fine_tune'].max_drawdown:>19.2%}")
        print(f"{'Training Time (hrs)':<25} {results['full'].total_training_time/3600:>20.1f} {results['fine_tune'].total_training_time/3600:>20.1f}")
        print("=" * 70)
        
        return results


def quick_validate(
    ticker: str,
    days: int = 5,
    timesteps: int = 10_000,
    paths: Optional[PathConfig] = None,
) -> WalkForwardResult:
    """
    Quick validation for testing/development.
    
    Runs a short walk-forward test with minimal training.
    
    Args:
        ticker: Stock ticker
        days: Number of days to test
        timesteps: Training timesteps per day
        paths: Path configuration
    
    Returns:
        WalkForwardResult
    """
    from datetime import date, timedelta
    
    paths = paths or PathConfig()
    calendar = TradingCalendar()
    data_loader = DataLoader(paths.data_dir)
    
    # Get recent date range
    latest_date = data_loader.get_latest_date(ticker)
    end_date = calendar.previous_trading_day(latest_date)
    
    # Go back 'days' trading days
    start_date = end_date
    for _ in range(days):
        start_date = calendar.previous_trading_day(start_date)
    
    # Quick config
    config = WalkForwardConfig.quick_validation()
    
    validator = WalkForwardValidator(config, paths)
    return validator.validate(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        timesteps=timesteps,
    )

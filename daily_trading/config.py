"""
Configuration management for daily trading system.

This module provides immutable, validated configuration dataclasses for
all aspects of the daily training and signal generation pipeline.

Design Principles:
    - Immutable configurations (frozen dataclasses) for thread safety
    - Comprehensive validation in __post_init__
    - Factory methods for common presets
    - Type hints throughout for IDE support
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from typing import Literal, Optional


class TrainingMode(Enum):
    """Mode of training for daily updates."""
    
    FULL = "full"  # Train from scratch
    FINE_TUNE = "fine_tune"  # Fine-tune existing model
    
    def __str__(self) -> str:
        return self.value


class WindowType(Enum):
    """Type of training data window."""
    
    ROLLING = "rolling"  # Fixed-size sliding window
    EXPANDING = "expanding"  # All available history
    
    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class TrainingWindowConfig:
    """
    Configuration for training data window.
    
    Determines how much historical data to use for training.
    
    Attributes:
        window_type: Rolling (fixed size) or expanding (all history)
        rolling_days: Number of days for rolling window (ignored if expanding)
        min_days: Minimum days required for training
        max_days: Maximum days to use (caps expanding window)
    
    Example:
        >>> # Rolling 2-year window
        >>> config = TrainingWindowConfig.rolling(504)
        >>> 
        >>> # Expanding window with 5-year cap
        >>> config = TrainingWindowConfig.expanding(max_days=1260)
    """
    
    window_type: WindowType = WindowType.ROLLING
    rolling_days: int = 504  # ~2 years of trading days
    min_days: int = 100  # Minimum required (lowered for flexibility)
    max_days: int = 2520  # Maximum 10 years
    
    def __post_init__(self) -> None:
        """Validate window configuration."""
        if self.rolling_days <= 0:
            raise ValueError(f"rolling_days must be positive, got {self.rolling_days}")
        if self.min_days <= 0:
            raise ValueError(f"min_days must be positive, got {self.min_days}")
        if self.min_days > self.rolling_days:
            raise ValueError(
                f"min_days ({self.min_days}) cannot exceed rolling_days ({self.rolling_days})"
            )
        if self.max_days < self.rolling_days:
            raise ValueError(
                f"max_days ({self.max_days}) cannot be less than rolling_days ({self.rolling_days})"
            )
    
    @classmethod
    def rolling(cls, days: int = 504, min_days: int = None) -> TrainingWindowConfig:
        """Create rolling window configuration."""
        # Default min_days to half of rolling days, but at least 60
        min_days = min_days or max(60, days // 2)
        return cls(window_type=WindowType.ROLLING, rolling_days=days, min_days=min_days)
    
    @classmethod
    def expanding(cls, max_days: int = 2520) -> TrainingWindowConfig:
        """Create expanding window configuration."""
        return cls(window_type=WindowType.EXPANDING, max_days=max_days, min_days=100)
    
    @classmethod
    def volatile_market(cls) -> TrainingWindowConfig:
        """Short window for volatile/rapidly changing markets."""
        return cls(window_type=WindowType.ROLLING, rolling_days=126, min_days=60)  # 6 months
    
    @classmethod
    def stable_market(cls) -> TrainingWindowConfig:
        """Longer window for stable market conditions."""
        return cls(window_type=WindowType.ROLLING, rolling_days=756, min_days=200)  # 3 years
    
    def get_start_date(self, as_of_date: date, available_start: date) -> date:
        """
        Calculate training start date based on window configuration.
        
        Args:
            as_of_date: Training cutoff date (no data after this)
            available_start: Earliest available data date
        
        Returns:
            Calculated start date for training data
        """
        if self.window_type == WindowType.ROLLING:
            calculated_start = as_of_date - timedelta(days=int(self.rolling_days * 1.5))
            return max(calculated_start, available_start)
        else:  # EXPANDING
            return available_start


@dataclass(frozen=True)
class DailyTrainingConfig:
    """
    Configuration for daily model training.
    
    Attributes:
        mode: Full training or fine-tuning
        window: Training data window configuration
        timesteps_full: Total timesteps for full training
        timesteps_fine_tune: Timesteps for fine-tuning
        learning_rate: Learning rate (lower for fine-tuning recommended)
        save_checkpoints: Whether to save intermediate checkpoints
        checkpoint_freq: Save checkpoint every N steps
        early_stopping: Enable early stopping based on validation
        patience: Early stopping patience (epochs without improvement)
    
    Example:
        >>> # Fast daily fine-tuning
        >>> config = DailyTrainingConfig.fine_tune()
        >>> 
        >>> # Full weekend retrain
        >>> config = DailyTrainingConfig.full_train()
    """
    
    mode: TrainingMode = TrainingMode.FINE_TUNE
    window: TrainingWindowConfig = field(default_factory=TrainingWindowConfig)
    timesteps_full: int = 500_000
    timesteps_fine_tune: int = 50_000
    learning_rate_full: float = 1e-4
    learning_rate_fine_tune: float = 5e-5
    save_checkpoints: bool = True
    checkpoint_freq: int = 25_000
    early_stopping: bool = True
    patience: int = 5
    validation_split: float = 0.15
    seed: int = 42
    
    def __post_init__(self) -> None:
        """Validate training configuration."""
        if self.timesteps_full <= 0:
            raise ValueError(f"timesteps_full must be positive")
        if self.timesteps_fine_tune <= 0:
            raise ValueError(f"timesteps_fine_tune must be positive")
        if self.learning_rate_full <= 0:
            raise ValueError(f"learning_rate_full must be positive")
        if self.learning_rate_fine_tune <= 0:
            raise ValueError(f"learning_rate_fine_tune must be positive")
        if not 0 < self.validation_split < 1:
            raise ValueError(f"validation_split must be in (0, 1)")
    
    @property
    def timesteps(self) -> int:
        """Get timesteps based on current mode."""
        return self.timesteps_fine_tune if self.mode == TrainingMode.FINE_TUNE else self.timesteps_full
    
    @property
    def learning_rate(self) -> float:
        """Get learning rate based on current mode."""
        return self.learning_rate_fine_tune if self.mode == TrainingMode.FINE_TUNE else self.learning_rate_full
    
    @classmethod
    def fine_tune(cls, window: Optional[TrainingWindowConfig] = None) -> DailyTrainingConfig:
        """Quick fine-tuning configuration for daily updates."""
        return cls(
            mode=TrainingMode.FINE_TUNE,
            window=window or TrainingWindowConfig.rolling(252),  # 1 year for fine-tune
            timesteps_fine_tune=50_000,
            learning_rate_fine_tune=5e-5,
        )
    
    @classmethod
    def full_train(cls, window: Optional[TrainingWindowConfig] = None) -> DailyTrainingConfig:
        """Full training configuration."""
        return cls(
            mode=TrainingMode.FULL,
            window=window or TrainingWindowConfig.rolling(504),
            timesteps_full=500_000,
            learning_rate_full=1e-4,
        )
    
    @classmethod
    def research(cls) -> DailyTrainingConfig:
        """Configuration for research/comparison experiments."""
        return cls(
            mode=TrainingMode.FULL,
            window=TrainingWindowConfig.rolling(504),
            timesteps_full=200_000,
            timesteps_fine_tune=25_000,
            save_checkpoints=True,
        )


@dataclass(frozen=True)
class SignalConfig:
    """
    Configuration for trading signal generation.
    
    Attributes:
        buy_threshold: Weight above this → BUY signal
        sell_threshold: Weight below this → SELL signal
        confidence_scale: How to scale weight to confidence [0, 1]
        require_min_confidence: Minimum confidence to generate non-HOLD signal
        deterministic: Use deterministic policy (no exploration)
    
    Example:
        >>> config = SignalConfig(buy_threshold=0.3, sell_threshold=-0.3)
        >>> # Weight 0.5 → BUY with confidence ~0.67
        >>> # Weight 0.0 → HOLD
        >>> # Weight -0.6 → SELL with confidence ~0.8
    """
    
    buy_threshold: float = 0.3
    sell_threshold: float = -0.3
    min_confidence: float = 0.0  # Minimum confidence to act (0 = always act)
    deterministic: bool = True
    
    def __post_init__(self) -> None:
        """Validate signal configuration."""
        if not -1 <= self.sell_threshold < self.buy_threshold <= 1:
            raise ValueError(
                f"Invalid thresholds: sell={self.sell_threshold}, buy={self.buy_threshold}. "
                f"Must satisfy: -1 <= sell < buy <= 1"
            )
        if not 0 <= self.min_confidence <= 1:
            raise ValueError(f"min_confidence must be in [0, 1]")
    
    def interpret_weight(self, weight: float) -> tuple[Literal["BUY", "SELL", "HOLD"], float]:
        """
        Interpret model weight as signal and confidence.
        
        Args:
            weight: Model output in [-1, 1]
        
        Returns:
            Tuple of (signal, confidence)
        """
        if weight > self.buy_threshold:
            signal = "BUY"
            # Scale confidence: threshold→0, 1→1
            confidence = (weight - self.buy_threshold) / (1 - self.buy_threshold)
        elif weight < self.sell_threshold:
            signal = "SELL"
            # Scale confidence: threshold→0, -1→1
            confidence = (self.sell_threshold - weight) / (1 + self.sell_threshold)
        else:
            signal = "HOLD"
            # Confidence for HOLD is how close to center
            confidence = 1 - abs(weight) / max(abs(self.buy_threshold), abs(self.sell_threshold))
        
        return signal, min(max(confidence, 0.0), 1.0)


@dataclass(frozen=True)
class WalkForwardConfig:
    """
    Configuration for walk-forward validation.
    
    Attributes:
        training_window: Window configuration for each training iteration
        training_config: Training configuration to use
        signal_config: Signal interpretation configuration
        skip_weekends: Skip weekends in walk-forward
        skip_holidays: Skip market holidays
        parallel_training: Enable parallel training (if multiple GPUs)
        save_intermediate_models: Save each day's model
    
    Example:
        >>> config = WalkForwardConfig(
        ...     training_window=TrainingWindowConfig.rolling(504),
        ...     training_config=DailyTrainingConfig.fine_tune(),
        ... )
    """
    
    training_window: TrainingWindowConfig = field(default_factory=TrainingWindowConfig)
    training_config: DailyTrainingConfig = field(default_factory=DailyTrainingConfig.fine_tune)
    signal_config: SignalConfig = field(default_factory=SignalConfig)
    skip_weekends: bool = True
    skip_holidays: bool = True
    parallel_training: bool = False
    save_intermediate_models: bool = False
    comparison_baseline: Literal["buy_hold", "random", "none"] = "buy_hold"
    
    @classmethod
    def quick_validation(cls) -> WalkForwardConfig:
        """Quick validation with minimal training."""
        return cls(
            training_window=TrainingWindowConfig.rolling(252),
            training_config=DailyTrainingConfig(
                mode=TrainingMode.FINE_TUNE,
                timesteps_fine_tune=10_000,
            ),
        )
    
    @classmethod
    def production_validation(cls) -> WalkForwardConfig:
        """Thorough validation for production deployment decision."""
        return cls(
            training_window=TrainingWindowConfig.rolling(504),
            training_config=DailyTrainingConfig.full_train(),
            save_intermediate_models=True,
        )


@dataclass(frozen=True)
class PathConfig:
    """
    Configuration for file paths and directories.
    
    Attributes:
        base_dir: Base project directory
        data_dir: Directory for market data
        models_dir: Directory for trained models
        signals_dir: Directory for generated signals
        results_dir: Directory for validation results
        logs_dir: Directory for logs
    """
    
    base_dir: Path = field(default_factory=lambda: Path.cwd())
    data_subdir: str = "yf_data"
    models_subdir: str = "daily_models"
    signals_subdir: str = "signals"
    results_subdir: str = "walk_forward_results"
    logs_subdir: str = "daily_trading_logs"
    
    @property
    def data_dir(self) -> Path:
        return self.base_dir / self.data_subdir
    
    @property
    def models_dir(self) -> Path:
        return self.base_dir / self.models_subdir
    
    @property
    def signals_dir(self) -> Path:
        return self.base_dir / self.signals_subdir
    
    @property
    def results_dir(self) -> Path:
        return self.base_dir / self.results_subdir
    
    @property
    def logs_dir(self) -> Path:
        return self.base_dir / self.logs_subdir
    
    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for path in [self.data_dir, self.models_dir, self.signals_dir, 
                     self.results_dir, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)
    
    def model_path(self, ticker: str, as_of_date: date) -> Path:
        """Get path for a specific model version."""
        return self.models_dir / ticker / as_of_date.isoformat()
    
    def signal_path(self, ticker: str, for_date: date) -> Path:
        """Get path for a specific signal file."""
        return self.signals_dir / ticker / f"{for_date.isoformat()}.json"
    
    def result_path(self, ticker: str, start_date: date, end_date: date) -> Path:
        """Get path for walk-forward results."""
        return self.results_dir / f"{ticker}_{start_date.isoformat()}_to_{end_date.isoformat()}.json"


@dataclass(frozen=True)
class DailyTradingConfig:
    """
    Master configuration combining all daily trading settings.
    
    This is the main configuration class that users should instantiate.
    
    Example:
        >>> # Use defaults
        >>> config = DailyTradingConfig()
        >>> 
        >>> # Custom configuration
        >>> config = DailyTradingConfig(
        ...     training=DailyTrainingConfig.full_train(),
        ...     signal=SignalConfig(buy_threshold=0.4, sell_threshold=-0.4),
        ... )
    """
    
    training: DailyTrainingConfig = field(default_factory=DailyTrainingConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    @classmethod
    def development(cls, base_dir: Path = None) -> DailyTradingConfig:
        """Development configuration with fast training."""
        return cls(
            training=DailyTrainingConfig.research(),
            paths=PathConfig(base_dir=base_dir or Path.cwd()),
        )
    
    @classmethod
    def production(cls, base_dir: Path = None) -> DailyTradingConfig:
        """Production configuration with thorough training."""
        return cls(
            training=DailyTrainingConfig.full_train(),
            walk_forward=WalkForwardConfig.production_validation(),
            paths=PathConfig(base_dir=base_dir or Path.cwd()),
        )

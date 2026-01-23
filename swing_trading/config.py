"""
Configuration management for swing trading environment.

This module provides dataclasses for all configuration parameters with validation,
factory methods for common presets, and immutable design for thread safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class MarginConfig:
    """
    Margin model configuration.
    
    Attributes:
        short_proceeds_haircut: Fraction of short-sale proceeds that cannot be used
            as buying power. Example: 0.5 means only 50% of short proceeds are usable.
            This prevents unrealistic "infinite cash" from shorting.
        initial_margin_long: Fraction of long position value that must be funded by
            equity (no borrowing). For strict cash-only longs, keep at 1.0.
    
    Example:
        >>> # Conservative retail trader settings
        >>> config = MarginConfig(short_proceeds_haircut=0.5, initial_margin_long=1.0)
        >>> 
        >>> # More aggressive with margin
        >>> config = MarginConfig(short_proceeds_haircut=0.3, initial_margin_long=0.5)
    """
    
    short_proceeds_haircut: float = 0.5
    initial_margin_long: float = 1.0
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 <= self.short_proceeds_haircut <= 1.0:
            raise ValueError(
                f"short_proceeds_haircut must be in [0, 1], got {self.short_proceeds_haircut}"
            )
        if not 0.0 < self.initial_margin_long <= 1.0:
            raise ValueError(
                f"initial_margin_long must be in (0, 1], got {self.initial_margin_long}"
            )


@dataclass(frozen=True)
class TradingConfig:
    """
    Trading cost and constraint configuration.
    
    Attributes:
        fee_rate: Transaction fee as fraction of traded notional (e.g., 0.0005 = 5 bps)
        slippage_rate: Slippage as fraction of traded notional (e.g., 0.0002 = 2 bps)
        allow_short: Whether short selling is permitted
        max_abs_weight: Maximum absolute portfolio weight (1.0 = 100% long or short)
        cash_only: If True, enforce cash-only constraints (no margin borrowing)
        turnover_penalty: Penalty coefficient for portfolio turnover (discourages churn)
        risk_penalty: Penalty coefficient for portfolio risk (optional, future use)
    
    Example:
        >>> # Low-cost environment for backtesting
        >>> config = TradingConfig(fee_rate=0.0001, slippage_rate=0.0001)
        >>> 
        >>> # Realistic retail trading costs
        >>> config = TradingConfig(fee_rate=0.0005, slippage_rate=0.0002)
    """
    
    fee_rate: float = 0.0005  # 5 bps
    slippage_rate: float = 0.0002  # 2 bps
    allow_short: bool = True
    max_abs_weight: float = 1.0
    cash_only: bool = True
    turnover_penalty: float = 0.0
    risk_penalty: float = 0.0
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.fee_rate < 0.0:
            raise ValueError(f"fee_rate must be non-negative, got {self.fee_rate}")
        if self.slippage_rate < 0.0:
            raise ValueError(f"slippage_rate must be non-negative, got {self.slippage_rate}")
        if self.max_abs_weight <= 0.0:
            raise ValueError(f"max_abs_weight must be positive, got {self.max_abs_weight}")
        if self.turnover_penalty < 0.0:
            raise ValueError(f"turnover_penalty must be non-negative, got {self.turnover_penalty}")
        if self.risk_penalty < 0.0:
            raise ValueError(f"risk_penalty must be non-negative, got {self.risk_penalty}")
    
    @property
    def total_cost_rate(self) -> float:
        """Total transaction cost rate (fees + slippage)."""
        return self.fee_rate + self.slippage_rate


@dataclass(frozen=True)
class EnvironmentConfig:
    """
    Environment behavior configuration.
    
    Attributes:
        initial_equity: Starting portfolio value in dollars
        lookback: Number of historical bars in observation window
        episode_length: Number of steps per episode
        random_start: If True, randomize episode start position
        seed: Random seed for reproducibility
        execution: Execution model - "next_open" means action at t executes at open(t+1)
        include_portfolio_features: If True, add portfolio state to observations
    
    Example:
        >>> # Short episodes for fast training
        >>> config = EnvironmentConfig(episode_length=60, lookback=30)
        >>> 
        >>> # Long episodes for realistic evaluation
        >>> config = EnvironmentConfig(episode_length=252, lookback=120)
    """
    
    initial_equity: float = 100_000.0
    lookback: int = 120
    episode_length: int = 252
    random_start: bool = True
    seed: int = 42
    execution: Literal["next_open"] = "next_open"
    include_portfolio_features: bool = True
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.initial_equity <= 0.0:
            raise ValueError(f"initial_equity must be positive, got {self.initial_equity}")
        if self.lookback <= 0:
            raise ValueError(f"lookback must be positive, got {self.lookback}")
        if self.episode_length <= 0:
            raise ValueError(f"episode_length must be positive, got {self.episode_length}")
        if self.execution not in ("next_open",):
            raise ValueError(f"execution must be 'next_open', got {self.execution}")


@dataclass(frozen=True)
class SwingTradingConfig:
    """
    Master configuration combining all settings.
    
    This is the main configuration class that users should instantiate.
    It combines margin, trading, and environment settings into a single object.
    
    Attributes:
        margin: Margin model configuration
        trading: Trading cost and constraint configuration
        environment: Environment behavior configuration
    
    Example:
        >>> # Use defaults
        >>> config = SwingTradingConfig()
        >>> 
        >>> # Customize specific settings
        >>> config = SwingTradingConfig(
        ...     trading=TradingConfig(fee_rate=0.0001, allow_short=False),
        ...     environment=EnvironmentConfig(episode_length=60)
        ... )
        >>> 
        >>> # Use factory presets
        >>> config = SwingTradingConfig.conservative()
        >>> config = SwingTradingConfig.aggressive()
    """
    
    margin: MarginConfig = field(default_factory=MarginConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    
    @classmethod
    def conservative(cls) -> SwingTradingConfig:
        """
        Conservative configuration for risk-averse trading.
        
        - No short selling
        - Higher transaction costs
        - Strict cash-only constraints
        - Longer lookback for stability
        
        Returns:
            SwingTradingConfig with conservative settings
        """
        return cls(
            margin=MarginConfig(
                short_proceeds_haircut=0.7,
                initial_margin_long=1.0,
            ),
            trading=TradingConfig(
                fee_rate=0.001,  # 10 bps
                slippage_rate=0.0005,  # 5 bps
                allow_short=False,
                max_abs_weight=0.8,
                cash_only=True,
                turnover_penalty=0.0001,
            ),
            environment=EnvironmentConfig(
                lookback=180,
                episode_length=252,
            ),
        )
    
    @classmethod
    def aggressive(cls) -> SwingTradingConfig:
        """
        Aggressive configuration for active trading.
        
        - Short selling enabled
        - Lower transaction costs
        - Higher leverage allowed
        - Shorter lookback for responsiveness
        
        Returns:
            SwingTradingConfig with aggressive settings
        """
        return cls(
            margin=MarginConfig(
                short_proceeds_haircut=0.3,
                initial_margin_long=1.0,
            ),
            trading=TradingConfig(
                fee_rate=0.0002,  # 2 bps
                slippage_rate=0.0001,  # 1 bp
                allow_short=True,
                max_abs_weight=1.5,
                cash_only=True,
                turnover_penalty=0.0,
            ),
            environment=EnvironmentConfig(
                lookback=60,
                episode_length=126,
            ),
        )
    
    @classmethod
    def backtest(cls) -> SwingTradingConfig:
        """
        Configuration optimized for backtesting.
        
        - Minimal transaction costs
        - Deterministic (no random starts)
        - Long episodes for full data coverage
        
        Returns:
            SwingTradingConfig with backtesting settings
        """
        return cls(
            trading=TradingConfig(
                fee_rate=0.0,
                slippage_rate=0.0,
                turnover_penalty=0.0,
            ),
            environment=EnvironmentConfig(
                random_start=False,
                episode_length=1000,
            ),
        )

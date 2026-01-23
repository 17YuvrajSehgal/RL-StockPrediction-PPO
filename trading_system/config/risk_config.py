"""
Risk management configuration for production trading.

This module provides comprehensive risk management settings including
position limits, drawdown controls, and circuit breakers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RiskConfig:
    """
    Risk management configuration for trading system.
    
    These settings enforce position limits, drawdown controls, and other
    risk constraints to protect capital and ensure regulatory compliance.
    
    Attributes:
        max_position_size: Maximum position size as fraction of equity (0.0-1.0)
        max_leverage: Maximum leverage allowed (1.0 = no leverage)
        max_drawdown_pct: Maximum drawdown before stopping (e.g., 0.20 = 20%)
        max_daily_loss_pct: Maximum daily loss before stopping
        max_concentration: Maximum concentration in single position
        enable_circuit_breakers: Enable automatic trading halts
        var_confidence: VaR confidence level (e.g., 0.95 = 95%)
        var_horizon_days: VaR time horizon in days
        kelly_fraction: Kelly criterion fraction (0.0-1.0, 0.25 = quarter Kelly)
        
    Example:
        >>> # Conservative institutional settings
        >>> config = RiskConfig.institutional()
        
        >>> # Aggressive hedge fund settings
        >>> config = RiskConfig.aggressive()
    """
    
    max_position_size: float = 1.0
    max_leverage: float = 1.0
    max_drawdown_pct: float = 0.20
    max_daily_loss_pct: float = 0.05
    max_concentration: float = 1.0
    enable_circuit_breakers: bool = True
    var_confidence: float = 0.95
    var_horizon_days: int = 1
    kelly_fraction: float = 0.25
    
    # Position sizing
    use_kelly_criterion: bool = False
    use_var_limits: bool = False
    
    # Monitoring
    alert_on_drawdown_pct: float = 0.10
    alert_on_daily_loss_pct: float = 0.03
    
    def __post_init__(self) -> None:
        """Validate risk parameters."""
        if not 0.0 < self.max_position_size <= 2.0:
            raise ValueError(
                f"max_position_size must be in (0, 2], got {self.max_position_size}"
            )
        if not 0.0 < self.max_leverage <= 10.0:
            raise ValueError(
                f"max_leverage must be in (0, 10], got {self.max_leverage}"
            )
        if not 0.0 < self.max_drawdown_pct <= 1.0:
            raise ValueError(
                f"max_drawdown_pct must be in (0, 1], got {self.max_drawdown_pct}"
            )
        if not 0.0 < self.max_daily_loss_pct <= 1.0:
            raise ValueError(
                f"max_daily_loss_pct must be in (0, 1], got {self.max_daily_loss_pct}"
            )
        if not 0.0 < self.max_concentration <= 1.0:
            raise ValueError(
                f"max_concentration must be in (0, 1], got {self.max_concentration}"
            )
        if not 0.0 < self.var_confidence < 1.0:
            raise ValueError(
                f"var_confidence must be in (0, 1), got {self.var_confidence}"
            )
        if self.var_horizon_days <= 0:
            raise ValueError(
                f"var_horizon_days must be positive, got {self.var_horizon_days}"
            )
        if not 0.0 <= self.kelly_fraction <= 1.0:
            raise ValueError(
                f"kelly_fraction must be in [0, 1], got {self.kelly_fraction}"
            )
    
    @classmethod
    def institutional(cls) -> RiskConfig:
        """
        Conservative risk settings for institutional investors.
        
        - Strict position limits
        - Low drawdown tolerance
        - Circuit breakers enabled
        - No leverage
        """
        return cls(
            max_position_size=0.5,
            max_leverage=1.0,
            max_drawdown_pct=0.15,
            max_daily_loss_pct=0.03,
            max_concentration=0.3,
            enable_circuit_breakers=True,
            kelly_fraction=0.1,
            alert_on_drawdown_pct=0.08,
            alert_on_daily_loss_pct=0.02,
        )
    
    @classmethod
    def aggressive(cls) -> RiskConfig:
        """
        Aggressive risk settings for hedge funds.
        
        - Higher position limits
        - Moderate drawdown tolerance
        - Leverage allowed
        - Kelly criterion enabled
        """
        return cls(
            max_position_size=1.5,
            max_leverage=2.0,
            max_drawdown_pct=0.30,
            max_daily_loss_pct=0.10,
            max_concentration=0.8,
            enable_circuit_breakers=True,
            use_kelly_criterion=True,
            kelly_fraction=0.25,
            alert_on_drawdown_pct=0.20,
            alert_on_daily_loss_pct=0.07,
        )
    
    @classmethod
    def research(cls) -> RiskConfig:
        """
        Permissive settings for research and backtesting.
        
        - Minimal constraints
        - Circuit breakers disabled
        - For simulation only, not production
        """
        return cls(
            max_position_size=2.0,
            max_leverage=5.0,
            max_drawdown_pct=0.50,
            max_daily_loss_pct=0.20,
            max_concentration=1.0,
            enable_circuit_breakers=False,
            kelly_fraction=0.5,
        )

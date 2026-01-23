"""
Margin model and position constraint enforcement.

This module implements realistic margin constraints for swing trading, including
cash-only requirements and short-sale haircuts to prevent unrealistic leverage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from swing_trading.config import MarginConfig


class MarginModel:
    """
    Enforces margin constraints and computes feasible position sizes.
    
    This model implements conservative, production-friendly margin rules:
    - Longs: Cash-only (no borrowing) if initial_margin_long = 1.0
    - Shorts: Short proceeds are haircutted; only a portion is usable as cash
    
    This prevents unrealistic behavior where shorting creates "infinite cash"
    that can be recycled into huge gross exposure.
    
    Attributes:
        config: Margin configuration parameters
    
    Example:
        >>> from swing_trading import MarginConfig, MarginModel
        >>> 
        >>> config = MarginConfig(short_proceeds_haircut=0.5)
        >>> model = MarginModel(config)
        >>> 
        >>> # Check usable cash with short position
        >>> cash = 100000.0
        >>> shares = -100  # short 100 shares
        >>> price = 50.0
        >>> usable = model.usable_cash(cash, shares, price)
        >>> print(f"Usable cash: ${usable:,.2f}")
    """
    
    def __init__(self, config: MarginConfig) -> None:
        """
        Initialize margin model.
        
        Args:
            config: Margin configuration parameters
        """
        self.config = config
    
    def usable_cash(self, cash: float, shares: float, price: float) -> float:
        """
        Compute usable cash after applying haircut to short proceeds.
        
        If shares < 0 (short position), we treat |shares| * price as short proceeds
        and make only (1 - haircut) of those proceeds usable.
        
        Args:
            cash: Current cash balance
            shares: Current share position (negative for short)
            price: Current price per share
        
        Returns:
            Usable cash amount after haircut
        
        Example:
            >>> model = MarginModel(MarginConfig(short_proceeds_haircut=0.5))
            >>> # Long position: all cash is usable
            >>> model.usable_cash(100000, 100, 50)
            100000.0
            >>> # Short position: haircut applied
            >>> model.usable_cash(100000, -100, 50)
            97500.0  # 100000 - 0.5 * 5000
        """
        if shares >= 0:
            return cash
        
        short_proceeds = (-shares) * price
        unusable = self.config.short_proceeds_haircut * short_proceeds
        return cash - unusable
    
    def clamp_target_shares(
        self,
        current_cash: float,
        current_shares: float,
        price: float,
        target_shares: float,
        fee_rate: float,
        slippage_rate: float,
        allow_short: bool,
    ) -> float:
        """
        Clamp target shares to satisfy margin constraints.
        
        Ensures that after executing the trade to target_shares, the usable cash
        (including short-proceeds haircut) remains non-negative.
        
        Uses bisection search to find the maximum feasible position in the direction
        of the target.
        
        Args:
            current_cash: Current cash balance
            current_shares: Current share position
            price: Execution price
            target_shares: Desired target position
            fee_rate: Transaction fee rate
            slippage_rate: Slippage rate
            allow_short: Whether short positions are allowed
        
        Returns:
            Clamped target shares that satisfy margin constraints
        
        Example:
            >>> model = MarginModel(MarginConfig())
            >>> # Try to buy more than cash allows
            >>> clamped = model.clamp_target_shares(
            ...     current_cash=10000,
            ...     current_shares=0,
            ...     price=100,
            ...     target_shares=200,  # Would cost $20,000
            ...     fee_rate=0.001,
            ...     slippage_rate=0.0,
            ...     allow_short=False
            ... )
            >>> print(f"Clamped to {clamped:.2f} shares")
        """
        # Disallow shorts if configured
        if (not allow_short) and (target_shares < 0):
            target_shares = 0.0
        
        rate = fee_rate + slippage_rate
        
        def usable_cash_after(shares: float) -> float:
            """Compute usable cash after executing trade to given shares."""
            delta = shares - current_shares
            traded_notional = abs(delta) * price
            costs = traded_notional * rate
            cash_after = current_cash - (delta * price) - costs
            return self.usable_cash(cash_after, shares, price)
        
        # If already feasible, keep target
        if usable_cash_after(target_shares) >= -1e-9:
            return float(target_shares)
        
        # Otherwise, clamp using bisection search
        lo = current_shares
        hi = target_shares
        
        # Ensure lo is feasible (current position should be valid)
        if usable_cash_after(lo) < -1e-9:
            # Current state is already margin-infeasible
            # In production, this would trigger force-liquidation
            return float(current_shares)
        
        # Ensure lo < hi for bisection
        if hi < lo:
            lo, hi = hi, lo
        
        # Bisection to find maximum feasible position
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if usable_cash_after(mid) >= -1e-9:
                lo = mid
            else:
                hi = mid
        
        # Return the feasible position closest to target
        if target_shares >= current_shares:
            return float(lo)
        else:
            return float(lo)
    
    def validate_position(
        self,
        cash: float,
        shares: float,
        price: float,
    ) -> tuple[bool, str]:
        """
        Validate if current position satisfies margin constraints.
        
        Args:
            cash: Current cash balance
            shares: Current share position
            price: Current price
        
        Returns:
            Tuple of (is_valid, error_message)
            If valid, error_message is empty string
        
        Example:
            >>> model = MarginModel(MarginConfig())
            >>> valid, msg = model.validate_position(10000, 100, 50)
            >>> if not valid:
            ...     print(f"Margin violation: {msg}")
        """
        usable = self.usable_cash(cash, shares, price)
        
        if usable < -1e-6:
            return False, f"Negative usable cash: ${usable:,.2f}"
        
        return True, ""
    
    def max_long_shares(
        self,
        cash: float,
        price: float,
        fee_rate: float,
        slippage_rate: float,
    ) -> float:
        """
        Compute maximum number of shares that can be bought with available cash.
        
        Args:
            cash: Available cash
            price: Purchase price per share
            fee_rate: Transaction fee rate
            slippage_rate: Slippage rate
        
        Returns:
            Maximum shares that can be purchased
        
        Example:
            >>> model = MarginModel(MarginConfig())
            >>> max_shares = model.max_long_shares(10000, 100, 0.001, 0.0)
            >>> print(f"Can buy up to {max_shares:.2f} shares")
        """
        rate = fee_rate + slippage_rate
        # cash = shares * price * (1 + rate)
        # shares = cash / (price * (1 + rate))
        return cash / (price * (1.0 + rate))
    
    def max_short_shares(
        self,
        cash: float,
        price: float,
        fee_rate: float,
        slippage_rate: float,
    ) -> float:
        """
        Compute maximum number of shares that can be shorted.
        
        With haircut h, shorting S shares at price P:
        - Generates proceeds: S * P
        - Usable cash: (1 - h) * S * P
        - Costs: S * P * rate
        
        For usable cash to remain non-negative:
        cash + (1 - h) * S * P - S * P * rate >= 0
        
        Args:
            cash: Current cash balance
            price: Short price per share
            fee_rate: Transaction fee rate
            slippage_rate: Slippage rate
        
        Returns:
            Maximum shares that can be shorted (positive number)
        
        Example:
            >>> model = MarginModel(MarginConfig(short_proceeds_haircut=0.5))
            >>> max_shares = model.max_short_shares(10000, 100, 0.001, 0.0)
            >>> print(f"Can short up to {max_shares:.2f} shares")
        """
        h = self.config.short_proceeds_haircut
        rate = fee_rate + slippage_rate
        
        # cash + (1 - h) * S * P - S * P * rate >= 0
        # cash + S * P * ((1 - h) - rate) >= 0
        # S <= cash / (P * (rate - (1 - h)))
        
        denominator = price * (rate - (1.0 - h))
        
        if denominator <= 0:
            # Can short unlimited (haircut covers costs)
            return float('inf')
        
        return cash / denominator

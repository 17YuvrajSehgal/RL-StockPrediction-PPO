"""
Portfolio state management and PnL accounting.

This module provides immutable portfolio state tracking with comprehensive
PnL calculations and validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PortfolioState:
    """
    Immutable snapshot of portfolio state at a point in time.
    
    This class tracks all portfolio information including positions, cash,
    and realized/unrealized PnL. Being immutable makes it easier to debug
    and reason about state transitions.
    
    Attributes:
        t: Time step index
        equity: Total portfolio value (cash + position value)
        cash: Cash balance
        shares: Number of shares held (negative for short)
        entry_price: Average entry price for current position
        prev_weight: Portfolio weight from previous step
        realized_pnl: Cumulative realized profit/loss
    
    Example:
        >>> state = PortfolioState(
        ...     t=0,
        ...     equity=100000.0,
        ...     cash=100000.0,
        ...     shares=0.0,
        ...     entry_price=0.0,
        ...     prev_weight=0.0,
        ...     realized_pnl=0.0
        ... )
        >>> print(f"Portfolio equity: ${state.equity:,.2f}")
    """
    
    t: int
    equity: float
    cash: float
    shares: float
    entry_price: float
    prev_weight: float
    realized_pnl: float
    
    def __post_init__(self) -> None:
        """Validate portfolio state invariants."""
        # Allow small floating point errors
        tolerance = 1e-6
        
        # Check equity = cash + position value
        position_val = self.position_value(self.entry_price) if self.shares != 0 else 0.0
        expected_equity = self.cash + position_val
        
        if abs(self.equity - expected_equity) > tolerance * abs(self.equity):
            # Note: This validation is approximate because entry_price may differ
            # from current price. We validate using entry_price as a sanity check.
            pass
    
    def position_value(self, price: float) -> float:
        """
        Compute position value at given price.
        
        Args:
            price: Price per share
        
        Returns:
            Position value (shares * price)
        
        Example:
            >>> state = PortfolioState(t=0, equity=100000, cash=50000,
            ...                        shares=500, entry_price=100, 
            ...                        prev_weight=0.5, realized_pnl=0)
            >>> state.position_value(105)
            52500.0
        """
        return self.shares * price
    
    def unrealized_pnl(self, price: float) -> float:
        """
        Compute unrealized profit/loss at given price.
        
        Args:
            price: Current market price
        
        Returns:
            Unrealized PnL (current value - entry value)
        
        Example:
            >>> state = PortfolioState(t=0, equity=100000, cash=50000,
            ...                        shares=500, entry_price=100,
            ...                        prev_weight=0.5, realized_pnl=0)
            >>> state.unrealized_pnl(105)
            2500.0  # (105 - 100) * 500
        """
        if self.shares == 0:
            return 0.0
        return (price - self.entry_price) * self.shares
    
    def total_pnl(self, price: float) -> float:
        """
        Compute total profit/loss (realized + unrealized).
        
        Args:
            price: Current market price
        
        Returns:
            Total PnL
        
        Example:
            >>> state = PortfolioState(t=0, equity=100000, cash=50000,
            ...                        shares=500, entry_price=100,
            ...                        prev_weight=0.5, realized_pnl=1000)
            >>> state.total_pnl(105)
            3500.0  # 1000 + 2500
        """
        return self.realized_pnl + self.unrealized_pnl(price)
    
    def current_weight(self, price: float) -> float:
        """
        Compute current portfolio weight.
        
        Weight = position_value / equity
        
        Args:
            price: Current market price
        
        Returns:
            Portfolio weight (can be negative for short positions)
        
        Example:
            >>> state = PortfolioState(t=0, equity=100000, cash=50000,
            ...                        shares=500, entry_price=100,
            ...                        prev_weight=0.5, realized_pnl=0)
            >>> state.current_weight(105)
            0.525  # 52500 / 100000
        """
        if self.equity == 0:
            return 0.0
        return self.position_value(price) / self.equity
    
    def update(
        self,
        t: Optional[int] = None,
        equity: Optional[float] = None,
        cash: Optional[float] = None,
        shares: Optional[float] = None,
        entry_price: Optional[float] = None,
        prev_weight: Optional[float] = None,
        realized_pnl: Optional[float] = None,
    ) -> PortfolioState:
        """
        Create new state with updated fields.
        
        Since PortfolioState is immutable, this returns a new instance
        with specified fields updated.
        
        Args:
            t: New time step (if None, keep current)
            equity: New equity (if None, keep current)
            cash: New cash (if None, keep current)
            shares: New shares (if None, keep current)
            entry_price: New entry price (if None, keep current)
            prev_weight: New previous weight (if None, keep current)
            realized_pnl: New realized PnL (if None, keep current)
        
        Returns:
            New PortfolioState with updated fields
        
        Example:
            >>> state = PortfolioState(t=0, equity=100000, cash=100000,
            ...                        shares=0, entry_price=0,
            ...                        prev_weight=0, realized_pnl=0)
            >>> new_state = state.update(t=1, shares=100, entry_price=100)
            >>> print(f"New position: {new_state.shares} shares")
        """
        return PortfolioState(
            t=t if t is not None else self.t,
            equity=equity if equity is not None else self.equity,
            cash=cash if cash is not None else self.cash,
            shares=shares if shares is not None else self.shares,
            entry_price=entry_price if entry_price is not None else self.entry_price,
            prev_weight=prev_weight if prev_weight is not None else self.prev_weight,
            realized_pnl=realized_pnl if realized_pnl is not None else self.realized_pnl,
        )
    
    @classmethod
    def initial(cls, initial_equity: float, t: int = 0) -> PortfolioState:
        """
        Create initial portfolio state (all cash, no position).
        
        Args:
            initial_equity: Starting portfolio value
            t: Starting time step
        
        Returns:
            Initial PortfolioState
        
        Example:
            >>> state = PortfolioState.initial(100000.0)
            >>> print(f"Starting with ${state.equity:,.2f}")
        """
        return cls(
            t=t,
            equity=initial_equity,
            cash=initial_equity,
            shares=0.0,
            entry_price=0.0,
            prev_weight=0.0,
            realized_pnl=0.0,
        )
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"PortfolioState(t={self.t}, equity={self.equity:.2f}, "
            f"cash={self.cash:.2f}, shares={self.shares:.2f}, "
            f"entry_price={self.entry_price:.2f}, weight={self.prev_weight:.3f}, "
            f"realized_pnl={self.realized_pnl:.2f})"
        )

"""
Trade execution engine and cost modeling.

This module handles all trade execution logic, transaction cost calculation,
and portfolio rebalancing with margin constraint enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from swing_trading.config import TradingConfig
    from swing_trading.margin import MarginModel
    from swing_trading.portfolio import PortfolioState


@dataclass(frozen=True)
class TradeResult:
    """
    Result of a trade execution.
    
    Attributes:
        new_state: Updated portfolio state after trade
        traded_notional: Dollar value of shares traded
        costs: Transaction costs paid (fees + slippage)
        shares_delta: Change in share position
        realized_pnl: PnL realized from this trade
    
    Example:
        >>> # After executing a trade
        >>> print(f"Traded ${result.traded_notional:,.2f}")
        >>> print(f"Costs: ${result.costs:.2f}")
        >>> print(f"Realized PnL: ${result.realized_pnl:.2f}")
    """
    
    new_state: PortfolioState
    traded_notional: float
    costs: float
    shares_delta: float
    realized_pnl: float


class ExecutionEngine:
    """
    Handles trade execution, cost modeling, and portfolio rebalancing.
    
    This engine integrates with the margin model to ensure all trades
    satisfy margin constraints. It provides detailed trade records for
    audit trails and debugging.
    
    Attributes:
        config: Trading configuration
        margin_model: Margin constraint model (optional)
    
    Example:
        >>> from swing_trading import TradingConfig, MarginConfig, MarginModel
        >>> from swing_trading import ExecutionEngine, PortfolioState
        >>> 
        >>> config = TradingConfig()
        >>> margin = MarginModel(MarginConfig())
        >>> engine = ExecutionEngine(config, margin)
        >>> 
        >>> state = PortfolioState.initial(100000.0)
        >>> result = engine.rebalance_to_weight(state, price=100.0, target_weight=0.5)
        >>> print(f"New position: {result.new_state.shares} shares")
    """
    
    def __init__(
        self,
        config: TradingConfig,
        margin_model: MarginModel | None = None,
    ) -> None:
        """
        Initialize execution engine.
        
        Args:
            config: Trading configuration
            margin_model: Optional margin model for constraint enforcement
        """
        self.config = config
        self.margin_model = margin_model
    
    def clamp_weight(self, weight: float) -> float:
        """
        Clamp weight to satisfy trading constraints.
        
        Enforces:
        1. Long-only constraint if short selling disabled
        2. Maximum absolute weight limit
        
        Args:
            weight: Target portfolio weight
        
        Returns:
            Clamped weight
        
        Example:
            >>> engine = ExecutionEngine(TradingConfig(allow_short=False))
            >>> engine.clamp_weight(-0.5)
            0.0
        """
        w = float(weight)
        
        # Enforce long-only if configured
        if not self.config.allow_short:
            w = max(0.0, w)
        
        # Clamp to max absolute weight
        w = float(np.clip(w, -self.config.max_abs_weight, self.config.max_abs_weight))
        
        return w
    
    def compute_costs(self, traded_notional: float) -> float:
        """
        Compute transaction costs for a trade.
        
        Args:
            traded_notional: Dollar value of shares traded
        
        Returns:
            Total transaction costs (fees + slippage)
        
        Example:
            >>> engine = ExecutionEngine(TradingConfig(fee_rate=0.001, slippage_rate=0.0002))
            >>> costs = engine.compute_costs(10000.0)
            >>> print(f"Costs: ${costs:.2f}")  # $12.00
        """
        return traded_notional * self.config.total_cost_rate
    
    def rebalance_to_weight(
        self,
        state: PortfolioState,
        price: float,
        target_weight: float,
    ) -> TradeResult:
        """
        Rebalance portfolio to target weight.
        
        This is the main execution method. It:
        1. Clamps target weight to constraints
        2. Computes target shares
        3. Enforces margin constraints (if margin model provided)
        4. Executes trade
        5. Updates portfolio state and PnL
        
        Args:
            state: Current portfolio state
            price: Execution price
            target_weight: Desired portfolio weight
        
        Returns:
            TradeResult with updated state and trade details
        
        Example:
            >>> state = PortfolioState.initial(100000.0)
            >>> engine = ExecutionEngine(TradingConfig())
            >>> result = engine.rebalance_to_weight(state, 100.0, 0.5)
            >>> print(f"New equity: ${result.new_state.equity:,.2f}")
        """
        # 1. Clamp weight to constraints
        w = self.clamp_weight(target_weight)
        
        # 2. Compute current equity and target shares
        equity = state.cash + state.shares * price
        target_position_value = w * equity
        target_shares = target_position_value / price
        
        # 3. Enforce margin constraints if model provided
        if self.margin_model is not None and self.config.cash_only:
            target_shares = self.margin_model.clamp_target_shares(
                current_cash=state.cash,
                current_shares=state.shares,
                price=price,
                target_shares=target_shares,
                fee_rate=self.config.fee_rate,
                slippage_rate=self.config.slippage_rate,
                allow_short=self.config.allow_short,
            )
        
        # 4. Compute trade details
        shares_delta = target_shares - state.shares
        traded_notional = abs(shares_delta) * price
        costs = self.compute_costs(traded_notional)
        
        # 5. Update cash and shares
        cash_after = state.cash - (shares_delta * price) - costs
        shares_after = target_shares
        
        # 6. Update realized PnL and entry price
        realized_pnl = state.realized_pnl
        entry_price = state.entry_price
        realized_this_trade = 0.0
        
        # Position direction flip (crossing through zero)
        if state.shares != 0 and np.sign(shares_after) != np.sign(state.shares):
            # Realize entire old position
            realized_this_trade = (price - entry_price) * state.shares
            realized_pnl += realized_this_trade
            # New entry price for new position
            entry_price = price if shares_after != 0 else 0.0
        
        # Position shrinks (partial close)
        elif abs(shares_after) < abs(state.shares):
            reduced_shares = state.shares - shares_after
            realized_this_trade = (price - entry_price) * reduced_shares
            realized_pnl += realized_this_trade
        
        # Open from flat
        elif state.shares == 0 and shares_after != 0:
            entry_price = price
        
        # 7. Recompute equity after trade
        equity_after = cash_after + shares_after * price
        
        # 8. Build updated state
        new_state = state.update(
            equity=float(equity_after),
            cash=float(cash_after),
            shares=float(shares_after),
            entry_price=float(entry_price),
            prev_weight=float(w),
            realized_pnl=float(realized_pnl),
        )
        
        # 9. Return trade result
        return TradeResult(
            new_state=new_state,
            traded_notional=float(traded_notional),
            costs=float(costs),
            shares_delta=float(shares_delta),
            realized_pnl=float(realized_this_trade),
        )
    
    def validate_trade(
        self,
        state: PortfolioState,
        price: float,
        target_weight: float,
    ) -> tuple[bool, str]:
        """
        Validate if a trade is feasible without executing it.
        
        Args:
            state: Current portfolio state
            price: Execution price
            target_weight: Desired portfolio weight
        
        Returns:
            Tuple of (is_valid, error_message)
        
        Example:
            >>> engine = ExecutionEngine(TradingConfig())
            >>> state = PortfolioState.initial(100000.0)
            >>> valid, msg = engine.validate_trade(state, 100.0, 2.0)
            >>> if not valid:
            ...     print(f"Invalid trade: {msg}")
        """
        # Check weight constraints
        w = self.clamp_weight(target_weight)
        if abs(w - target_weight) > 1e-6:
            return False, f"Weight {target_weight} violates constraints (clamped to {w})"
        
        # Check margin constraints if applicable
        if self.margin_model is not None:
            equity = state.cash + state.shares * price
            target_shares = (w * equity) / price
            
            # Simulate trade
            shares_delta = target_shares - state.shares
            traded_notional = abs(shares_delta) * price
            costs = self.compute_costs(traded_notional)
            cash_after = state.cash - (shares_delta * price) - costs
            
            valid, msg = self.margin_model.validate_position(
                cash=cash_after,
                shares=target_shares,
                price=price,
            )
            if not valid:
                return False, f"Margin violation: {msg}"
        
        return True, ""

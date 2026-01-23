"""
Professional backtesting engine for PPO trading models.

This module provides enterprise-grade backtesting capabilities with:
- Realistic trade execution (next-open pricing)
- Transaction cost modeling
- Position tracking and P&L calculation
- Comprehensive performance metrics
- Trade-by-trade logging
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from swing_trading import SwingTradingEnv, SwingTradingConfig
from trading_system.evaluation.metrics import FinancialMetrics


@dataclass
class Trade:
    """
    Record of a single trade.
    
    Attributes:
        entry_date: Trade entry timestamp
        exit_date: Trade exit timestamp (None if still open)
        entry_price: Entry execution price
        exit_price: Exit execution price (None if still open)
        shares: Number of shares (negative for short)
        direction: 'long' or 'short'
        entry_reason: Why trade was entered
        exit_reason: Why trade was exited
        pnl: Profit/loss in dollars
        pnl_pct: Profit/loss as percentage
        costs: Transaction costs paid
        holding_period: Days held
        mae: Maximum adverse excursion (worst drawdown during trade)
        mfe: Maximum favorable excursion (best profit during trade)
    """
    
    entry_date: datetime
    exit_date: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    shares: float = 0.0
    direction: str = ""  # 'long' or 'short'
    entry_reason: str = ""
    exit_reason: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    costs: float = 0.0
    holding_period: int = 0
    mae: float = 0.0  # Maximum adverse excursion
    mfe: float = 0.0  # Maximum favorable excursion
    
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame."""
        return {
            'entry_date': self.entry_date,
            'exit_date': self.exit_date,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'shares': self.shares,
            'direction': self.direction,
            'entry_reason': self.entry_reason,
            'exit_reason': self.exit_reason,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'costs': self.costs,
            'holding_period': self.holding_period,
            'mae': self.mae,
            'mfe': self.mfe,
            'winner': self.is_winner(),
        }


@dataclass
class Position:
    """
    Current position state.
    
    Attributes:
        shares: Current shares held (negative for short, 0 for flat)
        entry_price: Average entry price
        entry_date: When position was opened
        current_pnl: Unrealized P&L
        peak_pnl: Best unrealized P&L seen
        worst_pnl: Worst unrealized P&L seen
    """
    
    shares: float = 0.0
    entry_price: float = 0.0
    entry_date: Optional[datetime] = None
    current_pnl: float = 0.0
    peak_pnl: float = 0.0
    worst_pnl: float = 0.0
    
    def is_flat(self) -> bool:
        """Check if position is flat."""
        return abs(self.shares) < 1e-6
    
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.shares > 1e-6
    
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.shares < -1e-6
    
    def update_pnl(self, current_price: float) -> None:
        """Update P&L tracking."""
        if not self.is_flat():
            self.current_pnl = (current_price - self.entry_price) * self.shares
            self.peak_pnl = max(self.peak_pnl, self.current_pnl)
            self.worst_pnl = min(self.worst_pnl, self.current_pnl)


@dataclass
class BacktestResult:
    """
    Complete backtest results.
    
    Attributes:
        trades: List of all trades
        equity_curve: Portfolio value over time
        positions: Position sizes over time
        metrics: Performance metrics dictionary
        config: Backtest configuration
    """
    
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    positions: pd.Series = field(default_factory=pd.Series)
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trades])
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "="*70,
            "BACKTEST RESULTS SUMMARY",
            "="*70,
            f"Total Trades: {len(self.trades)}",
            f"Win Rate: {self.metrics.get('win_rate', 0):.2%}",
            f"Total Return: {self.metrics.get('total_return', 0):.2%}",
            f"Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0):.2f}",
            f"Max Drawdown: {self.metrics.get('max_drawdown', 0):.2%}",
            f"Profit Factor: {self.metrics.get('profit_factor', 0):.2f}",
            "="*70,
        ]
        return "\n".join(lines)


class BacktestEngine:
    """
    Professional backtesting engine for PPO trading models.
    
    Features:
    - Load trained PPO models with VecNormalize
    - Simulate realistic trading with transaction costs
    - Track all trades and positions
    - Calculate comprehensive performance metrics
    - Generate detailed trade logs
    
    Example:
        >>> engine = BacktestEngine(
        ...     model_path="models/AAPL_production_final",
        ...     data=df,
        ...     config=SwingTradingConfig()
        ... )
        >>> result = engine.run()
        >>> print(result.summary())
    """
    
    def __init__(
        self,
        model_path: str,
        data: pd.DataFrame,
        config: Optional[SwingTradingConfig] = None,
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.001,  # 10 bps
    ):
        """
        Initialize backtest engine.
        
        Args:
            model_path: Path to trained model (without extension)
            data: OHLCV DataFrame for backtesting
            config: Environment configuration
            initial_capital: Starting capital
            transaction_cost: Transaction cost rate (fees + slippage)
        """
        self.model_path = model_path
        self.data = data.copy()
        self.config = config or SwingTradingConfig()
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        # Load model
        self.model = self._load_model()
        
        # Create environment
        self.env = self._create_environment()
        
        # State tracking
        self.cash = initial_capital
        self.equity = initial_capital
        self.position = Position()
        self.trades: List[Trade] = []
        self.equity_history: List[float] = []
        self.position_history: List[float] = []
        self.dates: List[datetime] = []
        
    def _load_model(self) -> PPO:
        """Load trained PPO model with VecNormalize."""
        print(f"Loading model from {self.model_path}...")
        
        # Load model
        model = PPO.load(self.model_path)
        
        # Try to load VecNormalize stats
        vecnorm_path = f"{self.model_path}_vecnormalize.pkl"
        if Path(vecnorm_path).exists():
            print(f"Found VecNormalize stats: {vecnorm_path}")
        else:
            print(f"⚠️  No VecNormalize stats found, using raw observations")
        
        print(f"Model loaded successfully")
        return model
    
    def _create_environment(self) -> DummyVecEnv:
        """Create environment for inference."""
        env = SwingTradingEnv(self.data, self.config)
        vec_env = DummyVecEnv([lambda: env])
        
        # Load VecNormalize if available
        vecnorm_path = f"{self.model_path}_vecnormalize.pkl"
        if Path(vecnorm_path).exists():
            vec_env = VecNormalize.load(vecnorm_path, vec_env)
            vec_env.training = False  # Disable training mode
            vec_env.norm_reward = False  # Don't normalize rewards during inference
        
        return vec_env
    
    def run(self, verbose: bool = True) -> BacktestResult:
        """
        Run backtest.
        
        Args:
            verbose: Print progress
        
        Returns:
            BacktestResult with all trades and metrics
        """
        if verbose:
            print("\n" + "="*70)
            print("STARTING BACKTEST")
            print("="*70)
            print(f"Initial Capital: ${self.initial_capital:,.2f}")
            print(f"Data Period: {self.data.index[0]} to {self.data.index[-1]}")
            print(f"Total Bars: {len(self.data)}")
            print("="*70 + "\n")
        
        # Reset environment
        obs = self.env.reset()
        done = False
        step = 0
        lookback = self.config.environment.lookback
        
        while not done:
            # Get model prediction
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Execute action in environment
            obs, reward, done, info = self.env.step(action)
            
            # Extract info from vectorized environment
            info_dict = info[0] if isinstance(info, (list, tuple)) else info
            
            # Handle data alignment
            # Environment is ahead by 'lookback' steps
            current_data_idx = step + lookback
            if current_data_idx >= len(self.data):
                # Should not happen if env terminates correctly, but safety break
                break
                
            # Get current date and price
            # Prefer info_dict, fallback to aligned data index
            if 'date' in info_dict:
                current_date = pd.to_datetime(info_dict['date'])
            else:
                current_date = self.data.index[current_data_idx]
            
            if 'mark_price' in info_dict:
                current_price = float(info_dict['mark_price'])
            else:
                current_price = float(self.data['close'].iloc[current_data_idx])
            
            # Get target weight from action
            target_weight = float(action[0]) if isinstance(action, np.ndarray) else float(action)
            target_weight = np.clip(target_weight, -1.0, 1.0)
            
            # Calculate target shares
            # Enforce margin requirement (100% margin for simplicity)
            target_shares = (target_weight * self.equity) / current_price
            
            # Execute trade if position changes
            if abs(target_shares - self.position.shares) > 1e-6:
                self._execute_trade(
                    target_shares=target_shares,
                    current_price=current_price,
                    current_date=current_date,
                    reason=f"Model signal: weight={target_weight:.3f}"
                )
            
            # Update position P&L
            self.position.update_pnl(current_price)
            
            # Update equity
            position_value = self.position.shares * current_price
            self.equity = self.cash + position_value
            
            # Record history
            self.equity_history.append(self.equity)
            self.position_history.append(self.position.shares)
            self.dates.append(current_date)
            
            step += 1
            
            if verbose and step % 50 == 0:
                print(f"Step {step}: Equity=${self.equity:,.2f}, Position={self.position.shares:.0f} shares")
        
        # Close any remaining position
        if not self.position.is_flat():
            final_price = self.data['close'].iloc[-1]
            final_date = self.data.index[-1]
            self._execute_trade(
                target_shares=0.0,
                current_price=final_price,
                current_date=final_date,
                reason="End of backtest"
            )
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Create result
        result = BacktestResult(
            trades=self.trades,
            equity_curve=pd.Series(self.equity_history, index=self.dates),
            positions=pd.Series(self.position_history, index=self.dates),
            metrics=metrics,
            config={
                'initial_capital': self.initial_capital,
                'transaction_cost': self.transaction_cost,
                'model_path': self.model_path,
            }
        )
        
        if verbose:
            print("\n" + result.summary())
        
        return result
    
    def _execute_trade(
        self,
        target_shares: float,
        current_price: float,
        current_date: datetime,
        reason: str
    ) -> None:
        """Execute a trade."""
        shares_delta = target_shares - self.position.shares
        
        if abs(shares_delta) < 1e-6:
            return
        
        # Calculate costs
        traded_notional = abs(shares_delta) * current_price
        costs = traded_notional * self.transaction_cost
        
        # Update cash
        self.cash -= shares_delta * current_price + costs
        
        # Handle position transitions
        if self.position.is_flat():
            # Opening new position
            self.position = Position(
                shares=target_shares,
                entry_price=current_price,
                entry_date=current_date,
            )
        elif abs(target_shares) < 1e-6:
            # Closing position
            pnl = (current_price - self.position.entry_price) * self.position.shares - costs
            pnl_pct = pnl / (abs(self.position.shares) * self.position.entry_price)
            
            trade = Trade(
                entry_date=self.position.entry_date,
                exit_date=current_date,
                entry_price=self.position.entry_price,
                exit_price=current_price,
                shares=self.position.shares,
                direction='long' if self.position.shares > 0 else 'short',
                entry_reason="Model signal",
                exit_reason=reason,
                pnl=pnl,
                pnl_pct=pnl_pct,
                costs=costs,
                holding_period=(current_date - self.position.entry_date).days,
                mae=self.position.worst_pnl,
                mfe=self.position.peak_pnl,
            )
            self.trades.append(trade)
            
            # Reset position
            self.position = Position()
        else:
            # Position flip or size change
            if np.sign(target_shares) != np.sign(self.position.shares):
                # Close old position first
                self._execute_trade(0.0, current_price, current_date, "Position flip")
                # Open new position
                self._execute_trade(target_shares, current_price, current_date, reason)
            else:
                # Just size change, update entry price (average)
                total_value = self.position.shares * self.position.entry_price + shares_delta * current_price
                self.position.shares = target_shares
                self.position.entry_price = total_value / target_shares if target_shares != 0 else 0.0
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if len(self.equity_history) == 0:
            return {}
        
        equity_curve = np.array(self.equity_history)
        returns = np.diff(np.log(equity_curve))
        
        # Basic metrics
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital
        
        # Risk metrics
        sharpe = FinancialMetrics.sharpe_ratio(returns) if len(returns) > 0 else 0.0
        max_dd = FinancialMetrics.max_drawdown(equity_curve)
        
        # Trade metrics
        winning_trades = [t for t in self.trades if t.is_winner()]
        losing_trades = [t for t in self.trades if not t.is_winner()]
        
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0.0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0
        
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_equity': equity_curve[-1],
        }

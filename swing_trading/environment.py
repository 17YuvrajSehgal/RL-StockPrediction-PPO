"""
Gymnasium-compatible swing trading environment.

This module provides the main RL environment that integrates all components:
feature engineering, execution, margin constraints, and portfolio management.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from swing_trading.config import SwingTradingConfig
from swing_trading.execution import ExecutionEngine
from swing_trading.features import FeatureEngineer
from swing_trading.margin import MarginModel
from swing_trading.portfolio import PortfolioState


class SwingTradingEnv(gym.Env):
    """
    Gymnasium-compatible environment for swing trading with continuous actions.
    
    This environment supports:
    - Continuous action space: target portfolio weight ∈ [-1, 1]
    - Realistic execution: actions execute at next bar's open price
    - Transaction costs: fees and slippage
    - Margin constraints: cash-only with short haircuts
    - Feature engineering: technical indicators from OHLCV data
    - Flexible observations: market features + optional portfolio state
    
    Attributes:
        config: Environment configuration
        df: OHLCV DataFrame
        features: Computed feature DataFrame
        action_space: Continuous Box space [-1, 1]
        observation_space: Box space for features
    
    Example:
        >>> import pandas as pd
        >>> from swing_trading import SwingTradingEnv, SwingTradingConfig
        >>> 
        >>> df = pd.read_csv("AAPL.csv", parse_dates=['Date'], index_col='Date')
        >>> config = SwingTradingConfig()
        >>> env = SwingTradingEnv(df, config)
        >>> 
        >>> obs, info = env.reset()
        >>> action = 0.5  # Target 50% long
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        df_daily_ohlcv: pd.DataFrame,
        config: Optional[SwingTradingConfig] = None,
        feature_engineer: Optional[FeatureEngineer] = None,
    ) -> None:
        """
        Initialize swing trading environment.
        
        Args:
            df_daily_ohlcv: DataFrame with OHLCV data (columns: open, high, low, close, volume)
            config: Environment configuration (uses defaults if None)
            feature_engineer: Custom feature engineer (creates default if None)
        
        Raises:
            ValueError: If data is insufficient for lookback + episode length
        """
        super().__init__()
        
        # Store config
        self.config = config or SwingTradingConfig()
        self.env_cfg = self.config.environment
        self.trading_cfg = self.config.trading
        self.margin_cfg = self.config.margin
        
        # Validate and store OHLCV data
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df_daily_ohlcv.columns:
                raise ValueError(f"Missing required column: {col}")
        
        self.df = df_daily_ohlcv.copy().sort_index()
        
        # Build features
        self.fe = feature_engineer or FeatureEngineer(self.df)
        self.features = self.fe.build()
        
        # Align OHLCV to feature index (features drop initial NaNs)
        self.df = self.df.loc[self.features.index]
        
        # Initialize components
        self.margin_model = MarginModel(self.margin_cfg)
        self.exec_engine = ExecutionEngine(self.trading_cfg, self.margin_model)
        
        # Feature metadata
        self.feature_cols = list(self.features.columns)
        self.n_market_features = len(self.feature_cols)
        self.n_portfolio_features = 6 if self.env_cfg.include_portfolio_features else 0
        self.n_total_features = self.n_market_features + self.n_portfolio_features
        
        # Define action and observation spaces
        # Action: continuous weight in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
        
        # Observation: (lookback, n_features)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.env_cfg.lookback, self.n_total_features),
            dtype=np.float32,
        )
        
        # Episode state
        self.rng = np.random.default_rng(self.env_cfg.seed)
        self._start: Optional[int] = None
        self._end: Optional[int] = None
        self._t: Optional[int] = None
        self.state: Optional[PortfolioState] = None
        self._prev_equity: Optional[float] = None
        
        # Validate data length
        min_length = self.env_cfg.lookback + self.env_cfg.episode_length + 1
        if len(self.features) < min_length:
            raise ValueError(
                f"Insufficient data: need {min_length} rows, got {len(self.features)}"
            )
    
    def _get_obs(self) -> np.ndarray:
        """
        Construct observation at current time step.
        
        Returns:
            Observation array of shape (lookback, n_features)
        """
        L = self.env_cfg.lookback
        t = self._t
        
        # Market features window
        market_window = self.features.iloc[t - L + 1 : t + 1].to_numpy(dtype=np.float32)
        
        if market_window.shape[0] != L:
            raise RuntimeError(
                f"Observation window wrong length: expected {L}, got {market_window.shape[0]}"
            )
        
        if not self.env_cfg.include_portfolio_features:
            return market_window
        
        # Portfolio features (repeated across lookback window)
        px = self._current_price()
        equity = max(self.state.equity, 1e-12)
        
        portfolio_vec = np.array([
            self.state.cash / self.env_cfg.initial_equity,
            self.state.shares,
            self.state.entry_price if self.state.shares != 0 else 0.0,
            self.state.unrealized_pnl(px) / equity,
            self.state.realized_pnl / equity,
            self.state.prev_weight,
        ], dtype=np.float32)
        
        # Repeat portfolio features across time dimension
        portfolio_block = np.repeat(
            portfolio_vec.reshape(1, -1),
            repeats=L,
            axis=0,
        )
        
        # Concatenate market and portfolio features
        return np.concatenate([market_window, portfolio_block], axis=1)
    
    def _current_price(self) -> float:
        """Get current close price for mark-to-market."""
        return float(self.df['close'].iloc[self._t])
    
    def _next_open_price(self) -> float:
        """Get next bar's open price for execution."""
        return float(self.df['open'].iloc[self._t + 1])
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset environment to start of new episode.
        
        Args:
            seed: Random seed for episode
            options: Additional options (unused)
        
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        L = self.env_cfg.lookback
        ep_len = self.env_cfg.episode_length
        
        # Determine valid episode range
        # Need: lookback window + episode + 1 for next-open execution
        max_t = len(self.features) - 2
        min_t = L - 1
        
        if (max_t - min_t + 1) < ep_len:
            raise ValueError(
                f"Insufficient data for episode: need {ep_len} steps, "
                f"have {max_t - min_t + 1}"
            )
        
        # Select episode start
        if self.env_cfg.random_start:
            self._start = int(self.rng.integers(low=min_t, high=(max_t - ep_len + 2)))
        else:
            self._start = min_t
        
        self._end = self._start + ep_len - 1
        self._t = self._start
        
        # Initialize portfolio
        self.state = PortfolioState.initial(
            initial_equity=self.env_cfg.initial_equity,
            t=self._t,
        )
        self._prev_equity = self.env_cfg.initial_equity
        
        obs = self._get_obs()
        info = {
            'date': str(self.features.index[self._t]),
            'equity': self.state.equity,
            'start_idx': self._start,
            'end_idx': self._end,
        }
        
        return obs, info
    
    def step(
        self,
        action: np.ndarray | float,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one environment step.
        
        Process:
        1. Parse action as target weight
        2. Execute rebalance at next bar's open price
        3. Advance time by 1 day
        4. Mark-to-market at new close price
        5. Compute reward (log return - turnover penalty)
        6. Check termination
        
        Args:
            action: Target portfolio weight (scalar or array)
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.state is None:
            raise RuntimeError("Call reset() before step()")
        
        # Parse action
        if isinstance(action, (list, tuple, np.ndarray)):
            target_weight = float(np.asarray(action).reshape(-1)[0])
        else:
            target_weight = float(action)
        
        # Execute rebalance at next open
        exec_price = self._next_open_price()
        
        result = self.exec_engine.rebalance_to_weight(
            state=self.state,
            price=exec_price,
            target_weight=target_weight,
        )
        
        # Validate cash-only constraint
        if self.trading_cfg.cash_only and result.new_state.cash < -1e-6:
            raise RuntimeError(
                f"Cash-only violated: cash = ${result.new_state.cash:.2f}"
            )
        
        self.state = result.new_state
        
        # Advance time
        self._t += 1
        self.state = self.state.update(t=self._t)
        
        # Mark-to-market at new close
        mark_price = self._current_price()
        equity_now = self.state.cash + self.state.shares * mark_price
        
        # Compute reward: log return - turnover penalty
        prev_eq = max(self._prev_equity, 1e-12)
        log_return = np.log(max(equity_now, 1e-12) / prev_eq)
        
        turnover = result.traded_notional / prev_eq
        turnover_penalty = self.trading_cfg.turnover_penalty * turnover
        
        reward = float(log_return - turnover_penalty)
        
        # Update state
        self._prev_equity = float(equity_now)
        self.state = self.state.update(equity=float(equity_now))
        
        # Check termination
        terminated = False
        truncated = bool(self._t >= self._end)
        
        # Get observation
        obs = self._get_obs()
        
        # Build info dict
        info = {
            'date': str(self.features.index[self._t]),
            'exec_price': exec_price,
            'mark_price': mark_price,
            'target_weight': self.exec_engine.clamp_weight(target_weight),
            'actual_weight': self.state.current_weight(mark_price),
            'shares': self.state.shares,
            'cash': self.state.cash,
            'equity': self.state.equity,
            'traded_notional': result.traded_notional,
            'costs': result.costs,
            'turnover': turnover,
            'log_return': log_return,
            'realized_pnl': self.state.realized_pnl,
            'unrealized_pnl': self.state.unrealized_pnl(mark_price),
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self) -> None:
        """Render current state (human-readable print)."""
        if self.state is None:
            print("Environment not reset")
            return
        
        px = self._current_price()
        print({
            't': self._t,
            'date': str(self.features.index[self._t]),
            'equity': f"${self.state.equity:,.2f}",
            'cash': f"${self.state.cash:,.2f}",
            'shares': f"{self.state.shares:.2f}",
            'weight': f"{self.state.current_weight(px):.3f}",
            'unrealized_pnl': f"${self.state.unrealized_pnl(px):,.2f}",
            'realized_pnl': f"${self.state.realized_pnl:,.2f}",
        })

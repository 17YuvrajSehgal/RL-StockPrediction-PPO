"""
Custom Stable Baselines3 callbacks for financial trading.

This module provides specialized callbacks for monitoring training progress,
logging financial metrics, and enforcing risk constraints.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from trading_system.config import RiskConfig


class TensorBoardCallback(BaseCallback):
    """
    Enhanced TensorBoard logging callback.
    
    Logs additional metrics beyond the default SB3 logging including
    gradient norms, learning rate schedule, and custom scalars.
    """
    
    def __init__(self, log_dir: str, log_interval: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_interval = log_interval
    
    def _on_step(self) -> bool:
        if self.n_calls % self.log_interval == 0:
            # Log learning rate
            if hasattr(self.model, 'learning_rate'):
                lr = self.model.learning_rate
                if callable(lr):
                    lr = lr(self.model._current_progress_remaining)
                self.logger.record("train/learning_rate", lr)
            
            # Log gradient norm if available
            if hasattr(self.model.policy, 'optimizer'):
                total_norm = 0.0
                for p in self.model.policy.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.logger.record("train/gradient_norm", total_norm)
        
        return True


class FinancialMetricsCallback(BaseCallback):
    """
    Callback for logging financial performance metrics.
    
    Computes and logs Sharpe ratio, max drawdown, win rate, and other
    trading-specific metrics during training.
    """
    
    def __init__(
        self,
        eval_env: Optional[Any] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.episode_returns = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # Collect episode statistics
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'r' in info:
                    self.episode_returns.append(info['r'])
                if 'l' in info:
                    self.episode_lengths.append(info['l'])
        
        # Evaluate periodically
        if self.n_calls % self.eval_freq == 0 and self.eval_env is not None:
            self._evaluate()
        
        return True
    
    def _evaluate(self) -> None:
        """Run evaluation episodes and log metrics."""
        episode_rewards = []
        episode_equities = []
        
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0
            equity_curve = []
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                episode_reward += reward[0]
                
                if 'equity' in info[0]:
                    equity_curve.append(info[0]['equity'])
            
            episode_rewards.append(episode_reward)
            if equity_curve:
                episode_equities.append(equity_curve)
        
        # Compute metrics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        # Log to TensorBoard
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/std_reward", std_reward)
        
        # Compute Sharpe ratio if we have equity curves
        if episode_equities:
            returns_list = []
            for equity in episode_equities:
                if len(equity) > 1:
                    returns = np.diff(np.log(equity))
                    returns_list.extend(returns)
            
            if returns_list:
                sharpe = np.mean(returns_list) / (np.std(returns_list) + 1e-8) * np.sqrt(252)
                self.logger.record("eval/sharpe_ratio", sharpe)
                
                # Max drawdown
                for equity in episode_equities:
                    running_max = np.maximum.accumulate(equity)
                    drawdown = (equity - running_max) / running_max
                    max_dd = np.min(drawdown)
                    self.logger.record("eval/max_drawdown", max_dd)
                    break  # Just log first episode's drawdown


class RiskMonitorCallback(BaseCallback):
    """
    Callback for monitoring risk constraints during training.
    
    Tracks drawdowns, position sizes, and other risk metrics.
    Triggers alerts or stops training if risk limits are breached.
    """
    
    def __init__(
        self,
        risk_config: RiskConfig,
        checkpoint_dir: str,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.risk_config = risk_config
        self.checkpoint_dir = checkpoint_dir
        self.peak_equity = 0.0
        self.current_drawdown = 0.0
        self.daily_start_equity = 0.0
        self.daily_loss = 0.0
    
    def _on_step(self) -> bool:
        # Extract equity from info if available
        if len(self.model.ep_info_buffer) > 0:
            latest_info = self.model.ep_info_buffer[-1]
            
            if 'equity' in latest_info:
                equity = latest_info['equity']
                
                # Update peak equity
                if equity > self.peak_equity:
                    self.peak_equity = equity
                
                # Compute drawdown
                if self.peak_equity > 0:
                    self.current_drawdown = (self.peak_equity - equity) / self.peak_equity
                    
                    # Log drawdown
                    self.logger.record("risk/current_drawdown", self.current_drawdown)
                    self.logger.record("risk/peak_equity", self.peak_equity)
                    
                    # Check drawdown limit
                    if (self.risk_config.enable_circuit_breakers and 
                        self.current_drawdown > self.risk_config.max_drawdown_pct):
                        print(f"\n⚠️  CIRCUIT BREAKER: Max drawdown exceeded ({self.current_drawdown:.2%})")
                        print(f"   Limit: {self.risk_config.max_drawdown_pct:.2%}")
                        print(f"   Stopping training for risk management")
                        return False  # Stop training
                    
                    # Alert on significant drawdown
                    if self.current_drawdown > self.risk_config.alert_on_drawdown_pct:
                        if self.verbose > 0:
                            print(f"\n⚠️  Alert: Drawdown at {self.current_drawdown:.2%}")
        
        return True

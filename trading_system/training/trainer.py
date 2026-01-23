"""
Production-ready PPO Trainer with GPU support and comprehensive monitoring.

This module provides an enterprise-grade training orchestrator that wraps
Stable Baselines3 PPO with financial-specific enhancements.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from swing_trading import SwingTradingEnv
from trading_system.config import TrainingConfig, RiskConfig
from trading_system.training.callbacks import (
    TensorBoardCallback,
    FinancialMetricsCallback,
    RiskMonitorCallback,
)


class PPOTrainer:
    """
    Production-ready PPO trainer for financial trading.
    
    Features:
    - Automatic GPU/CPU device selection
    - Comprehensive TensorBoard logging
    - Model checkpointing with versioning
    - Risk monitoring during training
    - Training resume from checkpoints
    - Hyperparameter logging
    
    Attributes:
        config: Training configuration
        risk_config: Risk management configuration
        env: Training environment
        model: PPO model
        device: PyTorch device
    
    Example:
        >>> from swing_trading import SwingTradingEnv, SwingTradingConfig
        >>> from trading_system import PPOTrainer, TrainingConfig
        >>> 
        >>> # Create environment
        >>> env = SwingTradingEnv(df, SwingTradingConfig())
        >>> 
        >>> # Create trainer
        >>> config = TrainingConfig.production("aapl_v1")
        >>> trainer = PPOTrainer(env, config)
        >>> 
        >>> # Train
        >>> trainer.train()
        >>> 
        >>> # Save
        >>> trainer.save("models/aapl_v1_final")
    """
    
    def __init__(
        self,
        env: SwingTradingEnv,
        config: TrainingConfig,
        risk_config: Optional[RiskConfig] = None,
        eval_env: Optional[SwingTradingEnv] = None,
    ) -> None:
        """
        Initialize PPO trainer.
        
        Args:
            env: Training environment
            config: Training configuration
            risk_config: Risk management configuration
            eval_env: Optional evaluation environment
        """
        self.config = config
        self.risk_config = risk_config or RiskConfig.institutional()
        
        # Setup device
        self.device = config.device.get_device()
        device_name = config.device.get_device_name()
        print(f"🖥️  Using device: {device_name}")
        
        if self.device.type == 'cuda':
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        
        # Wrap environment
        self.env = DummyVecEnv([lambda: env])
        self.eval_env = DummyVecEnv([lambda: eval_env]) if eval_env else None
        
        # Normalize observations for better training
        self.env = VecNormalize(
            self.env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )
        
        if self.eval_env:
            self.eval_env = VecNormalize(
                self.eval_env,
                training=False,
                norm_obs=True,
                norm_reward=False,
            )
        
        # Create model
        self.model: Optional[PPO] = None
        self._create_model()
        
        # Training state
        self.training_start_time: Optional[float] = None
        self.best_mean_reward: float = -np.inf
        self.training_history: Dict[str, list] = {
            'timesteps': [],
            'mean_reward': [],
            'std_reward': [],
            'mean_ep_length': [],
        }
    
    def _create_model(self) -> None:
        """Create PPO model with configuration."""
        if self.config.resume_from and Path(self.config.resume_from).exists():
            print(f"📂 Resuming from checkpoint: {self.config.resume_from}")
            self.model = PPO.load(
                self.config.resume_from,
                env=self.env,
                device=self.device,
            )
        else:
            print("🆕 Creating new PPO model")
            self.model = PPO(
                policy="MlpPolicy",
                env=self.env,
                learning_rate=self.config.ppo.learning_rate,
                n_steps=self.config.ppo.n_steps,
                batch_size=self.config.ppo.batch_size,
                n_epochs=self.config.ppo.n_epochs,
                gamma=self.config.ppo.gamma,
                gae_lambda=self.config.ppo.gae_lambda,
                clip_range=self.config.ppo.clip_range,
                clip_range_vf=self.config.ppo.clip_range_vf,
                ent_coef=self.config.ppo.ent_coef,
                vf_coef=self.config.ppo.vf_coef,
                max_grad_norm=self.config.ppo.max_grad_norm,
                target_kl=self.config.ppo.target_kl,
                tensorboard_log=self.config.logging.tensorboard_log,
                verbose=self.config.logging.verbose,
                seed=self.config.seed,
                device=self.device,
            )
        
        # Log hyperparameters
        self._log_hyperparameters()
    
    def _log_hyperparameters(self) -> None:
        """Log all hyperparameters to file."""
        hparams = {
            'experiment_name': self.config.experiment_name,
            'device': str(self.device),
            'total_timesteps': self.config.total_timesteps,
            'seed': self.config.seed,
            'ppo': {
                'learning_rate': self.config.ppo.learning_rate,
                'n_steps': self.config.ppo.n_steps,
                'batch_size': self.config.ppo.batch_size,
                'n_epochs': self.config.ppo.n_epochs,
                'gamma': self.config.ppo.gamma,
                'gae_lambda': self.config.ppo.gae_lambda,
                'clip_range': self.config.ppo.clip_range,
                'ent_coef': self.config.ppo.ent_coef,
                'vf_coef': self.config.ppo.vf_coef,
                'max_grad_norm': self.config.ppo.max_grad_norm,
            },
            'risk': {
                'max_position_size': self.risk_config.max_position_size,
                'max_leverage': self.risk_config.max_leverage,
                'max_drawdown_pct': self.risk_config.max_drawdown_pct,
            },
            'timestamp': datetime.now().isoformat(),
        }
        
        hparams_file = Path(self.config.checkpoint_dir) / f"{self.config.experiment_name}_hparams.json"
        with open(hparams_file, 'w') as f:
            json.dump(hparams, f, indent=2)
        
        print(f"📝 Hyperparameters logged to: {hparams_file}")
    
    def _create_callbacks(self) -> CallbackList:
        """Create training callbacks."""
        callbacks = []
        
        # TensorBoard callback
        tb_callback = TensorBoardCallback(
            log_dir=self.config.logging.tensorboard_log,
            log_interval=self.config.logging.log_interval,
        )
        callbacks.append(tb_callback)
        
        # Financial metrics callback
        if self.config.logging.log_financial_metrics:
            fin_callback = FinancialMetricsCallback(
                eval_env=self.eval_env,
                eval_freq=self.config.logging.eval_freq,
            )
            callbacks.append(fin_callback)
        
        # Risk monitor callback
        risk_callback = RiskMonitorCallback(
            risk_config=self.risk_config,
            checkpoint_dir=self.config.checkpoint_dir,
        )
        callbacks.append(risk_callback)
        
        return CallbackList(callbacks)
    
    def train(self) -> None:
        """
        Train the PPO model.
        
        This method orchestrates the entire training process including:
        - Callback setup
        - Training loop
        - Progress monitoring
        - Model checkpointing
        """
        print("\n" + "="*70)
        print(f"🚀 Starting Training: {self.config.experiment_name}")
        print("="*70)
        print(f"Total timesteps: {self.config.total_timesteps:,}")
        print(f"Eval frequency: {self.config.logging.eval_freq:,}")
        print(f"Save frequency: {self.config.logging.save_freq:,}")
        print(f"Device: {self.config.device.get_device_name()}")
        print("="*70 + "\n")
        
        self.training_start_time = time.time()
        
        # Create callbacks
        callbacks = self._create_callbacks()
        
        try:
            # Train model
            self.model.learn(
                total_timesteps=self.config.total_timesteps,
                callback=callbacks,
                progress_bar=True,
            )
            
            # Training completed successfully
            elapsed = time.time() - self.training_start_time
            print("\n" + "="*70)
            print("✅ Training Completed Successfully!")
            print("="*70)
            print(f"Total time: {elapsed/3600:.2f} hours")
            print(f"Steps per second: {self.config.total_timesteps/elapsed:.1f}")
            print("="*70 + "\n")
            
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
            self._save_checkpoint("interrupted")
        except Exception as e:
            print(f"\n❌ Training failed with error: {e}")
            self._save_checkpoint("error")
            raise
    
    def _save_checkpoint(self, suffix: str = "") -> None:
        """Save model checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.experiment_name}_{suffix}_{timestamp}" if suffix else f"{self.config.experiment_name}_{timestamp}"
        path = Path(self.config.checkpoint_dir) / filename
        
        self.model.save(path)
        print(f"💾 Checkpoint saved: {path}")
    
    def save(self, path: str) -> None:
        """
        Save final model.
        
        Args:
            path: Path to save model (without extension)
        """
        self.model.save(path)
        
        # Also save normalization statistics
        if isinstance(self.env, VecNormalize):
            self.env.save(f"{path}_vecnormalize.pkl")
        
        print(f"💾 Model saved: {path}")
    
    def load(self, path: str) -> None:
        """
        Load trained model.
        
        Args:
            path: Path to model (without extension)
        """
        self.model = PPO.load(path, device=self.device)
        
        # Load normalization statistics if available
        norm_path = f"{path}_vecnormalize.pkl"
        if Path(norm_path).exists():
            self.env = VecNormalize.load(norm_path, self.env)
        
        print(f"📂 Model loaded: {path}")
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            n_episodes: Number of episodes to evaluate
        
        Returns:
            Dictionary of evaluation metrics
        """
        env = self.eval_env if self.eval_env else self.env
        
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0]
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
        }
        
        print("\n📊 Evaluation Results:")
        print(f"   Mean Reward: {metrics['mean_reward']:.4f} ± {metrics['std_reward']:.4f}")
        print(f"   Min/Max: {metrics['min_reward']:.4f} / {metrics['max_reward']:.4f}")
        print(f"   Mean Length: {metrics['mean_length']:.1f}")
        
        return metrics

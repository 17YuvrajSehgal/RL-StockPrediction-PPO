"""
Daily model trainer for incremental PPO training.

This module provides the core training logic for daily model updates,
supporting both full retraining and fine-tuning modes with proper
temporal data handling to prevent future data leakage.

Design Principles:
    - Strict temporal ordering (no future data leakage)
    - Support for both full training and fine-tuning
    - Proper model versioning with date stamps
    - Comprehensive logging and metrics
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from daily_trading.config import (
    DailyTrainingConfig,
    TrainingMode,
    TrainingWindowConfig,
    PathConfig,
)
from daily_trading.utils import DataLoader, TradingCalendar, setup_logging

# Import from parent package
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from swing_trading import SwingTradingEnv, SwingTradingConfig


@dataclass
class TrainResult:
    """
    Result of a training run.
    
    Attributes:
        ticker: Stock ticker trained on
        as_of_date: Training cutoff date
        mode: Training mode used (full/fine_tune)
        model_path: Path to saved model
        timesteps: Total timesteps trained
        training_time_seconds: Wall clock training time
        final_reward: Final episode reward
        metrics: Training metrics dictionary
        success: Whether training completed successfully
        error: Error message if training failed
    """
    
    ticker: str
    as_of_date: date
    mode: TrainingMode
    model_path: Path
    timesteps: int
    training_time_seconds: float
    final_reward: float
    metrics: Dict[str, float] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ticker": self.ticker,
            "as_of_date": self.as_of_date.isoformat(),
            "mode": str(self.mode),
            "model_path": str(self.model_path),
            "timesteps": self.timesteps,
            "training_time_seconds": self.training_time_seconds,
            "final_reward": self.final_reward,
            "metrics": self.metrics,
            "success": self.success,
            "error": self.error,
        }
    
    def save(self, path: Path) -> None:
        """Save result to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class TrainingProgressCallback(BaseCallback):
    """Callback to track training progress and metrics."""
    
    def __init__(self, log_interval: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []
        self._last_log = 0
    
    def _on_step(self) -> bool:
        # Collect episode info
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'r' in info:
                    self.episode_rewards.append(info['r'])
                if 'l' in info:
                    self.episode_lengths.append(info['l'])
        
        # Log progress
        if self.n_calls - self._last_log >= self.log_interval:
            if self.verbose > 0 and self.episode_rewards:
                recent_rewards = self.episode_rewards[-10:]
                mean_reward = np.mean(recent_rewards)
                print(f"  Step {self.n_calls}: Mean reward (last 10 eps): {mean_reward:.4f}")
            self._last_log = self.n_calls
        
        return True
    
    def get_metrics(self) -> Dict[str, float]:
        """Get summary metrics from training."""
        if not self.episode_rewards:
            return {}
        
        return {
            "mean_reward": float(np.mean(self.episode_rewards)),
            "std_reward": float(np.std(self.episode_rewards)),
            "min_reward": float(np.min(self.episode_rewards)),
            "max_reward": float(np.max(self.episode_rewards)),
            "total_episodes": len(self.episode_rewards),
            "mean_episode_length": float(np.mean(self.episode_lengths)) if self.episode_lengths else 0,
        }


class DailyTrainer:
    """
    Daily model trainer supporting full training and fine-tuning.
    
    This class handles all aspects of daily model updates:
    - Loading and filtering data with temporal integrity
    - Creating training environments
    - Training or fine-tuning PPO models
    - Saving models with date versioning
    
    Attributes:
        config: Daily training configuration
        paths: Path configuration
        data_loader: Data loading utility
        calendar: Trading calendar
        logger: Logger instance
    
    Example:
        >>> trainer = DailyTrainer(config, paths)
        >>> 
        >>> # Full training from scratch
        >>> result = trainer.train(
        ...     ticker="AAPL",
        ...     as_of_date=date(2026, 1, 24),
        ...     mode=TrainingMode.FULL
        ... )
        >>> 
        >>> # Fine-tune existing model
        >>> result = trainer.fine_tune(
        ...     ticker="AAPL",
        ...     as_of_date=date(2026, 1, 24),
        ...     base_model_path=Path("models/AAPL_base.zip")
        ... )
    """
    
    def __init__(
        self,
        config: DailyTrainingConfig,
        paths: PathConfig,
        env_config: Optional[SwingTradingConfig] = None,
        device: str = "auto",
    ):
        """
        Initialize daily trainer.
        
        Args:
            config: Training configuration
            paths: Path configuration
            env_config: Trading environment configuration
            device: PyTorch device (auto, cuda, cpu)
        """
        self.config = config
        self.paths = paths
        self.env_config = env_config or SwingTradingConfig()
        self.device = self._resolve_device(device)
        
        self.data_loader = DataLoader(paths.data_dir)
        self.calendar = TradingCalendar()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Ensure directories exist
        paths.ensure_directories()
    
    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def train(
        self,
        ticker: str,
        as_of_date: date,
        mode: Optional[TrainingMode] = None,
        base_model_path: Optional[Path] = None,
        timesteps: Optional[int] = None,
    ) -> TrainResult:
        """
        Train or fine-tune model for specified date.
        
        This is the main entry point for training. It handles:
        1. Loading data up to as_of_date (no future leakage)
        2. Creating train/validation environments
        3. Training or fine-tuning the model
        4. Saving the model with date versioning
        
        Args:
            ticker: Stock ticker to train on
            as_of_date: Cutoff date for training data (inclusive)
            mode: Training mode (full or fine_tune). Uses config default if None.
            base_model_path: Base model for fine-tuning. Required if mode is FINE_TUNE.
            timesteps: Override default timesteps
        
        Returns:
            TrainResult with training outcome
        
        Raises:
            ValueError: If insufficient data or invalid configuration
            FileNotFoundError: If base_model_path not found for fine-tuning
        """
        mode = mode or self.config.mode
        timesteps = timesteps or (
            self.config.timesteps_fine_tune if mode == TrainingMode.FINE_TUNE 
            else self.config.timesteps_full
        )
        
        self.logger.info(f"Starting {mode.value} training for {ticker} as of {as_of_date}")
        self.logger.info(f"Timesteps: {timesteps:,}, Device: {self.device}")
        
        start_time = time.time()
        
        try:
            # 1. Load and prepare data
            train_df, val_df = self._prepare_data(ticker, as_of_date)
            
            # 2. Create environments
            train_env, val_env = self._create_environments(train_df, val_df)
            
            # 3. Create or load model
            if mode == TrainingMode.FINE_TUNE:
                if base_model_path is None:
                    # Try to find previous day's model
                    base_model_path = self._find_latest_model(ticker, as_of_date)
                    if base_model_path is None:
                        self.logger.warning(
                            "No base model found for fine-tuning, falling back to full training"
                        )
                        mode = TrainingMode.FULL
                
                if mode == TrainingMode.FINE_TUNE:
                    model = self._load_model_for_fine_tune(base_model_path, train_env)
            
            if mode == TrainingMode.FULL:
                model = self._create_new_model(train_env)
            
            # 4. Train
            callback = TrainingProgressCallback(log_interval=5000, verbose=1)
            
            self.logger.info(f"Training for {timesteps:,} timesteps...")
            model.learn(
                total_timesteps=timesteps,
                callback=callback,
                progress_bar=True,
            )
            
            # 5. Save model
            model_path = self._save_model(model, train_env, ticker, as_of_date, mode)
            
            # 6. Collect metrics
            training_time = time.time() - start_time
            metrics = callback.get_metrics()
            
            result = TrainResult(
                ticker=ticker,
                as_of_date=as_of_date,
                mode=mode,
                model_path=model_path,
                timesteps=timesteps,
                training_time_seconds=training_time,
                final_reward=metrics.get("mean_reward", 0.0),
                metrics=metrics,
                success=True,
            )
            
            # Save result metadata
            result.save(model_path.parent / "train_result.json")
            
            self.logger.info(
                f"Training completed in {training_time:.1f}s. "
                f"Final reward: {result.final_reward:.4f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            training_time = time.time() - start_time
            
            return TrainResult(
                ticker=ticker,
                as_of_date=as_of_date,
                mode=mode,
                model_path=Path(""),
                timesteps=timesteps,
                training_time_seconds=training_time,
                final_reward=0.0,
                success=False,
                error=str(e),
            )
    
    def fine_tune(
        self,
        ticker: str,
        as_of_date: date,
        base_model_path: Path,
        timesteps: Optional[int] = None,
    ) -> TrainResult:
        """
        Fine-tune an existing model with recent data.
        
        Convenience method for fine-tuning mode.
        
        Args:
            ticker: Stock ticker
            as_of_date: Cutoff date
            base_model_path: Path to base model
            timesteps: Override default fine-tune timesteps
        
        Returns:
            TrainResult with training outcome
        """
        return self.train(
            ticker=ticker,
            as_of_date=as_of_date,
            mode=TrainingMode.FINE_TUNE,
            base_model_path=base_model_path,
            timesteps=timesteps,
        )
    
    def _prepare_data(
        self, 
        ticker: str, 
        as_of_date: date,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and split data for training.
        
        CRITICAL: Only data up to and including as_of_date is used.
        """
        # Determine date range based on window config
        window = self.config.window
        available_start = self.data_loader.get_earliest_date(ticker)
        start_date = window.get_start_date(as_of_date, available_start)
        
        self.logger.info(f"Loading data from {start_date} to {as_of_date}")
        
        # Load data with strict date filtering
        df = self.data_loader.load_ticker(
            ticker=ticker,
            as_of_date=as_of_date,
            start_date=start_date,
        )
        
        self.logger.info(f"Loaded {len(df)} trading days")
        
        # Validate minimum data
        if len(df) < self.config.window.min_days:
            raise ValueError(
                f"Insufficient data: {len(df)} days, minimum {self.config.window.min_days} required"
            )
        
        # Split into train/validation
        # Ensure validation set is large enough for feature engineering + lookback
        # Feature lag (50) + min lookback (20) + min episode (20) = 90 days roughly
        min_val_days = 100
        min_train_days = 60 # Absolute minimum for anything useful
        
        val_days = int(len(df) * self.config.validation_split)
        
        if val_days < min_val_days:
            # Try to boost validation size
            available_for_val = len(df) - min_train_days
            val_days = min(min_val_days, available_for_val)
            
            if val_days < 50: # If we really can't get enough data
                 self.logger.warning(f"Validation set extremely small ({val_days} days). Evaluation may fail.")
        
        split_idx = len(df) - val_days
        
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]
        
        self.logger.info(
            f"Train: {len(train_df)} days ({train_df.index[0].date()} to {train_df.index[-1].date()})"
        )
        self.logger.info(
            f"Val: {len(val_df)} days ({val_df.index[0].date()} to {val_df.index[-1].date()})"
        )
        
        return train_df, val_df
    
    def _create_environments(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> tuple[VecNormalize, VecNormalize]:
        """Create vectorized and normalized environments."""
        from dataclasses import replace
        
        # 1. Calculate Lookback from Training Data
        # Use minimum of 20 lookback to ensure stable features
        # Max 60, or 10% of training data
        feature_overhead = 50 # Conservative estimate for Technical Indicators
        
        train_features_len = len(train_df) - feature_overhead
        if train_features_len < 20: 
             # Very small training set, fallback to minimal lookback
             lookback = 10
        else:
             lookback = min(60, max(20, train_features_len // 5))
        
        # 2. Configure Training Environment
        # Episode length: leave buffer
        train_ep_len = min(126, max(20, train_features_len - lookback - 5))
        
        train_env_cfg = replace(
            self.env_config.environment,
            lookback=lookback,
            episode_length=train_ep_len,
        )
        full_train_config = replace(self.env_config, environment=train_env_cfg)
        
        self.logger.info(f"Train Env: lookback={lookback}, episode_length={train_ep_len}")
        
        # 3. Configure Validation Environment
        # Must use SAME lookback, but adapt episode length to available validation data
        val_features_len = len(val_df) - feature_overhead
        
        # Calculate max possible episode length for validation
        # min_length = lookback + ep + 1 <= val_features_len
        val_ep_len = max(10, val_features_len - lookback - 2)
        
        # Cap at standard length
        val_ep_len = min(126, val_ep_len)
        
        # Check if validation is possible
        if val_features_len < lookback + val_ep_len + 1:
             self.logger.warning(
                 f"Validation data too small for strict evaluation. "
                 f"Features: {val_features_len}, Need: {lookback + val_ep_len + 1}. "
                 f"Validation metrics may be unreliable."
             )
             # Try to squeeze it
             val_ep_len = val_features_len - lookback - 2
        
        if val_ep_len < 1:
             self.logger.warning("Validation episode length < 1. Validation environment will likely fail.")
             val_ep_len = 5 # Force minimal and hope
             
        val_env_cfg = replace(
            self.env_config.environment,
            lookback=lookback,
            episode_length=val_ep_len,
        )
        full_val_config = replace(self.env_config, environment=val_env_cfg)
        
        self.logger.info(f"Val Env: lookback={lookback}, episode_length={val_ep_len}")

        # Create environments
        train_env = SwingTradingEnv(train_df, full_train_config)
        
        try:
            val_env = SwingTradingEnv(val_df, full_val_config)
        except ValueError as e:
            self.logger.error(f"Failed to create validation env: {e}")
            # Fallback: create a dummy validation env using training data just to satisfy VecNormalize
            # (Note: this means validation metrics will be meaningless, but training won't crash)
            self.logger.warning("Using training data for validation env as fallback (IGNORE VAL METRICS)")
            val_env = SwingTradingEnv(train_df, full_train_config)
        
        # Vectorize
        train_vec = DummyVecEnv([lambda: train_env])
        val_vec = DummyVecEnv([lambda: val_env])
        
        # Normalize
        train_vec = VecNormalize(
            train_vec,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )
        
        val_vec = VecNormalize(
            val_vec,
            training=False,
            norm_obs=True,
            norm_reward=False,
        )
        
        return train_vec, val_vec
    
    def _create_new_model(self, env: VecNormalize) -> PPO:
        """Create new PPO model from scratch."""
        self.logger.info("Creating new PPO model")
        
        return PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=self.config.learning_rate_full,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=str(self.paths.logs_dir),
            verbose=0,
            seed=self.config.seed,
            device=self.device,
        )
    
    def _load_model_for_fine_tune(
        self, 
        model_path: Path, 
        env: VecNormalize,
    ) -> PPO:
        """Load existing model for fine-tuning."""
        self.logger.info(f"Loading model from {model_path} for fine-tuning")
        
        # Handle both .zip and non-.zip paths
        if not str(model_path).endswith('.zip'):
            model_path = Path(str(model_path) + '.zip') if Path(str(model_path) + '.zip').exists() else model_path
        
        model = PPO.load(model_path, env=env, device=self.device)
        
        # Adjust learning rate for fine-tuning (lower)
        model.learning_rate = self.config.learning_rate_fine_tune
        
        return model
    
    def _find_latest_model(self, ticker: str, before_date: date) -> Optional[Path]:
        """Find the most recent model before given date."""
        ticker_dir = self.paths.models_dir / ticker
        
        if not ticker_dir.exists():
            return None
        
        # Get all dated model directories
        model_dates = []
        for d in ticker_dir.iterdir():
            if d.is_dir():
                try:
                    model_date = date.fromisoformat(d.name)
                    if model_date < before_date:
                        model_dates.append((model_date, d))
                except ValueError:
                    continue
        
        if not model_dates:
            return None
        
        # Get most recent
        model_dates.sort(key=lambda x: x[0], reverse=True)
        latest_dir = model_dates[0][1]
        
        # Find model file
        model_file = latest_dir / "model.zip"
        if model_file.exists():
            return model_file
        
        return None
    
    def _save_model(
        self,
        model: PPO,
        env: VecNormalize,
        ticker: str,
        as_of_date: date,
        mode: TrainingMode,
    ) -> Path:
        """Save model with date versioning."""
        # Create versioned directory
        model_dir = self.paths.model_path(ticker, as_of_date)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / "model"
        model.save(model_path)
        
        # Save normalization stats
        vecnorm_path = model_dir / "vecnormalize.pkl"
        env.save(str(vecnorm_path))
        
        # Save metadata
        metadata = {
            "ticker": ticker,
            "as_of_date": as_of_date.isoformat(),
            "mode": str(mode),
            "created_at": datetime.now().isoformat(),
            "device": self.device,
        }
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Model saved to {model_dir}")
        
        return model_dir / "model.zip"


def compare_training_modes(
    ticker: str,
    as_of_date: date,
    base_model_path: Optional[Path] = None,
    paths: Optional[PathConfig] = None,
) -> Dict[str, TrainResult]:
    """
    Compare full training vs fine-tuning for research.
    
    Runs both training modes and returns results for comparison.
    
    Args:
        ticker: Stock ticker
        as_of_date: Training cutoff date
        base_model_path: Base model for fine-tuning
        paths: Path configuration
    
    Returns:
        Dictionary with "full" and "fine_tune" TrainResults
    """
    paths = paths or PathConfig()
    results = {}
    
    # Full training
    full_config = DailyTrainingConfig.full_train()
    trainer = DailyTrainer(full_config, paths)
    results["full"] = trainer.train(ticker, as_of_date, mode=TrainingMode.FULL)
    
    # Fine-tuning (requires base model)
    if base_model_path or results["full"].success:
        fine_config = DailyTrainingConfig.fine_tune()
        trainer = DailyTrainer(fine_config, paths)
        
        # Use full training result as base if no base provided
        base = base_model_path or results["full"].model_path
        results["fine_tune"] = trainer.train(
            ticker, as_of_date, 
            mode=TrainingMode.FINE_TUNE,
            base_model_path=base,
        )
    
    return results

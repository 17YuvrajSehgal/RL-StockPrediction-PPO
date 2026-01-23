"""
Training configuration for production PPO system.

This module provides comprehensive configuration management for PPO training
with automatic GPU detection, validation, and preset configurations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import torch


@dataclass(frozen=True)
class DeviceConfig:
    """
    Device configuration with automatic GPU detection.
    
    Attributes:
        device: Device to use ('auto', 'cuda', 'cpu', 'mps')
        gpu_id: Specific GPU ID to use (None for default)
        allow_cpu_fallback: If True, fallback to CPU if GPU unavailable
        mixed_precision: Use mixed precision training (FP16)
    
    Example:
        >>> # Automatic device selection
        >>> config = DeviceConfig()
        >>> print(config.get_device())
        
        >>> # Force specific GPU
        >>> config = DeviceConfig(device='cuda', gpu_id=0)
    """
    
    device: Literal['auto', 'cuda', 'cpu', 'mps'] = 'auto'
    gpu_id: Optional[int] = None
    allow_cpu_fallback: bool = True
    mixed_precision: bool = False
    
    def get_device(self) -> torch.device:
        """
        Get PyTorch device with automatic detection and fallback.
        
        Returns:
            torch.device object
        
        Example:
            >>> config = DeviceConfig()
            >>> device = config.get_device()
            >>> print(f"Using device: {device}")
        """
        if self.device == 'auto':
            if torch.cuda.is_available():
                device_str = f'cuda:{self.gpu_id}' if self.gpu_id is not None else 'cuda'
                return torch.device(device_str)
            elif torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        
        elif self.device == 'cuda':
            if not torch.cuda.is_available():
                if self.allow_cpu_fallback:
                    print("WARNING: CUDA not available, falling back to CPU")
                    return torch.device('cpu')
                else:
                    raise RuntimeError("CUDA requested but not available")
            device_str = f'cuda:{self.gpu_id}' if self.gpu_id is not None else 'cuda'
            return torch.device(device_str)
        
        elif self.device == 'mps':
            if not torch.backends.mps.is_available():
                if self.allow_cpu_fallback:
                    print("WARNING: MPS not available, falling back to CPU")
                    return torch.device('cpu')
                else:
                    raise RuntimeError("MPS requested but not available")
            return torch.device('mps')
        
        else:  # cpu
            return torch.device('cpu')
    
    def get_device_name(self) -> str:
        """Get human-readable device name."""
        device = self.get_device()
        if device.type == 'cuda':
            return f"CUDA GPU {device.index if device.index is not None else 0}"
        elif device.type == 'mps':
            return "Apple MPS"
        else:
            return "CPU"


@dataclass(frozen=True)
class PPOConfig:
    """
    PPO algorithm hyperparameters.
    
    These are the core PPO parameters from the original paper with
    conservative defaults suitable for financial trading.
    
    Attributes:
        learning_rate: Learning rate for optimizer
        n_steps: Number of steps to collect per environment
        batch_size: Minibatch size for training
        n_epochs: Number of epochs for policy update
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: PPO clipping parameter
        clip_range_vf: Value function clipping (None to disable)
        ent_coef: Entropy coefficient for exploration
        vf_coef: Value function coefficient
        max_grad_norm: Maximum gradient norm for clipping
        target_kl: Target KL divergence for early stopping
    
    Example:
        >>> # Conservative settings for production
        >>> config = PPOConfig.conservative()
        
        >>> # Aggressive settings for research
        >>> config = PPOConfig.aggressive()
    """
    
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.015
    
    def __post_init__(self) -> None:
        """Validate PPO hyperparameters."""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.batch_size > self.n_steps:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be <= n_steps ({self.n_steps})"
            )
        if self.n_epochs <= 0:
            raise ValueError(f"n_epochs must be positive, got {self.n_epochs}")
        if not 0 <= self.gamma <= 1:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if not 0 <= self.gae_lambda <= 1:
            raise ValueError(f"gae_lambda must be in [0, 1], got {self.gae_lambda}")
        if self.clip_range <= 0:
            raise ValueError(f"clip_range must be positive, got {self.clip_range}")
    
    @classmethod
    def conservative(cls) -> PPOConfig:
        """Conservative hyperparameters for production trading."""
        return cls(
            learning_rate=1e-4,
            n_steps=4096,
            batch_size=128,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.005,
            target_kl=0.01,
        )
    
    @classmethod
    def aggressive(cls) -> PPOConfig:
        """Aggressive hyperparameters for research/exploration."""
        return cls(
            learning_rate=5e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=15,
            gamma=0.995,
            gae_lambda=0.98,
            clip_range=0.3,
            ent_coef=0.02,
            target_kl=0.02,
        )


@dataclass(frozen=True)
class LoggingConfig:
    """
    Logging and monitoring configuration.
    
    Attributes:
        tensorboard_log: Directory for TensorBoard logs
        log_interval: Log every N steps
        eval_freq: Evaluate every N steps
        save_freq: Save checkpoint every N steps
        verbose: Verbosity level (0=none, 1=info, 2=debug)
        log_financial_metrics: Log custom financial metrics
        log_gradients: Log gradient statistics
        log_weights: Log weight distributions
    
    Example:
        >>> config = LoggingConfig(
        ...     tensorboard_log="./runs/experiment_1",
        ...     eval_freq=10000
        ... )
    """
    
    tensorboard_log: str = "./tensorboard_logs"
    log_interval: int = 100
    eval_freq: int = 10000
    save_freq: int = 50000
    verbose: int = 1
    log_financial_metrics: bool = True
    log_gradients: bool = True
    log_weights: bool = False
    
    def __post_init__(self) -> None:
        """Validate and create log directories."""
        if self.log_interval <= 0:
            raise ValueError(f"log_interval must be positive, got {self.log_interval}")
        if self.eval_freq <= 0:
            raise ValueError(f"eval_freq must be positive, got {self.eval_freq}")
        if self.save_freq <= 0:
            raise ValueError(f"save_freq must be positive, got {self.save_freq}")
        if self.verbose not in (0, 1, 2):
            raise ValueError(f"verbose must be 0, 1, or 2, got {self.verbose}")
        
        # Create tensorboard log directory
        Path(self.tensorboard_log).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TrainingConfig:
    """
    Master training configuration.
    
    Combines all configuration components for a complete training setup.
    
    Attributes:
        ppo: PPO hyperparameters
        device: Device configuration
        logging: Logging configuration
        total_timesteps: Total training steps
        seed: Random seed for reproducibility
        checkpoint_dir: Directory for model checkpoints
        experiment_name: Name for this experiment
        resume_from: Path to checkpoint to resume from
    
    Example:
        >>> config = TrainingConfig(
        ...     ppo=PPOConfig.conservative(),
        ...     total_timesteps=1_000_000,
        ...     experiment_name="aapl_conservative"
        ... )
    """
    
    ppo: PPOConfig = field(default_factory=PPOConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    total_timesteps: int = 1_000_000
    seed: int = 42
    checkpoint_dir: str = "./checkpoints"
    experiment_name: str = "ppo_trading"
    resume_from: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate and setup training configuration."""
        if self.total_timesteps <= 0:
            raise ValueError(f"total_timesteps must be positive, got {self.total_timesteps}")
        
        # Create checkpoint directory
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def production(cls, experiment_name: str) -> TrainingConfig:
        """Production configuration with conservative settings."""
        return cls(
            ppo=PPOConfig.conservative(),
            device=DeviceConfig(device='auto', mixed_precision=True),
            logging=LoggingConfig(
                tensorboard_log=f"./runs/{experiment_name}",
                eval_freq=25000,
                save_freq=50000,
            ),
            total_timesteps=2_000_000,
            experiment_name=experiment_name,
        )
    
    @classmethod
    def research(cls, experiment_name: str) -> TrainingConfig:
        """Research configuration with faster iteration."""
        return cls(
            ppo=PPOConfig.aggressive(),
            device=DeviceConfig(device='auto'),
            logging=LoggingConfig(
                tensorboard_log=f"./runs/{experiment_name}",
                eval_freq=5000,
                save_freq=10000,
                log_gradients=True,
                log_weights=True,
            ),
            total_timesteps=500_000,
            experiment_name=experiment_name,
        )

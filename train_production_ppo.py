"""
Production-Ready PPO Training Script for Financial Institutions

This script demonstrates enterprise-grade PPO training with:
- Automatic GPU/CPU detection
- Comprehensive TensorBoard monitoring
- Risk management and circuit breakers
- Model checkpointing and versioning
- Financial metrics tracking

Usage:
    python train_production_ppo.py --ticker AAPL --mode production
    python train_production_ppo.py --ticker MSFT --mode research --gpu 0
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from swing_trading import SwingTradingEnv, SwingTradingConfig
from trading_system import PPOTrainer
from trading_system.config import TrainingConfig, RiskConfig


def load_and_prepare_data(ticker: str, data_dir: str = "yf_data") -> pd.DataFrame:
    """
    Load and prepare OHLCV data for training.
    
    Args:
        ticker: Stock ticker symbol
        data_dir: Directory containing data files
    
    Returns:
        Prepared DataFrame with OHLCV data
    """
    print(f"\n📊 Loading data for {ticker}...")
    
    # Find data file
    data_path = Path(data_dir)
    csv_files = list(data_path.glob(f"{ticker}*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No data file found for {ticker} in {data_dir}")
    
    df = pd.read_csv(csv_files[0])
    
    # Fix column names (handle Yahoo Finance tuple format)
    new_cols = {}
    for col in df.columns:
        if 'Date' in str(col):
            new_cols[col] = 'Date'
        elif 'Open' in str(col):
            new_cols[col] = 'Open'
        elif 'High' in str(col):
            new_cols[col] = 'High'
        elif 'Low' in str(col):
            new_cols[col] = 'Low'
        elif 'Close' in str(col) and 'Adj' not in str(col):
            new_cols[col] = 'Close'
        elif 'Volume' in str(col):
            new_cols[col] = 'Volume'
    
    df = df.rename(columns=new_cols)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df.columns = [c.lower() for c in df.columns]
    
    print(f"✅ Loaded {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
    
    return df


def create_train_val_split(df: pd.DataFrame, train_ratio: float = 0.8):
    """
    Split data into training and validation sets.
    
    Args:
        df: Full dataset
        train_ratio: Fraction for training
    
    Returns:
        Tuple of (train_df, val_df)
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    print(f"\n📈 Data split:")
    print(f"   Training: {len(train_df)} rows ({train_df.index[0].date()} to {train_df.index[-1].date()})")
    print(f"   Validation: {len(val_df)} rows ({val_df.index[0].date()} to {val_df.index[-1].date()})")
    
    return train_df, val_df


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Production PPO Training for Stock Trading")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--mode", type=str, choices=["production", "research"], default="production",
                        help="Training mode (production=conservative, research=fast iteration)")
    parser.add_argument("--timesteps", type=int, default=None, help="Total training timesteps")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID to use (None=auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment name")
    
    args = parser.parse_args()
    
    # Set experiment name
    experiment_name = args.experiment or f"{args.ticker.lower()}_{args.mode}_v1"
    
    print("\n" + "="*80)
    print("🏦 PRODUCTION PPO TRADING SYSTEM")
    print("="*80)
    print(f"Ticker: {args.ticker}")
    print(f"Mode: {args.mode}")
    print(f"Experiment: {experiment_name}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
    print("="*80)
    
    # Load data
    df = load_and_prepare_data(args.ticker)
    
    # Split data
    train_df, val_df = create_train_val_split(df, train_ratio=0.8)
    
    # Create environments
    print("\n🏗️  Creating environments...")
    
    # Adjust config for available data
    if args.mode == "production":
        env_config = SwingTradingConfig.conservative()
    else:
        env_config = SwingTradingConfig()
    
    # For testing with limited data, reduce lookback and episode length
    from swing_trading.config import EnvironmentConfig, TradingConfig
    from dataclasses import replace
    
    # Ensure we have enough data for validation
    min_required = 150  # Minimum rows needed
    if len(val_df) < min_required:
        print(f"⚠️  Validation data too short ({len(val_df)} rows), adjusting split...")
        # Use more data for validation
        split_idx = max(len(df) - min_required - 50, int(len(df) * 0.7))
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]
        print(f"   New training: {len(train_df)} rows")
        print(f"   New validation: {len(val_df)} rows")
    
    # Adjust environment config for available data
    lookback = min(60, len(val_df) // 3)
    episode_length = min(126, len(val_df) - lookback - 10)
    
    env_cfg = replace(
        env_config.environment,
        lookback=lookback,
        episode_length=episode_length,
    )
    env_config = replace(env_config, environment=env_cfg)
    
    train_env = SwingTradingEnv(train_df, env_config)
    val_env = SwingTradingEnv(val_df, env_config)

    
    print(f"✅ Training environment created")
    print(f"   Observation space: {train_env.observation_space.shape}")
    print(f"   Action space: {train_env.action_space.shape}")
    
    # Create training configuration
    print("\n⚙️  Configuring training...")
    
    if args.mode == "production":
        training_config = TrainingConfig.production(experiment_name)
        risk_config = RiskConfig.institutional()
    else:
        training_config = TrainingConfig.research(experiment_name)
        risk_config = RiskConfig.research()
    
    # Override timesteps if specified
    if args.timesteps:
        from dataclasses import replace
        training_config = replace(training_config, total_timesteps=args.timesteps)
    
    # Override GPU if specified
    if args.gpu is not None:
        from trading_system.config import DeviceConfig
        device_config = DeviceConfig(device='cuda', gpu_id=args.gpu)
        training_config = replace(training_config, device=device_config)
    
    # Override seed
    training_config = replace(training_config, seed=args.seed)
    
    print(f"✅ Configuration created")
    print(f"   Total timesteps: {training_config.total_timesteps:,}")
    print(f"   Learning rate: {training_config.ppo.learning_rate}")
    print(f"   Batch size: {training_config.ppo.batch_size}")
    print(f"   Max drawdown limit: {risk_config.max_drawdown_pct:.1%}")
    
    # Create trainer
    print("\n🚀 Initializing trainer...")
    trainer = PPOTrainer(
        env=train_env,
        config=training_config,
        risk_config=risk_config,
        eval_env=val_env,
    )
    
    # Train
    try:
        trainer.train()
        
        # Save final model
        final_model_path = f"models/{experiment_name}_final"
        trainer.save(final_model_path)
        
        # Evaluate
        print("\n📊 Running final evaluation...")
        metrics = trainer.evaluate(n_episodes=20)
        
        # Print summary
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Model saved to: {final_model_path}")
        print(f"TensorBoard logs: {training_config.logging.tensorboard_log}")
        print(f"\nTo view training progress:")
        print(f"  tensorboard --logdir {training_config.logging.tensorboard_log}")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        trainer.save(f"models/{experiment_name}_interrupted")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

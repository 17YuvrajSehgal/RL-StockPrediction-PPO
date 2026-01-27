"""
Walk-Forward Validation Script for RL Trading.

This script implements a rolling window approach:
1. Train on N years (e.g., 2 years)
2. Test on M months (e.g., 3 months)
3. Move window forward by M months
4. Repeat to form a continuous out-of-sample equity curve

Usage:
    python walk_forward.py --ticker IAU --start-date 2020-01-01 --train-years 2 --test-months 3
"""

import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from dataclasses import replace

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from swing_trading import SwingTradingEnv, SwingTradingConfig
from swing_trading.margin import MarginModel
from trading_system.config import TrainingConfig, RiskConfig
from trading_system.backtesting.backtest_engine import BacktestEngine, BacktestResult

def load_data(ticker: str, data_dir: str = "yf_data") -> pd.DataFrame:
    """Load and prepare data."""
    print(f"Loading data for {ticker}...")
    data_path = Path(data_dir)
    csv_files = list(data_path.glob(f"{ticker}*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No data file found for {ticker} in {data_dir}")
    
    df = pd.read_csv(csv_files[0])
    
    # Standard cleaning
    new_cols = {}
    for col in df.columns:
        if 'Date' in col: new_cols[col] = 'Date'
        elif 'Open' in col: new_cols[col] = 'Open'
        elif 'High' in col: new_cols[col] = 'High'
        elif 'Low' in col: new_cols[col] = 'Low'
        elif 'Close' in col and 'Adj' not in col: new_cols[col] = 'Close'
        elif 'Volume' in col: new_cols[col] = 'Volume'
    
    df = df.rename(columns=new_cols)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    df.columns = [c.lower() for c in df.columns]
    
    return df

def train_model_for_window(df_train: pd.DataFrame, experiment_name: str, timesteps: int = 100000) -> str:
    """Train a model on the specific window."""
    
    # Config
    env_config = SwingTradingConfig.conservative()
    
    # Adjust lookback/episode length for smaller windows if needed
    effective_len = len(df_train)
    lookback = 60
    episode_length = min(252, effective_len - lookback - 10)
    
    env_cfg = replace(env_config.environment, lookback=lookback, episode_length=episode_length)
    env_config = replace(env_config, environment=env_cfg)
    
    train_env = SwingTradingEnv(df_train, env_config)
    
    # Vectorize and Normalize
    env = DummyVecEnv([lambda: train_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    
    # Training Config
    training_config = TrainingConfig.production(experiment_name)
    # Reduce epochs/steps for speed in walk-forward if desired, but keep robust
    # Using defaults: 100k timesteps passed as arg
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=0,
        tensorboard_log=None # Disable TB for individual VF steps to save space
    )
    
    
    # Eval Callback to check if model is learning
    from stable_baselines3.common.callbacks import EvalCallback
    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"models/wf/best/{experiment_name}",
        log_path=f"models/wf/logs/{experiment_name}",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # Train
    model.learn(total_timesteps=timesteps, progress_bar=True, callback=eval_callback)

    
    # Save
    model_path = f"models/wf/{experiment_name}"
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    env.save(f"{model_path}_vecnormalize.pkl")
    
    return model_path

def run_backtest_for_window(model_path: str, df_test: pd.DataFrame) -> BacktestResult:
    """Run backtest on the test window."""
    
    # Load model and env stats
    model = PPO.load(model_path)
    
    # Create Env
    env_config = SwingTradingConfig()
    
    # CRITICAL: Lookback must match training
    lookback = 60 
    
    # For backtesting a small slice, we need PREVIOUS data for lookback
    # This is handled by passing a larger DF to BacktestEngine usually, 
    # but here we might just have the test slice.
    # Actually, BacktestEngine expects the 'data' to contain everything needed.
    # So if df_test starts at T, and we need T-60, df_test passed here MUST have that history?
    # NO. BacktestEngine handles data. But we need to be careful.
    # The clean way: Pass df_test which INCLUDES the lookback buffer.
    
    effective_rows = len(df_test) - 50 # Buffer for MA
    episode_length = max(1, effective_rows - lookback - 1)
    
    env_cfg = replace(env_config.environment, lookback=lookback, episode_length=episode_length, random_start=False)
    env_config = replace(env_config, environment=env_cfg)
    
    # Engine
    engine = BacktestEngine(
        model_path=model_path,
        data=df_test,
        config=env_config,
        initial_capital=100000.0, # Reset capital? Or carry over? 
                                  # Ideally carry over, but for simplicity we run 100k 
                                  # and calculate % return then apply to master equity.
        transaction_cost=0.001,
        debug=False
    )
    
    return engine.run(verbose=False)

def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Validation")
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--start-date", type=str, default="2020-01-01", help="Start of the first TEST period")
    parser.add_argument("--train-years", type=int, default=2, help="Years of history to train on")
    parser.add_argument("--test-months", type=int, default=3, help="Months to test forward")
    parser.add_argument("--timesteps", type=int, default=100000, help="Training steps per window")
    
    args = parser.parse_args()
    
    # Load Data
    full_df = load_data(args.ticker)
    
    # Define Walk-Forward Loop
    test_start = pd.to_datetime(args.start_date)
    final_date = full_df.index[-1]
    
    results = []
    equity_curve_segments = []
    
    current_test_start = test_start
    
    print(f"\n🚀 Starting Walk-Forward Validation for {args.ticker}")
    print(f"   Train Window: {args.train_years} years")
    print(f"   Test Window:  {args.test_months} months")
    print(f"   Start Date:   {test_start.date()}")
    print("="*80)
    
    master_equity = 100000.0
    master_equity_curve = []
    master_dates = []
    
    step_count = 0
    
    while current_test_start < final_date:
        step_count += 1
        
        # Define Time Ranges
        train_start = current_test_start - relativedelta(years=args.train_years)
        test_end = current_test_start + relativedelta(months=args.test_months)
        
        # Clip test end
        if test_end > final_date:
            test_end = final_date
            
        print(f"\n🔄 Step {step_count}: Test Period {current_test_start.date()} -> {test_end.date()}")
        
        # Slice Data
        # Training Data
        mask_train = (full_df.index >= train_start) & (full_df.index < current_test_start)
        df_train = full_df[mask_train]
        
        if len(df_train) < 200:
            print("⚠️  Insufficient training data. Skipping.")
            break
            
        # Testing Data
        # We need a buffer for lookback (60) + indicators (50) = ~110 days before test_start
        buffer_start = current_test_start - timedelta(days=200) 
        mask_test_buffer = (full_df.index >= buffer_start) & (full_df.index <= test_end)
        df_test_buffered = full_df[mask_test_buffer]
        
        # 1. Train
        print(f"   Training on {len(df_train)} rows ({train_start.date()} -> {current_test_start.date()})...")
        exp_name = f"{args.ticker.lower()}_wf_{current_test_start.strftime('%Y%m%d')}"
        model_path = train_model_for_window(df_train, exp_name, args.timesteps)
        
        # 2. Test
        print(f"   Testing...")
        result = run_backtest_for_window(model_path, df_test_buffered)
        
        # 3. Stitch Results
        # Extract only the part of the curve inside the actual test window
        segment = result.equity_curve
        # Filter strictly for test window
        segment = segment[(segment.index >= current_test_start) & (segment.index <= test_end)]
        
        if not segment.empty:
            # Calculate returns for this segment
            # We treat the first point of segment as the continuation of master equity
            segment_pct_returns = segment.pct_change().fillna(0)
            
            # Apply to master equity
            for date, pct in segment_pct_returns.items():
                master_equity *= (1 + pct)
                master_equity_curve.append(master_equity)
                master_dates.append(date)
                
            total_ret_seg = (segment.iloc[-1] - segment.iloc[0]) / segment.iloc[0]
            print(f"   Segment Return: {total_ret_seg:.2%}")
            
            results.append({
                'start': current_test_start,
                'end': test_end,
                'return': total_ret_seg,
                'trades': len(result.trades)
            })
        
        # Advance Window
        current_test_start = test_end
    
    # Final Report
    print("\n" + "="*80)
    print("🏁 WALK-FORWARD COMPLETE")
    print("="*80)
    
    if master_equity_curve:
        final_return = (master_equity - 100000) / 100000
        print(f"Final Equity: ${master_equity:,.2f}")
        print(f"Total Return: {final_return:.2%}")
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(master_dates, master_equity_curve)
        plt.title(f"Walk-Forward Equity Curve: {args.ticker}")
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"reports/{args.ticker}_walk_forward_equity.png")
        print(f"Chart saved to reports/{args.ticker}_walk_forward_equity.png")
    
    # Save CSV
    df_res = pd.DataFrame(results)
    df_res.to_csv(f"reports/{args.ticker}_walk_forward_segments.csv")

if __name__ == "__main__":
    main()

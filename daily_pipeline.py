#!/usr/bin/env python3
"""
Daily Trading Pipeline - Command Line Interface

This script provides a comprehensive CLI for the daily trading system:
- Generate trading signals for the next trading day
- Run walk-forward validation
- Compare training modes (full vs fine-tune)
- Train models on demand

Usage:
    # Generate signal for AAPL for next Monday
    python daily_pipeline.py signal --ticker AAPL
    
    # Run walk-forward validation for last 30 days
    python daily_pipeline.py validate --ticker AAPL --days 30
    
    # Compare training modes
    python daily_pipeline.py compare --ticker AAPL --days 10
    
    # Full training for specific date
    python daily_pipeline.py train --ticker AAPL --date 2026-01-24 --mode full
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

# Ensure parent directory is in path
sys.path.insert(0, str(Path(__file__).parent))

from daily_trading.config import (
    DailyTradingConfig,
    DailyTrainingConfig,
    TrainingMode,
    TrainingWindowConfig,
    SignalConfig,
    WalkForwardConfig,
    PathConfig,
)
from daily_trading.trainer import DailyTrainer, TrainResult
from daily_trading.signals import SignalGenerator, TradingSignal
from daily_trading.walk_forward import WalkForwardValidator, WalkForwardResult, quick_validate
from daily_trading.utils import TradingCalendar, DataLoader, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Daily Trading Pipeline for RL Stock Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate signal for next trading day
  python daily_pipeline.py signal --ticker AAPL
  
  # Generate signal for specific date
  python daily_pipeline.py signal --ticker AAPL --for-date 2026-01-27
  
  # Run walk-forward validation for last 30 trading days
  python daily_pipeline.py validate --ticker AAPL --days 30
  
  # Run validation with specific date range
  python daily_pipeline.py validate --ticker AAPL --start 2025-10-01 --end 2025-12-31
  
  # Compare full training vs fine-tuning
  python daily_pipeline.py compare --ticker AAPL --days 10
  
  # Train model for specific date
  python daily_pipeline.py train --ticker AAPL --date 2026-01-24 --mode full
  
  # Quick 5-day validation test
  python daily_pipeline.py quick-test --ticker AAPL
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # =====================================================================
    # Signal command
    # =====================================================================
    signal_parser = subparsers.add_parser(
        "signal",
        help="Generate trading signal for next trading day"
    )
    signal_parser.add_argument(
        "--ticker", "-t",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g., AAPL)"
    )
    signal_parser.add_argument(
        "--for-date", "-d",
        type=str,
        default=None,
        help="Date to generate signal for (YYYY-MM-DD). Default: next trading day"
    )
    signal_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model (optional, uses latest if not specified)"
    )
    signal_parser.add_argument(
        "--buy-threshold",
        type=float,
        default=0.3,
        help="Weight threshold for BUY signal (default: 0.3)"
    )
    signal_parser.add_argument(
        "--sell-threshold",
        type=float,
        default=-0.3,
        help="Weight threshold for SELL signal (default: -0.3)"
    )
    
    # =====================================================================
    # Validate command
    # =====================================================================
    validate_parser = subparsers.add_parser(
        "validate",
        help="Run walk-forward validation"
    )
    validate_parser.add_argument(
        "--ticker", "-t",
        type=str,
        required=True,
        help="Stock ticker symbol"
    )
    validate_parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Number of trading days to test (from most recent)"
    )
    validate_parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD)"
    )
    validate_parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD)"
    )
    validate_parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "fine_tune"],
        default="fine_tune",
        help="Training mode (default: fine_tune)"
    )
    validate_parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Training timesteps per day (default: mode-dependent)"
    )
    validate_parser.add_argument(
        "--window",
        type=int,
        default=504,
        help="Training window in trading days (default: 504 = ~2 years)"
    )
    
    # =====================================================================
    # Compare command
    # =====================================================================
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare full training vs fine-tuning"
    )
    compare_parser.add_argument(
        "--ticker", "-t",
        type=str,
        required=True,
        help="Stock ticker symbol"
    )
    compare_parser.add_argument(
        "--days",
        type=int,
        default=10,
        help="Number of trading days to test (default: 10)"
    )
    compare_parser.add_argument(
        "--timesteps-full",
        type=int,
        default=100_000,
        help="Timesteps for full training (default: 100000)"
    )
    compare_parser.add_argument(
        "--timesteps-finetune",
        type=int,
        default=25_000,
        help="Timesteps for fine-tuning (default: 25000)"
    )
    
    # =====================================================================
    # Train command
    # =====================================================================
    train_parser = subparsers.add_parser(
        "train",
        help="Train model for specific date"
    )
    train_parser.add_argument(
        "--ticker", "-t",
        type=str,
        required=True,
        help="Stock ticker symbol"
    )
    train_parser.add_argument(
        "--date", "-d",
        type=str,
        required=True,
        help="Training cutoff date (YYYY-MM-DD)"
    )
    train_parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "fine_tune"],
        default="full",
        help="Training mode (default: full)"
    )
    train_parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Training timesteps"
    )
    train_parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model for fine-tuning"
    )
    train_parser.add_argument(
        "--window",
        type=int,
        default=504,
        help="Training window in trading days (default: 504)"
    )
    
    # =====================================================================
    # Quick-test command
    # =====================================================================
    quick_parser = subparsers.add_parser(
        "quick-test",
        help="Quick 5-day validation test with minimal training"
    )
    quick_parser.add_argument(
        "--ticker", "-t",
        type=str,
        required=True,
        help="Stock ticker symbol"
    )
    quick_parser.add_argument(
        "--days",
        type=int,
        default=5,
        help="Number of days to test (default: 5)"
    )
    quick_parser.add_argument(
        "--timesteps",
        type=int,
        default=10_000,
        help="Timesteps per training (default: 10000)"
    )
    
    # Global options
    parser.add_argument(
        "--data-dir",
        type=str,
        default="yf_data",
        help="Directory containing market data (default: yf_data)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug logging"
    )
    
    return parser.parse_args()


def cmd_signal(args: argparse.Namespace, paths: PathConfig) -> int:
    """Generate trading signal command."""
    print("\n" + "=" * 70)
    print("GENERATING TRADING SIGNAL")
    print("=" * 70)
    
    calendar = TradingCalendar()
    
    # Determine signal date
    if args.for_date:
        for_date = date.fromisoformat(args.for_date)
    else:
        today = date.today()
        for_date = calendar.next_trading_day(today)
    
    print(f"Ticker: {args.ticker}")
    print(f"Signal for: {for_date}")
    print("=" * 70)
    
    # Create signal generator
    signal_config = SignalConfig(
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
    )
    
    generator = SignalGenerator(signal_config, paths)
    
    try:
        model_path = Path(args.model) if args.model else None
        signal = generator.generate(
            ticker=args.ticker,
            for_date=for_date,
            model_path=model_path,
        )
        
        print(signal.format_report())
        return 0
        
    except Exception as e:
        print(f"\nError generating signal: {e}")
        return 1


def cmd_validate(args: argparse.Namespace, paths: PathConfig) -> int:
    """Run walk-forward validation command."""
    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION")
    print("=" * 70)
    
    calendar = TradingCalendar()
    data_loader = DataLoader(paths.data_dir)
    
    # Determine date range
    if args.start and args.end:
        start_date = date.fromisoformat(args.start)
        end_date = date.fromisoformat(args.end)
    elif args.days:
        latest = data_loader.get_latest_date(args.ticker)
        end_date = calendar.previous_trading_day(latest)
        
        # Go back N trading days
        start_date = end_date
        for _ in range(args.days):
            start_date = calendar.previous_trading_day(start_date)
    else:
        print("Error: Specify --days or --start/--end")
        return 1
    
    print(f"Ticker: {args.ticker}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Mode: {args.mode}")
    print(f"Window: {args.window} trading days")
    print("=" * 70 + "\n")
    
    # Create config
    training_mode = TrainingMode.FULL if args.mode == "full" else TrainingMode.FINE_TUNE
    window_config = TrainingWindowConfig.rolling(args.window)
    
    training_config = DailyTrainingConfig(
        mode=training_mode,
        window=window_config,
        timesteps_full=args.timesteps or 200_000,
        timesteps_fine_tune=args.timesteps or 50_000,
    )
    
    wf_config = WalkForwardConfig(
        training_window=window_config,
        training_config=training_config,
    )
    
    validator = WalkForwardValidator(wf_config, paths)
    
    try:
        result = validator.validate(
            ticker=args.ticker,
            start_date=start_date,
            end_date=end_date,
            training_mode=training_mode,
            timesteps=args.timesteps,
        )
        
        print("\n" + result.summary())
        return 0
        
    except Exception as e:
        print(f"\nValidation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_compare(args: argparse.Namespace, paths: PathConfig) -> int:
    """Compare training modes command."""
    print("\n" + "=" * 70)
    print("TRAINING MODE COMPARISON")
    print("=" * 70)
    
    calendar = TradingCalendar()
    data_loader = DataLoader(paths.data_dir)
    
    # Determine date range
    latest = data_loader.get_latest_date(args.ticker)
    end_date = calendar.previous_trading_day(latest)
    
    start_date = end_date
    for _ in range(args.days):
        start_date = calendar.previous_trading_day(start_date)
    
    print(f"Ticker: {args.ticker}")
    print(f"Period: {start_date} to {end_date} ({args.days} days)")
    print(f"Full train timesteps: {args.timesteps_full:,}")
    print(f"Fine-tune timesteps: {args.timesteps_finetune:,}")
    print("=" * 70 + "\n")
    
    # Create config
    wf_config = WalkForwardConfig(
        training_config=DailyTrainingConfig(
            timesteps_full=args.timesteps_full,
            timesteps_fine_tune=args.timesteps_finetune,
        ),
    )
    
    validator = WalkForwardValidator(wf_config, paths)
    
    try:
        results = validator.validate_comparison(
            ticker=args.ticker,
            start_date=start_date,
            end_date=end_date,
        )
        return 0
        
    except Exception as e:
        print(f"\nComparison failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_train(args: argparse.Namespace, paths: PathConfig) -> int:
    """Train model command."""
    print("\n" + "=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)
    
    as_of_date = date.fromisoformat(args.date)
    training_mode = TrainingMode.FULL if args.mode == "full" else TrainingMode.FINE_TUNE
    
    print(f"Ticker: {args.ticker}")
    print(f"As of date: {as_of_date}")
    print(f"Mode: {args.mode}")
    print(f"Window: {args.window} trading days")
    print("=" * 70 + "\n")
    
    # Create config
    window_config = TrainingWindowConfig.rolling(args.window)
    
    training_config = DailyTrainingConfig(
        mode=training_mode,
        window=window_config,
        timesteps_full=args.timesteps or 500_000,
        timesteps_fine_tune=args.timesteps or 50_000,
    )
    
    trainer = DailyTrainer(training_config, paths)
    
    try:
        base_model = Path(args.base_model) if args.base_model else None
        
        result = trainer.train(
            ticker=args.ticker,
            as_of_date=as_of_date,
            mode=training_mode,
            base_model_path=base_model,
            timesteps=args.timesteps,
        )
        
        if result.success:
            print("\n" + "=" * 70)
            print("TRAINING COMPLETED")
            print("=" * 70)
            print(f"Model saved to: {result.model_path}")
            print(f"Training time: {result.training_time_seconds:.1f} seconds")
            print(f"Final reward: {result.final_reward:.4f}")
            print("=" * 70)
            return 0
        else:
            print(f"\nTraining failed: {result.error}")
            return 1
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_quick_test(args: argparse.Namespace, paths: PathConfig) -> int:
    """Quick test command."""
    print("\n" + "=" * 70)
    print("QUICK VALIDATION TEST")
    print("=" * 70)
    print(f"Ticker: {args.ticker}")
    print(f"Days: {args.days}")
    print(f"Timesteps: {args.timesteps:,}")
    print("=" * 70 + "\n")
    
    try:
        result = quick_validate(
            ticker=args.ticker,
            days=args.days,
            timesteps=args.timesteps,
            paths=paths,
        )
        
        print("\n" + result.summary())
        return 0
        
    except Exception as e:
        print(f"\nQuick test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    if not args.command:
        print("Error: No command specified. Use --help for usage.")
        return 1
    
    # Setup logging
    level = logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING)
    setup_logging("daily_trading", level)
    
    # Setup paths
    paths = PathConfig(base_dir=Path.cwd())
    paths = PathConfig(
        base_dir=Path.cwd(),
        data_subdir=args.data_dir,
    )
    paths.ensure_directories()
    
    # Dispatch command
    if args.command == "signal":
        return cmd_signal(args, paths)
    elif args.command == "validate":
        return cmd_validate(args, paths)
    elif args.command == "compare":
        return cmd_compare(args, paths)
    elif args.command == "train":
        return cmd_train(args, paths)
    elif args.command == "quick-test":
        return cmd_quick_test(args, paths)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

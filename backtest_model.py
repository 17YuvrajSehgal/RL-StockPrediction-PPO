"""
Command-line interface for backtesting PPO trading models.

Professional backtesting tool with:
- Easy model loading
- Automatic data handling
- Comprehensive reporting
- Interactive visualizations

Usage:
    python backtest_model.py --model models/AAPL_final --ticker AAPL
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from swing_trading import SwingTradingConfig
from trading_system.backtesting.backtest_engine import BacktestEngine
from trading_system.backtesting.report_generator import ReportGenerator


def load_data(ticker: str, data_dir: str = "yf_data") -> pd.DataFrame:
    """
    Load OHLCV data for ticker.
    
    Args:
        ticker: Stock ticker symbol
        data_dir: Directory containing data files
    
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Loading data for {ticker}...")
    
    # Find data file
    data_path = Path(data_dir)
    csv_files = list(data_path.glob(f"{ticker}*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No data file found for {ticker} in {data_dir}")
    
    df = pd.read_csv(csv_files[0])
    
    # Fix column names
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
    
    print(f"Loaded {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
    
    return df


def filter_date_range(df: pd.DataFrame, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Filter DataFrame by date range.
    
    Args:
        df: DataFrame with datetime index
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        Filtered DataFrame
    """
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]
    
    return df


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Backtest PPO trading model with professional reporting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic backtest
  python backtest_model.py --model models/AAPL_production_final --ticker AAPL
  
  # Specify date range
  python backtest_model.py --model models/AAPL_final --ticker AAPL \\
      --start-date 2024-11-01 --end-date 2026-01-22
  
  # Custom output directory
  python backtest_model.py --model models/AAPL_final --ticker AAPL \\
      --output-dir reports/aapl_backtest_validation
  
  # With custom initial capital
  python backtest_model.py --model models/AAPL_final --ticker AAPL \\
      --initial-capital 500000
"""
    )
    
    # Required arguments
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model (without extension)'
    )
    parser.add_argument(
        '--ticker',
        type=str,
        required=True,
        help='Stock ticker symbol'
    )
    
    # Optional arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        default='yf_data',
        help='Directory containing data files (default: yf_data)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for reports (default: reports/<ticker>_<timestamp>)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date for backtest (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date for backtest (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=100000.0,
        help='Initial capital (default: 100000)'
    )
    parser.add_argument(
        '--transaction-cost',
        type=float,
        default=0.001,
        help='Transaction cost rate (default: 0.001 = 10 bps)'
    )
    parser.add_argument(
        '--no-charts',
        action='store_true',
        help='Skip chart generation (faster)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable detailed step-by-step logging for debugging'
    )
    parser.add_argument(
        '--no-margin',
        action='store_true',
        help='Disable margin constraint enforcement (for comparison)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*70)
    print("PROFESSIONAL BACKTESTING SYSTEM")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Ticker: {args.ticker}")
    print(f"Initial Capital: ${args.initial_capital:,.2f}")
    print("="*70 + "\n")
    
    try:
        # Load data
        df = load_data(args.ticker, args.data_dir)
        
        # Filter date range
        if args.start_date or args.end_date:
            print(f"Filtering date range...")
            df = filter_date_range(df, args.start_date, args.end_date)
            print(f"Using {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
        
        # Create environment config
        config = SwingTradingConfig()
        
        # Adjust config for available data
        from swing_trading.config import EnvironmentConfig
        from dataclasses import replace
        
        # Account for ~50 rows lost during feature engineering (MA warmup)
        # IMPORTANT: lookback MUST match training (default 60) for observation space compatibility
        effective_rows = len(df) - 50
        lookback = 60  # Must match training - don't change!
        episode_length = min(252, max(20, effective_rows - lookback - 10))
        
        # Check if we have enough data
        min_required = lookback + episode_length + 10
        if effective_rows < min_required:
            print(f"\n⚠️  Warning: Limited data ({len(df)} rows, ~{effective_rows} after features)")
            print(f"   Backtest will run on {episode_length} trading days")
            print(f"   For longer backtest, use earlier --start-date\n")
        
        env_cfg = replace(
            config.environment,
            lookback=lookback,
            episode_length=episode_length,
            random_start=False,  # Deterministic for backtesting
        )
        config = replace(config, environment=env_cfg)
        
        # Create backtest engine
        engine = BacktestEngine(
            model_path=args.model,
            data=df,
            config=config,
            initial_capital=args.initial_capital,
            transaction_cost=args.transaction_cost,
            debug=args.debug,
            enforce_margin=not args.no_margin,
        )
        
        # Run backtest
        result = engine.run(verbose=True)
        
        # Determine output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"reports/{args.ticker}_backtest_{timestamp}"
        
        # Generate reports
        if not args.no_charts:
            generator = ReportGenerator(result, df)
            generator.generate_all_reports(output_dir)
        else:
            # Just save trade log
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            result.to_dataframe().to_csv(f"{output_dir}/trades.csv", index=False)
            print(f"Trade log saved to {output_dir}/trades.csv")
        
        # Print final summary
        print("\n" + result.summary())
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(f"   Make sure data file exists in {args.data_dir}/")
        return 1
    except Exception as e:
        print(f"\nBacktest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

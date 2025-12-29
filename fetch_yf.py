#!/usr/bin/env python3
"""
Download historical OHLCV data from Yahoo Finance using yfinance and save to disk.

Supports:
- Multiple tickers
- Period-based or start/end date queries
- Multiple intervals (daily, weekly, etc.)
- CSV or Parquet output
- Separate or combined output files

Example usage:
  python fetch_yf.py --tickers AAPL MSFT TSLA --start 2023-01-01 --end 2023-12-31 --interval 1d --out data
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import List, Optional

import pandas as pd

# -----------------------------
# Dependency check
# -----------------------------
try:
    import yfinance as yf
except ImportError as e:
    # Fail early with clear instructions
    raise SystemExit(
        "Missing dependency: yfinance\n"
        "Install with: pip install yfinance pandas\n"
    ) from e


# -----------------------------
# Command-line argument parsing
# -----------------------------
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments controlling what data is downloaded and how it is saved.
    """
    p = argparse.ArgumentParser(description="Fetch historical price data from Yahoo Finance.")

    # Required inputs
    p.add_argument("--tickers", nargs="+", required=True,
                   help="One or more ticker symbols (e.g., AAPL MSFT TSLA)")

    # Output options
    p.add_argument("--out", default="yf_data",
                   help="Output directory (default: yf_data)")
    p.add_argument("--format", choices=["csv", "parquet"], default="csv",
                   help="Output file format (default: csv)")
    p.add_argument("--group", choices=["separate", "combined"], default="separate",
                   help="Save tickers separately or as one combined file")

    # Date range options (choose ONE mode)
    p.add_argument("--period", default=None,
                   help="Yahoo Finance period (e.g., 1y, 5y, max)")
    p.add_argument("--start", default=None,
                   help="Start date YYYY-MM-DD (used if --period not set)")
    p.add_argument("--end", default=None,
                   help="End date YYYY-MM-DD (used if --period not set)")

    # Market data options
    p.add_argument("--interval", default="1d",
                   help="Data interval (e.g., 1d, 1wk, 1mo)")
    p.add_argument("--adjust", action="store_true",
                   help="Auto-adjust prices for dividends/splits")
    p.add_argument("--actions", action="store_true",
                   help="Include dividends and splits columns")
    p.add_argument("--prepost", action="store_true",
                   help="Include pre/post market data")

    return p.parse_args()


# -----------------------------
# Input validation
# -----------------------------
def validate_dates(start: Optional[str], end: Optional[str]) -> None:
    """
    Validate that provided start/end dates follow YYYY-MM-DD format.
    """
    for name, val in [("start", start), ("end", end)]:
        if val is None:
            continue
        try:
            datetime.strptime(val, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid {name} date '{val}'. Expected YYYY-MM-DD.")


# -----------------------------
# Download a single ticker
# -----------------------------
def download_one(
    ticker: str,
    period: Optional[str],
    start: Optional[str],
    end: Optional[str],
    interval: str,
    auto_adjust: bool,
    actions: bool,
    prepost: bool,
) -> pd.DataFrame:
    """
    Download OHLCV data for a single ticker using yfinance.
    Returns an empty DataFrame if no data is available.
    """
    df = yf.download(
        tickers=ticker,
        period=period,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        actions=actions,
        prepost=prepost,
        progress=False,
        threads=True,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Convert DatetimeIndex to a regular column for easier saving
    df = df.reset_index()

    # Normalize column names (spaces → underscores)
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]

    # Add ticker symbol explicitly as a column
    df.insert(0, "Ticker", ticker)

    return df


# -----------------------------
# Save DataFrame to disk
# -----------------------------
def save_df(df: pd.DataFrame, path: str, fmt: str) -> None:
    """
    Save DataFrame to CSV or Parquet format.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if fmt == "csv":
        df.to_csv(path, index=False)
    else:
        # Parquet requires additional dependencies
        try:
            df.to_parquet(path, index=False)
        except Exception as e:
            raise SystemExit(
                "Failed to write parquet. Install one of:\n"
                "  pip install pyarrow\n"
                "  pip install fastparquet\n"
                f"Original error: {e}"
            )


# -----------------------------
# Main program
# -----------------------------
def main() -> int:
    """
    Entry point:
    - Parse arguments
    - Validate inputs
    - Download requested tickers
    - Save data to disk
    """
    args = parse_args()
    validate_dates(args.start, args.end)

    # Ensure valid date selection mode
    if args.period is None and (args.start is None or args.end is None):
        print("Error: Provide either --period OR both --start and --end.", file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)

    all_rows: List[pd.DataFrame] = []

    # Download each ticker individually
    for t in args.tickers:
        print(f"Downloading {t} ...")

        df = download_one(
            ticker=t,
            period=args.period,
            start=args.start,
            end=args.end,
            interval=args.interval,
            auto_adjust=args.adjust,
            actions=args.actions,
            prepost=args.prepost,
        )

        if df.empty:
            print(f"  -> No data returned for {t}", file=sys.stderr)
            continue

        # Save individual files if requested
        if args.group == "separate":
            safe_t = t.replace("/", "_").replace(" ", "_")
            fname = f"{safe_t}_{args.interval}"

            if args.period:
                fname += f"_{args.period}"
            else:
                fname += f"_{args.start}_to_{args.end}"

            fpath = os.path.join(args.out, f"{fname}.{args.format}")
            save_df(df, fpath, args.format)
            print(f"  -> saved: {fpath}")

        all_rows.append(df)

    # Save combined file if requested
    if args.group == "combined" and all_rows:
        combined = pd.concat(all_rows, ignore_index=True)

        fname = f"combined_{args.interval}"
        if args.period:
            fname += f"_{args.period}"
        else:
            fname += f"_{args.start}_to_{args.end}"

        fpath = os.path.join(args.out, f"{fname}.{args.format}")
        save_df(combined, fpath, args.format)
        print(f"Saved combined file: {fpath}")

    if not all_rows:
        print("No data downloaded (all tickers returned empty).", file=sys.stderr)
        return 1

    return 0


# -----------------------------
# Script entry point
# -----------------------------
if __name__ == "__main__":
    raise SystemExit(main())


# -----------------------------
# Example commands
# -----------------------------
# python fetch_yf.py --tickers AAPL MSFT --start 2020-01-01 --end 2025-12-23 --interval 1d --out data --group combined
# python fetch_yf.py --tickers AAPL --start 2020-01-01 --end 2025-12-23 --interval 1d

"""
run_hft.py — Command-line entry point for the HFT arbitrage engine.

Usage examples:

  # Run pairs arb for 1 hour on paper trading (default):
  python run_hft.py

  # Run for 30 minutes:
  python run_hft.py --duration 1800

  # Run on live trading (CAUTION — real money):
  python run_hft.py --mode live

  # Custom pair, tighter z-score, smaller order size:
  python run_hft.py --zscore-entry 1.5 --order-size 50

  # Verbose signal logging:
  python run_hft.py --debug-signals

  # Custom risk limits:
  python run_hft.py --max-daily-loss 500 --max-total-notional 50000

  # Override TWS connection details:
  python run_hft.py --host 127.0.0.1 --port 7497 --client-id 2
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys


def parse_args(argv=None) -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="run_hft",
        description="HFT Statistical Pairs Arbitrage — IBKR Paper/Live Trading",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Session ──────────────────────────────────────────────────────────
    session = parser.add_argument_group("Session")
    session.add_argument(
        "--duration", type=float, default=None,
        metavar="SECONDS",
        help="Run duration in seconds. Omit to run until Ctrl+C.",
    )
    session.add_argument(
        "--mode", choices=["paper", "live"], default="paper",
        help="Trading mode. 'paper' is safe; 'live' uses real capital.",
    )

    # ── IBKR connection ───────────────────────────────────────────────
    conn = parser.add_argument_group("IBKR Connection")
    conn.add_argument("--host", default="127.0.0.1", help="TWS/Gateway host")
    conn.add_argument(
        "--port", type=int, default=None,
        help=(
            "TWS/Gateway port. "
            "Defaults to 7497 (paper) or 7496 (live) based on --mode."
        ),
    )
    conn.add_argument(
        "--client-id", type=int, default=1, dest="client_id",
        help="IB client ID (must be unique per connection to TWS).",
    )

    # ── Strategy ──────────────────────────────────────────────────────
    strat = parser.add_argument_group("Pairs Arb Strategy")
    strat.add_argument(
        "--zscore-entry", type=float, default=2.0, dest="zscore_entry",
        help="Z-score threshold to enter a trade.",
    )
    strat.add_argument(
        "--zscore-exit", type=float, default=0.5, dest="zscore_exit",
        help="Z-score threshold to exit a trade (must be < entry).",
    )
    strat.add_argument(
        "--lookback", type=int, default=100,
        help="Rolling window size (ticks) for spread statistics.",
    )
    strat.add_argument(
        "--order-size", type=int, default=100, dest="order_size",
        help="Shares per leg per signal.",
    )
    strat.add_argument(
        "--cooldown", type=float, default=5.0,
        help="Minimum seconds between signals for the same pair.",
    )
    strat.add_argument(
        "--limit-offset", type=float, default=0.01, dest="limit_offset",
        help="Limit order price offset from mid in dollars.",
    )

    # ── Risk ──────────────────────────────────────────────────────────
    risk = parser.add_argument_group("Risk Limits")
    risk.add_argument(
        "--max-daily-loss", type=float, default=1000.0, dest="max_daily_loss",
        help="Auto-halt when session loss exceeds this value (USD).",
    )
    risk.add_argument(
        "--max-total-notional", type=float, default=100_000.0,
        dest="max_total_notional",
        help="Maximum total portfolio notional exposure (USD).",
    )
    risk.add_argument(
        "--max-position", type=int, default=2000, dest="max_position",
        help="Maximum shares held per symbol (absolute value).",
    )
    risk.add_argument(
        "--max-orders-per-second", type=float, default=45.0,
        dest="max_orders_per_second",
        help="Order submission rate cap (IBKR hard limit is 50/s).",
    )

    # ── Logging ───────────────────────────────────────────────────────
    log_grp = parser.add_argument_group("Logging")
    log_grp.add_argument(
        "--log-dir", default="hft_logs", dest="log_dir",
        help="Directory for log files and fill CSV.",
    )
    log_grp.add_argument(
        "--no-csv", action="store_true", dest="no_csv",
        help="Disable CSV fill logging.",
    )
    log_grp.add_argument(
        "--dashboard-interval", type=float, default=10.0,
        dest="dashboard_interval",
        help="Seconds between live dashboard prints.",
    )
    log_grp.add_argument(
        "--debug-signals", action="store_true", dest="debug_signals",
        help="Log every evaluated z-score at DEBUG level (very verbose).",
    )
    log_grp.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO", dest="log_level",
        help="Console log level.",
    )

    args = parser.parse_args(argv)

    # Validation
    if args.zscore_exit >= args.zscore_entry:
        parser.error(
            f"--zscore-exit ({args.zscore_exit}) must be less than "
            f"--zscore-entry ({args.zscore_entry})"
        )

    return args


def build_config(args: argparse.Namespace):
    """Translate parsed CLI args into IBKRConfig + HFTConfig objects."""
    from ibkr.config import IBKRConfig, TradingMode
    from hft.config import (
        HFTConfig, PairsArbConfig, RiskConfig,
        ExecutorConfig, LoggingConfig,
    )

    # ── IBKR config ──────────────────────────────────────────────────
    trading_mode = TradingMode.PAPER if args.mode == "paper" else TradingMode.LIVE
    port = args.port or (7497 if trading_mode == TradingMode.PAPER else 7496)

    ibkr_config = IBKRConfig(
        host=args.host,
        port=port,
        client_id=args.client_id,
        trading_mode=trading_mode,
        log_level=args.log_level,
    )

    # ── Strategy config ───────────────────────────────────────────────
    pairs_arb_cfg = PairsArbConfig(
        zscore_entry=args.zscore_entry,
        zscore_exit=args.zscore_exit,
        lookback=args.lookback,
        order_size=args.order_size,
        cooldown_secs=args.cooldown,
        limit_offset=args.limit_offset,
    )

    # ── Risk config ───────────────────────────────────────────────────
    risk_cfg = RiskConfig(
        max_daily_loss=args.max_daily_loss,
        max_total_notional=args.max_total_notional,
        max_position_per_symbol=args.max_position,
        max_orders_per_second=args.max_orders_per_second,
    )

    # ── Executor config ───────────────────────────────────────────────
    executor_cfg = ExecutorConfig(cancel_on_shutdown=True)

    # ── Logging config ────────────────────────────────────────────────
    log_cfg = LoggingConfig(
        log_dir=args.log_dir,
        save_fills_csv=not args.no_csv,
        dashboard_interval=args.dashboard_interval,
        debug_signals=args.debug_signals,
    )

    # ── Compose top-level HFTConfig ───────────────────────────────────
    hft_config = HFTConfig(
        pairs_arb=pairs_arb_cfg,
        risk=risk_cfg,
        executor=executor_cfg,
        logging=log_cfg,
        enable_pairs_arb=True,
        enable_etf_arb=False,
        enable_market_making=False,
    )

    return ibkr_config, hft_config


async def main(argv=None) -> int:
    """Async main — parse args, build config, run engine, return exit code."""
    args = parse_args(argv)

    # Configure root logger before importing engine to catch all messages
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    from hft.engine import HFTEngine

    ibkr_config, hft_config = build_config(args)

    # Safety prompt for live mode
    if args.mode == "live":
        print("\n" + "⚠️ " * 20)
        print("  WARNING: LIVE TRADING MODE — Real money at risk!")
        print("⚠️ " * 20 + "\n")
        confirm = input("Type 'CONFIRM' to proceed: ").strip()
        if confirm != "CONFIRM":
            print("Cancelled.")
            return 0

    # Print startup summary
    print("\n" + "=" * 60)
    print("  HFT PAIRS ARBITRAGE ENGINE")
    print("=" * 60)
    print(f"  Mode          : {args.mode.upper()}")
    print(f"  TWS           : {ibkr_config.host}:{ibkr_config.port} (client {ibkr_config.client_id})")
    print(f"  Pairs         : {[f'{a}/{b}' for a, b in hft_config.pairs_arb.pairs]}")
    print(f"  Z-score       : entry={hft_config.pairs_arb.zscore_entry}, exit={hft_config.pairs_arb.zscore_exit}")
    print(f"  Order size    : {hft_config.pairs_arb.order_size} shares/leg")
    print(f"  Limit offset  : ${hft_config.pairs_arb.limit_offset:.3f}")
    print(f"  Max daily loss: ${hft_config.risk.max_daily_loss:,.0f}")
    print(f"  Duration      : {f'{args.duration:.0f}s' if args.duration else 'until Ctrl+C'}")
    print(f"  Logs          : {args.log_dir}/")
    print("=" * 60 + "\n")

    engine = HFTEngine(ibkr_config, hft_config)

    try:
        await engine.run(duration_seconds=args.duration)
    except KeyboardInterrupt:
        pass   # engine handles Ctrl+C via SIGINT handler

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

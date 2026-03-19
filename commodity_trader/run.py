"""
Main entry point to run the Commodity HFT Trading Bot.
"""
import asyncio
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ibkr.config import IBKRConfig
from commodity_trader.config import (
    CommodityTraderConfig, IndicatorConfig, SignalConfig, PositionConfig
)
from commodity_trader.engine import CommodityEngine


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    ibkr_config = IBKRConfig.paper_trading()

    trader_config = CommodityTraderConfig(
        indicators=IndicatorConfig(
            bar_duration_secs=10.0,   # 10-second bars
            rsi_period=9,
            ema_fast=5,
            ema_slow=13,
        ),
        signal=SignalConfig(
            rsi_entry=58.0,
            rsi_exit=42.0,
            vwap_dev_entry_pct=0.05,  # price must be 0.05% above VWAP
            ema_confirmation=True,
            cooldown_secs=3.0,
            min_spread_bps=5.0,
        ),
        position=PositionConfig(
            order_size=500,
            stop_loss_bps=15.0,
            trailing_stop_bps=20.0,
            trail_distance_bps=10.0,
            limit_offset=0.01,
            limit_timeout_secs=8.0,
            max_daily_loss_usd=500.0,
        ),
    )

    engine = CommodityEngine(ibkr_config, trader_config)

    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        print("\nInterrupted by user, exiting...")


if __name__ == "__main__":
    main()

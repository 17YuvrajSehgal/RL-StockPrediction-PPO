"""
HFT package — High-Frequency Arbitrage Trading System.

Built on top of the ibkr package which wraps ib_async for Interactive Brokers
TWS / IB Gateway connectivity.

Primary entry points:
  - HFTEngine  : top-level orchestrator (start/stop/run)
  - HFTConfig  : all tunable parameters in one frozen dataclass

Typical usage:
    >>> from hft import HFTEngine, HFTConfig
    >>> from ibkr import IBKRConfig
    >>>
    >>> hft_config = HFTConfig()
    >>> ibkr_config = IBKRConfig.paper_trading()
    >>> engine = HFTEngine(ibkr_config, hft_config)
    >>> asyncio.run(engine.run(duration_seconds=3600))

Note on imports:
    Pure-logic modules (config, signals) do NOT import ib_async, so they
    can be imported and unit-tested without a broker.  The broker-dependent
    modules (engine, executor) are imported lazily via TYPE_CHECKING guards
    or on-demand in run_hft.py.
"""

# Always safe to import — no ib_async dependency
from hft.config import HFTConfig, PairsArbConfig, RiskConfig, ExecutorConfig

__version__ = "0.1.0"

__all__ = [
    "HFTConfig",
    "PairsArbConfig",
    "RiskConfig",
    "ExecutorConfig",
    # HFTEngine is NOT re-exported here to avoid pulling in ib_async at
    # package import time.  Import it directly: from hft.engine import HFTEngine
]

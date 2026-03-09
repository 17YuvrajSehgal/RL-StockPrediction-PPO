"""
HFT Engine — main orchestrator for the high-frequency arbitrage system.

This is the top-level class that wires all components together:
  - IBKRConnection      : broker connectivity
  - MarketDataManager   : real-time L1 quote feeds
  - OrderManager        : order placement (via ibkr package)
  - PositionManager     : open-position tracking
  - RiskManager         : pre-flight risk checks
  - PairsArbDetector(s) : one per configured pair
  - HFTExecutor         : async order queue + dispatch
  - PerformanceTracker  : metrics, dashboard, CSV logging

Signal flow (fully event-driven, no polling):
  TWS tick → MarketDataManager → Quote callback → PairsArbDetector
           → Signal → HFTExecutor.submit_signal()
           → asyncio.Queue → _process_request()
           → RiskManager.check_order() → OrderManager.place_limit_order()
           → ib_async order-status event → fill callback → RiskManager.record_fill()
           → PerformanceTracker.record_fill()

Usage:
    >>> from hft import HFTEngine, HFTConfig
    >>> from ibkr import IBKRConfig
    >>>
    >>> engine = HFTEngine(IBKRConfig.paper_trading(), HFTConfig())
    >>> asyncio.run(engine.run(duration_seconds=3600))
"""

from __future__ import annotations

import asyncio
import logging
import signal as _signal
import sys
from datetime import datetime
from typing import Optional

from ibkr.config import IBKRConfig
from ibkr.connection import IBKRConnection
from ibkr.market_data import MarketDataManager, Quote
from ibkr.risk import RiskManager
from ibkr.trading import OrderManager
from ibkr.positions import PositionManager
from ibkr.logging_config import setup_logging

from hft.config import HFTConfig
from hft.signals import PairsArbDetector, Signal
from hft.executor import HFTExecutor
from hft.performance import PerformanceTracker

logger = logging.getLogger(__name__)


class HFTEngine:
    """
    Top-level orchestrator for the HFT arbitrage system.

    Composes all components and manages the session lifecycle.  Designed
    to be run via asyncio.run() from the entry-point script.

    Args:
        ibkr_config: IBKR connection configuration (host, port, mode, etc.)
        hft_config:  HFT system configuration (pairs, risk limits, etc.)
    """

    def __init__(
        self,
        ibkr_config: IBKRConfig,
        hft_config:  HFTConfig,
    ) -> None:
        self._ibkr_cfg = ibkr_config
        self._hft_cfg  = hft_config

        # ---- Component placeholders (initialised in start()) ----

        self._connection:  Optional[IBKRConnection]   = None
        self._mkt_data:    Optional[MarketDataManager] = None
        self._order_mgr:   Optional[OrderManager]      = None
        self._pos_mgr:     Optional[PositionManager]   = None
        self._risk_mgr:    Optional[RiskManager]       = None
        self._executor:    Optional[HFTExecutor]       = None
        self._tracker:     Optional[PerformanceTracker] = None

        # One detector per configured pair
        self._detectors: list[PairsArbDetector] = []

        # Session state
        self._running   = False
        self._stop_event = asyncio.Event()

        logger.info(f"HFTEngine created — {hft_config}")

    # ------------------------------------------------------------------
    # Public – lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Connect to TWS/Gateway and initialise all components.

        Raises:
            IBKRConnectionError: If the broker connection fails
        """
        logger.info("=" * 60)
        logger.info("  HFT ARBITRAGE ENGINE STARTING")
        logger.info("=" * 60)

        # 1. Configure logging
        setup_logging(
            log_dir=self._hft_cfg.logging.log_dir,
            log_level=logging.INFO,
        )

        # 2. Connect to IBKR
        logger.info(f"Connecting to IBKR ({self._ibkr_cfg.trading_mode})...")
        self._connection = IBKRConnection(self._ibkr_cfg)
        await self._connection.connect()
        ib = self._connection.ib

        # 3. Instantiate broker-layer components
        self._order_mgr = OrderManager(ib)
        self._pos_mgr   = PositionManager(ib)

        # 4. Instantiate risk manager
        risk_limits = self._hft_cfg.to_risk_limits()
        self._risk_mgr = RiskManager(risk_limits)

        # 5. Instantiate performance tracker and start dashboard
        self._tracker = PerformanceTracker(self._hft_cfg.logging)
        await self._tracker.start()

        # 6. Instantiate executor and wire fill tracking
        self._executor = HFTExecutor(
            order_manager=self._order_mgr,
            risk_manager=self._risk_mgr,
            exec_config=self._hft_cfg.executor,
            pairs_config=self._hft_cfg.pairs_arb,
        )
        self._executor.subscribe_fills(self._tracker.record_fill)
        await self._executor.start()

        # 7. Build signal detectors
        self._build_detectors()

        # 8. Start market data subscriptions
        symbols = self._hft_cfg.all_symbols()
        self._mkt_data = MarketDataManager(ib)
        for sym in symbols:
            await self._mkt_data.subscribe(sym)
            logger.info(f"  Subscribed: {sym}")

        # Wire the quote callback → detector fan-out
        self._mkt_data.subscribe_quotes(self._on_quote)

        self._running = True
        logger.info("═" * 60)
        logger.info(
            f"  ENGINE LIVE — {len(self._detectors)} pair(s), "
            f"{len(symbols)} symbol(s)"
        )
        logger.info("═" * 60)

    async def stop(self) -> None:
        """
        Gracefully shut down: drain orders, log final summary, disconnect.
        """
        if not self._running:
            return

        logger.info("Stopping HFT engine...")
        self._running = False
        self._stop_event.set()

        # Unsubscribe from market data first (stop new signals)
        if self._mkt_data:
            self._mkt_data.unsubscribe_all()

        # Drain and stop executor
        if self._executor:
            await self._executor.stop()

        # Stop dashboard and print final summary
        if self._tracker:
            await self._tracker.stop()

        # Disconnect from broker
        if self._connection:
            await self._connection.disconnect()

        logger.info("HFT engine stopped cleanly")

    async def run(self, duration_seconds: Optional[float] = None) -> None:
        """
        Start the engine, run for `duration_seconds`, then stop.

        If `duration_seconds` is None the engine runs until a SIGINT (Ctrl+C)
        is received or stop() is called externally.

        Args:
            duration_seconds: How long to run (None = run indefinitely)
        """
        # Register OS-level interrupt handler so Ctrl+C triggers clean stop
        self._register_signal_handlers()

        await self.start()

        try:
            if duration_seconds is not None:
                logger.info(f"Running for {duration_seconds:.0f} seconds...")
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=duration_seconds,
                )
            else:
                logger.info("Running indefinitely (Ctrl+C to stop)...")
                await self._stop_event.wait()

        except asyncio.TimeoutError:
            logger.info(f"Session duration ({duration_seconds:.0f}s) elapsed — stopping")
        except asyncio.CancelledError:
            logger.info("Run cancelled — stopping")
        finally:
            await self.stop()

    # ------------------------------------------------------------------
    # Public – runtime controls
    # ------------------------------------------------------------------

    def halt_trading(self, reason: str = "Manual halt via engine") -> None:
        """
        Immediately halt all new order submissions via the risk manager.

        Does NOT close existing positions. Use the executor's cancel logic
        for that.

        Args:
            reason: Explanation logged to the risk manager
        """
        if self._risk_mgr:
            self._risk_mgr.halt(reason)
            logger.warning(f"Trading halted: {reason}")

    def resume_trading(self) -> None:
        """Resume trading after a manual halt."""
        if self._risk_mgr:
            self._risk_mgr.resume()

    def get_health(self) -> dict:
        """
        Return a health snapshot combining connection and risk state.

        Useful for external monitoring or a web dashboard.
        """
        health: dict = {
            "running":    self._running,
            "timestamp":  datetime.now().isoformat(),
        }
        if self._connection:
            health["broker_connected"] = self._connection.is_connected
        if self._risk_mgr:
            health["risk"] = self._risk_mgr.summary()
        if self._tracker:
            summary = self._tracker.get_summary()
            health["session"] = {
                "signals":     summary.total_signals,
                "orders_sent": summary.total_orders_sent,
                "orders_filled": summary.total_orders_filled,
                "fill_rate":   summary.fill_rate,
                "realized_pnl": summary.realized_pnl,
            }
        if self._executor:
            stats = self._executor.get_stats()
            health["executor"] = {
                "queue_depth":       stats.queue_depth,
                "avg_fill_latency":  stats.avg_fill_latency_ms,
            }
        return health

    # ------------------------------------------------------------------
    # Private – component construction
    # ------------------------------------------------------------------

    def _build_detectors(self) -> None:
        """Construct one PairsArbDetector for each configured pair."""
        if not self._hft_cfg.enable_pairs_arb:
            logger.info("Pairs arb disabled — no detectors created")
            return

        cfg = self._hft_cfg.pairs_arb
        for sym_a, sym_b in cfg.pairs:
            detector = PairsArbDetector(
                symbol_a=sym_a,
                symbol_b=sym_b,
                lookback=cfg.lookback,
                zscore_entry=cfg.zscore_entry,
                zscore_exit=cfg.zscore_exit,
                order_size=cfg.order_size,
                cooldown_secs=cfg.cooldown_secs,
                debug=self._hft_cfg.logging.debug_signals,
            )
            self._detectors.append(detector)
            logger.info(f"  Detector: {sym_a}/{sym_b}")

    # ------------------------------------------------------------------
    # Private – event-driven quote callback
    # ------------------------------------------------------------------

    def _on_quote(self, quote: Quote) -> None:
        """
        Called by MarketDataManager on every incoming tick.

        Fans the quote out to all registered detectors.  If a detector
        returns a Signal it is immediately handed to the executor's
        non-blocking submit path.

        This is the hot path — keep it fast (no I/O, no blocking calls).
        """
        if not self._running:
            return

        # Skip non-tradeable quotes (missing bid or ask)
        if not quote.is_tradeable:
            return

        mid = quote.mid

        for detector in self._detectors:
            try:
                signal: Optional[Signal] = detector.on_quote(quote.symbol, mid)
                if signal is not None:
                    self._tracker.increment_signals()
                    self._executor.submit_signal(signal)
            except Exception as exc:
                logger.error(f"Detector error ({detector.symbol_a}/{detector.symbol_b}): {exc}")

    # ------------------------------------------------------------------
    # Private – OS signal handling
    # ------------------------------------------------------------------

    def _register_signal_handlers(self) -> None:
        """Register SIGINT / SIGTERM handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()

        def _shutdown(sig_name: str) -> None:
            logger.warning(f"Received {sig_name} — initiating clean shutdown")
            self._stop_event.set()

        for sig in (_signal.SIGINT, _signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda s=sig.name: _shutdown(s))
            except (NotImplementedError, OSError):
                # Windows does not support add_signal_handler for all signals
                pass

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "RUNNING" if self._running else "STOPPED"
        return (
            f"HFTEngine("
            f"status={status}, "
            f"detectors={len(self._detectors)}, "
            f"mode={self._ibkr_cfg.trading_mode})"
        )

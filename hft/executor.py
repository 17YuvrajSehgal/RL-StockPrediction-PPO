"""
Async order execution engine for high-frequency trading.

The executor sits between the signal layer and the broker.  It:
  1. Accepts order submissions via a non-blocking async queue
  2. Runs a pre-flight risk check on every order
  3. Places limit (or market) orders via ibkr.trading.OrderManager
  4. Applies a token-bucket rate limiter (shared with RiskManager)
  5. Monitors fills and feeds confirmed executions back to RiskManager
  6. Gathers latency and fill-rate statistics for the performance tracker

Design goals:
  - Submit path is non-blocking (put_nowait into asyncio.Queue)
  - Processing happens in a single background asyncio task (no thread-safety issues)
  - Fill callbacks are push-driven via ib_async events (no polling)
  - All state transitions are logged with enough detail for post-session analysis
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from ibkr.trading import OrderManager, OrderInfo, OrderAction, OrderStatus
from ibkr.risk import RiskManager
from hft.config import ExecutorConfig, PairsArbConfig
from hft.signals import Signal, SignalType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Execution record
# ---------------------------------------------------------------------------

@dataclass
class ExecutionRecord:
    """
    Full audit trail for a single order submitted by the executor.

    Attributes:
        symbol:         Ticker
        action:         "BUY" or "SELL"
        quantity:       Shares ordered
        limit_price:    Submitted limit price (None if market order)
        submit_time:    Monotonic timestamp at submission
        ack_time:       Monotonic timestamp when order was acknowledged by IB
        fill_time:      Monotonic timestamp when order reached FILLED status
        fill_price:     Actual execution price
        order_info:     Final OrderInfo snapshot from the broker
        signal:         The Signal that triggered this execution (if any)
        rejected:       True if risk check blocked this order
        reject_reason:  Human-readable rejection explanation
    """
    symbol:       str
    action:       str
    quantity:     int
    limit_price:  Optional[float]
    submit_time:  float = field(default_factory=time.monotonic)
    ack_time:     Optional[float] = None
    fill_time:    Optional[float] = None
    fill_price:   Optional[float] = None
    order_info:   Optional[OrderInfo] = None
    signal:       Optional[Signal] = None
    rejected:     bool = False
    reject_reason: str = ""

    @property
    def ack_latency_ms(self) -> Optional[float]:
        """Milliseconds from submit to broker ACK."""
        if self.ack_time:
            return (self.ack_time - self.submit_time) * 1000
        return None

    @property
    def fill_latency_ms(self) -> Optional[float]:
        """Milliseconds from submit to fill."""
        if self.fill_time:
            return (self.fill_time - self.submit_time) * 1000
        return None

    @property
    def is_filled(self) -> bool:
        return self.fill_time is not None

    def __repr__(self) -> str:
        status = "FILLED" if self.is_filled else ("REJECTED" if self.rejected else "PENDING")
        return (
            f"ExecutionRecord({status} {self.action} {self.quantity} "
            f"{self.symbol} @ {self.limit_price})"
        )


# ---------------------------------------------------------------------------
# Statistics snapshot
# ---------------------------------------------------------------------------

@dataclass
class ExecutorStats:
    """Point-in-time statistics snapshot from the executor."""
    orders_submitted:   int
    orders_rejected:    int
    orders_filled:      int
    orders_cancelled:   int
    orders_pending:     int
    fill_rate:          float    # filled / submitted (0.0 – 1.0)
    avg_ack_latency_ms: float
    avg_fill_latency_ms: float
    queue_depth:        int


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class HFTExecutor:
    """
    Async, rate-limited order execution engine.

    The executor owns a single background coroutine (_process_queue) that
    drains a ``asyncio.Queue`` of pending orders, applies risk checks, and
    dispatches to OrderManager.  Fill events come back through an ib_async
    order-status callback.

    Usage:
        >>> executor = HFTExecutor(order_mgr, risk_mgr, exec_cfg, pairs_cfg)
        >>> await executor.start()
        >>>
        >>> # Signal detected — non-blocking submit:
        >>> executor.submit_signal(signal)
        >>>
        >>> await executor.stop()

    Args:
        order_manager: Connected OrderManager instance
        risk_manager:  RiskManager instance (shared with engine)
        exec_config:   ExecutorConfig (queue size, timeouts, etc.)
        pairs_config:  PairsArbConfig (limit offset, order size, etc.)
    """

    def __init__(
        self,
        order_manager: OrderManager,
        risk_manager:  RiskManager,
        exec_config:   ExecutorConfig,
        pairs_config:  PairsArbConfig,
    ) -> None:
        self._order_mgr   = order_manager
        self._risk_mgr    = risk_manager
        self._exec_cfg    = exec_config
        self._pairs_cfg   = pairs_config

        # Bounded async queue — excess orders are dropped with a warning
        self._queue: asyncio.Queue[_OrderRequest] = asyncio.Queue(
            maxsize=exec_config.queue_max_size
        )

        # All execution records (submitted + rejected) for analysis
        self._records: list[ExecutionRecord] = []

        # order_id → ExecutionRecord (for fill correlation)
        self._pending: dict[int, ExecutionRecord] = {}

        # Fill update callbacks (performance tracker hooks in here)
        self._fill_callbacks: list[Callable[[ExecutionRecord], None]] = []

        # Background processing task
        self._task: Optional[asyncio.Task] = None
        self._running = False

        logger.info(
            f"HFTExecutor initialised — "
            f"queue_size={exec_config.queue_max_size}, "
            f"fill_timeout={exec_config.fill_timeout}s"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background order-processing coroutine."""
        if self._running:
            logger.warning("Executor already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._process_queue(), name="hft_executor")
        # Hook into order status events for fill tracking
        self._order_mgr.subscribe_order_updates(self._on_order_update)
        logger.info("HFTExecutor started")

    async def stop(self) -> None:
        """
        Gracefully stop the executor.

        Drains the queue, optionally cancels open orders, then cancels
        the background task.
        """
        self._running = False

        # Let the queue drain (up to 2 seconds)
        try:
            await asyncio.wait_for(self._queue.join(), timeout=2.0)
        except asyncio.TimeoutError:
            logger.warning("Queue not fully drained on stop — some orders may be lost")

        if self._exec_cfg.cancel_on_shutdown:
            await self._cancel_all_pending()

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info(f"HFTExecutor stopped — {len(self._records)} total orders processed")

    # ------------------------------------------------------------------
    # Public – signal submission
    # ------------------------------------------------------------------

    def submit_signal(self, signal: Signal) -> None:
        """
        Convert a Signal into one or two order submissions and enqueue them.

        This is a fire-and-forget call (non-blocking). The signal determines
        the action for each leg:

          ENTER_LONG_A  → BUY symbol_a, SELL symbol_b
          ENTER_SHORT_A → SELL symbol_a, BUY symbol_b
          EXIT          → BUY (if currently SHORT_A) or SELL (if LONG_A)
                          — exit direction is looked up from context

        For EXIT signals, the executor places opposing orders for both legs.

        Args:
            signal: Signal emitted by a detector
        """
        quantity = signal.quantity

        if signal.type == SignalType.ENTER_LONG_A:
            # Long A / Short B entry
            self._enqueue(signal.symbol_a, "BUY",  quantity, signal.mid_a, signal)
            self._enqueue(signal.symbol_b, "SELL", quantity, signal.mid_b, signal)

        elif signal.type == SignalType.ENTER_SHORT_A:
            # Short A / Long B entry
            self._enqueue(signal.symbol_a, "SELL", quantity, signal.mid_a, signal)
            self._enqueue(signal.symbol_b, "BUY",  quantity, signal.mid_b, signal)

        elif signal.type == SignalType.EXIT:
            # Close both legs — for an exit we reverse each leg.
            # We determine direction by looking at current risk state.
            pos_a = self._risk_mgr.get_position(signal.symbol_a)
            pos_b = self._risk_mgr.get_position(signal.symbol_b)

            if pos_a > 0:
                self._enqueue(signal.symbol_a, "SELL", abs(pos_a), signal.mid_a, signal)
            elif pos_a < 0:
                self._enqueue(signal.symbol_a, "BUY",  abs(pos_a), signal.mid_a, signal)

            if pos_b > 0:
                self._enqueue(signal.symbol_b, "SELL", abs(pos_b), signal.mid_b, signal)
            elif pos_b < 0:
                self._enqueue(signal.symbol_b, "BUY",  abs(pos_b), signal.mid_b, signal)

        else:
            logger.warning(f"Unknown signal type: {signal.type}")

    def submit_order(
        self,
        symbol:   str,
        action:   str,
        quantity: int,
        price:    float,
        signal:   Optional[Signal] = None,
    ) -> None:
        """
        Directly enqueue a single order (not signal-driven).

        Args:
            symbol:   Ticker
            action:   "BUY" or "SELL"
            quantity: Shares
            price:    Limit price
            signal:   Originating signal (optional, for record-keeping)
        """
        self._enqueue(symbol, action, quantity, price, signal)

    # ------------------------------------------------------------------
    # Public – fill subscriptions
    # ------------------------------------------------------------------

    def subscribe_fills(self, callback: Callable[[ExecutionRecord], None]) -> None:
        """
        Register a callback that fires every time an order is filled.

        Args:
            callback: Callable[[ExecutionRecord], None]
        """
        self._fill_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Public – stats
    # ------------------------------------------------------------------

    def get_stats(self) -> ExecutorStats:
        """Return a point-in-time statistics snapshot."""
        submitted  = len(self._records)
        rejected   = sum(1 for r in self._records if r.rejected)
        filled     = sum(1 for r in self._records if r.is_filled)
        cancelled  = sum(
            1 for r in self._records
            if r.order_info and r.order_info.is_cancelled()
        )
        pending    = len(self._pending)

        ack_latencies  = [r.ack_latency_ms  for r in self._records if r.ack_latency_ms  is not None]
        fill_latencies = [r.fill_latency_ms for r in self._records if r.fill_latency_ms is not None]

        return ExecutorStats(
            orders_submitted=submitted,
            orders_rejected=rejected,
            orders_filled=filled,
            orders_cancelled=cancelled,
            orders_pending=pending,
            fill_rate=filled / submitted if submitted > 0 else 0.0,
            avg_ack_latency_ms=sum(ack_latencies)   / len(ack_latencies)  if ack_latencies  else 0.0,
            avg_fill_latency_ms=sum(fill_latencies) / len(fill_latencies) if fill_latencies else 0.0,
            queue_depth=self._queue.qsize(),
        )

    # ------------------------------------------------------------------
    # Private – queue helpers
    # ------------------------------------------------------------------

    def _enqueue(
        self,
        symbol:   str,
        action:   str,
        quantity: int,
        mid:      float,
        signal:   Optional[Signal],
    ) -> None:
        """Compute the limit price and put the request into the queue."""
        # Apply limit offset: buy slightly below mid, sell slightly above
        if action.upper() == "BUY":
            limit_price = round(mid - self._pairs_cfg.limit_offset, 4)
        else:
            limit_price = round(mid + self._pairs_cfg.limit_offset, 4)

        request = _OrderRequest(
            symbol=symbol,
            action=action,
            quantity=quantity,
            limit_price=limit_price,
            signal=signal,
        )

        try:
            self._queue.put_nowait(request)
        except asyncio.QueueFull:
            logger.warning(
                f"Order queue full ({self._exec_cfg.queue_max_size}) — "
                f"dropping {action} {quantity} {symbol}"
            )

    # ------------------------------------------------------------------
    # Private – background processor
    # ------------------------------------------------------------------

    async def _process_queue(self) -> None:
        """
        Background coroutine that drains the order queue.

        Runs until self._running is False and the queue is empty.
        """
        logger.debug("Executor queue processor started")
        while self._running or not self._queue.empty():
            try:
                request = await asyncio.wait_for(self._queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            try:
                await self._process_request(request)
            except Exception as exc:
                logger.error(f"Error processing order request: {exc}", exc_info=True)
            finally:
                self._queue.task_done()

        logger.debug("Executor queue processor exiting")

    async def _process_request(self, request: "_OrderRequest") -> None:
        """Validate, risk-check, and place a single order."""
        record = ExecutionRecord(
            symbol=request.symbol,
            action=request.action,
            quantity=request.quantity,
            limit_price=request.limit_price,
            submit_time=time.monotonic(),
            signal=request.signal,
        )

        # --- Risk check ---
        result = self._risk_mgr.check_order(
            request.symbol,
            request.action,
            request.quantity,
            request.limit_price,
        )

        if not result.approved:
            record.rejected     = True
            record.reject_reason = result.reason
            self._records.append(record)
            logger.debug(f"Order rejected: {result.reason}")
            return

        # --- Place order ---
        try:
            order_info = await self._order_mgr.place_limit_order(
                symbol=request.symbol,
                quantity=request.quantity,
                action=request.action,
                limit_price=request.limit_price,
                outside_rth=False,
            )

            record.ack_time  = time.monotonic()
            record.order_info = order_info

            # Track for fill correlation
            self._pending[order_info.order_id] = record
            self._records.append(record)

            logger.info(
                f"Order placed: {request.action} {request.quantity} "
                f"{request.symbol} @ ${request.limit_price:.4f} "
                f"[id={order_info.order_id}]"
            )

        except Exception as exc:
            record.rejected = True
            record.reject_reason = f"Broker error: {exc}"
            self._records.append(record)
            logger.error(f"Order placement failed for {request.symbol}: {exc}")

    # ------------------------------------------------------------------
    # Private – fill tracking
    # ------------------------------------------------------------------

    def _on_order_update(self, order_info: OrderInfo) -> None:
        """
        Callback wired into OrderManager.subscribe_order_updates().

        Updates the pending record when our orders fill.
        """
        oid = order_info.order_id

        if oid not in self._pending:
            return   # Not one of ours

        record = self._pending[oid]
        record.order_info = order_info

        if order_info.status == OrderStatus.FILLED:
            record.fill_time  = time.monotonic()
            record.fill_price = order_info.avg_fill_price

            # Notify risk manager
            self._risk_mgr.record_fill(
                record.symbol,
                record.action,
                order_info.filled_quantity,
                order_info.avg_fill_price,
            )
            if order_info.pnl:
                self._risk_mgr.record_pnl(order_info.pnl)

            # Remove from pending
            del self._pending[oid]

            logger.info(
                f"Fill confirmed: {record.action} {order_info.filled_quantity} "
                f"{record.symbol} @ ${order_info.avg_fill_price:.4f} "
                f"[latency={record.fill_latency_ms:.0f}ms]"
            )

            # Fan out to fill subscribers
            for cb in self._fill_callbacks:
                try:
                    cb(record)
                except Exception as exc:
                    logger.error(f"Fill callback error: {exc}")

        elif order_info.status == OrderStatus.CANCELLED:
            logger.info(f"Order {oid} cancelled for {record.symbol}")
            del self._pending[oid]

    # ------------------------------------------------------------------
    # Private – cleanup
    # ------------------------------------------------------------------

    async def _cancel_all_pending(self) -> None:
        """Cancel all outstanding open orders on shutdown."""
        if not self._pending:
            return

        logger.info(f"Cancelling {len(self._pending)} open orders on shutdown...")
        for oid in list(self._pending.keys()):
            try:
                await self._order_mgr.cancel_order(oid)
            except Exception as exc:
                logger.warning(f"Could not cancel order {oid}: {exc}")

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"HFTExecutor("
            f"submitted={stats.orders_submitted}, "
            f"filled={stats.orders_filled}, "
            f"fill_rate={stats.fill_rate:.1%}, "
            f"queue={stats.queue_depth})"
        )


# ---------------------------------------------------------------------------
# Internal request struct (private to this module)
# ---------------------------------------------------------------------------

@dataclass
class _OrderRequest:
    """Internal queue item — not part of the public API."""
    symbol:      str
    action:      str
    quantity:    int
    limit_price: float
    signal:      Optional[Signal] = None

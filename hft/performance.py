"""
Performance tracking and dashboard for the HFT system.

Tracks session-level statistics in real-time and provides both a rich
console dashboard and a CSV fill log for post-session analysis.

Responsibilities:
  - Receive fill notifications from HFTExecutor (push-based)
  - Track cumulative P&L, order counts, latency stats, per-symbol metrics
  - Render a formatted console dashboard on a configurable interval
  - Optionally flush fill records to a timestamped CSV file

Usage:
    >>> tracker = PerformanceTracker(config)
    >>> await tracker.start()
    >>> executor.subscribe_fills(tracker.record_fill)
    >>> # ... runs during session ...
    >>> await tracker.stop()
    >>> tracker.print_summary()
"""

from __future__ import annotations

import asyncio
import csv
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from hft.config import LoggingConfig
from hft.executor import ExecutionRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-symbol metrics
# ---------------------------------------------------------------------------

@dataclass
class SymbolMetrics:
    """Aggregated per-symbol statistics for a session."""
    symbol:          str
    fills:           int   = 0
    total_shares:    int   = 0
    total_notional:  float = 0.0
    realized_pnl:    float = 0.0
    avg_fill_price:  float = 0.0
    avg_latency_ms:  float = 0.0
    _latency_sum:    float = field(default=0.0, repr=False)
    _pnl_basis:      float = field(default=0.0, repr=False)   # cost basis for unrealised

    def update(self, record: ExecutionRecord) -> None:
        """Integrate a single fill record."""
        if not record.is_filled or record.fill_price is None:
            return

        self.fills        += 1
        self.total_shares += record.order_info.filled_quantity if record.order_info else record.quantity
        shares = record.order_info.filled_quantity if record.order_info else record.quantity
        self.total_notional += shares * record.fill_price

        # Running average fill price
        self.avg_fill_price = self.total_notional / self.total_shares if self.total_shares else 0.0

        # Realised PnL from OrderInfo
        if record.order_info and record.order_info.pnl:
            self.realized_pnl += record.order_info.pnl

        # Running average latency
        if record.fill_latency_ms is not None:
            self._latency_sum += record.fill_latency_ms
            self.avg_latency_ms = self._latency_sum / self.fills


# ---------------------------------------------------------------------------
# Session summary
# ---------------------------------------------------------------------------

@dataclass
class SessionSummary:
    """Complete session snapshot for display or logging."""
    start_time:          datetime
    elapsed:             timedelta
    total_signals:       int
    total_orders_sent:   int
    total_orders_filled: int
    total_orders_rejected: int
    fill_rate:           float
    realized_pnl:        float
    avg_ack_latency_ms:  float
    avg_fill_latency_ms: float
    orders_per_minute:   float
    symbol_metrics:      dict[str, SymbolMetrics]

    def __str__(self) -> str:
        elapsed_str = str(self.elapsed).split(".")[0]   # HH:MM:SS
        lines = [
            "╔══════════════════════════════════════════════════╗",
            "║          HFT SESSION PERFORMANCE SUMMARY         ║",
            "╠══════════════════════════════════════════════════╣",
            f"║  Runtime          : {elapsed_str:<28s}  ║",
            f"║  Signals fired    : {self.total_signals:<28d}  ║",
            f"║  Orders sent      : {self.total_orders_sent:<28d}  ║",
            f"║  Orders filled    : {self.total_orders_filled:<28d}  ║",
            f"║  Orders rejected  : {self.total_orders_rejected:<28d}  ║",
            f"║  Fill rate        : {self.fill_rate:<27.1%}  ║",
            f"║  Realized P&L     : ${self.realized_pnl:<27.2f}  ║",
            f"║  Orders / minute  : {self.orders_per_minute:<28.1f}  ║",
            f"║  Avg ACK latency  : {self.avg_ack_latency_ms:<25.1f} ms  ║",
            f"║  Avg fill latency : {self.avg_fill_latency_ms:<25.1f} ms  ║",
            "╠══════════════════════════════════════════════════╣",
            "║  Symbol           Fills   PnL        Avg Price   ║",
            "╠══════════════════════════════════════════════════╣",
        ]
        for sym, m in sorted(self.symbol_metrics.items()):
            lines.append(
                f"║  {sym:<10s}  {m.fills:>6d}   ${m.realized_pnl:>+9.2f}   ${m.avg_fill_price:>9.4f}  ║"
            )
        lines.append("╚══════════════════════════════════════════════════╝")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Performance tracker
# ---------------------------------------------------------------------------

class PerformanceTracker:
    """
    Real-time performance tracker and dashboard for the HFT engine.

    Receives fill notifications from HFTExecutor.subscribe_fills(),
    aggregates per-symbol and session-level metrics, and prints periodic
    live dashboards to stdout.  Optionally writes fills to a CSV file for
    post-session analysis.

    Args:
        config: LoggingConfig from HFTConfig
    """

    def __init__(self, config: LoggingConfig) -> None:
        self._config = config
        self._start_time = datetime.now()

        # Aggregate counters
        self._total_signals:   int   = 0
        self._total_sent:      int   = 0
        self._total_filled:    int   = 0
        self._total_rejected:  int   = 0
        self._realized_pnl:    float = 0.0

        # Per-symbol breakdown
        self._symbol_metrics: dict[str, SymbolMetrics] = {}

        # Latency tracking
        self._ack_latencies:  list[float] = []
        self._fill_latencies: list[float] = []

        # All fill records (for CSV)
        self._fill_records: list[ExecutionRecord] = []

        # Dashboard task
        self._task: Optional[asyncio.Task] = None

        # CSV writer (initialised in start())
        self._csv_file   = None
        self._csv_writer = None

        logger.info("PerformanceTracker initialised")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the dashboard refresh background task and open CSV log."""
        if self._config.save_fills_csv:
            self._open_csv()
        self._task = asyncio.create_task(
            self._dashboard_loop(),
            name="hft_dashboard",
        )
        logger.info("PerformanceTracker started")

    async def stop(self) -> None:
        """Stop dashboard and flush CSV."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        if self._csv_file:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None

        self.print_summary()
        logger.info("PerformanceTracker stopped")

    # ------------------------------------------------------------------
    # Public – data ingestion
    # ------------------------------------------------------------------

    def record_fill(self, record: ExecutionRecord) -> None:
        """
        Integrate a completed fill into session statistics.

        Called by HFTExecutor via subscribe_fills() — should be fast.

        Args:
            record: ExecutionRecord with fill_time populated
        """
        self._total_sent += 1

        if record.rejected:
            self._total_rejected += 1
            return

        if record.is_filled:
            self._total_filled += 1
            self._fill_records.append(record)

            # Update symbol metrics
            sym = record.symbol
            if sym not in self._symbol_metrics:
                self._symbol_metrics[sym] = SymbolMetrics(symbol=sym)
            self._symbol_metrics[sym].update(record)

            # Aggregate P&L
            if record.order_info and record.order_info.pnl:
                self._realized_pnl += record.order_info.pnl

            # Latencies
            if record.ack_latency_ms is not None:
                self._ack_latencies.append(record.ack_latency_ms)
            if record.fill_latency_ms is not None:
                self._fill_latencies.append(record.fill_latency_ms)

            # Write to CSV
            if self._csv_writer:
                self._write_csv_row(record)

    def increment_signals(self, count: int = 1) -> None:
        """Increment the signal counter (called by the engine)."""
        self._total_signals += count

    # ------------------------------------------------------------------
    # Public – snapshot
    # ------------------------------------------------------------------

    def get_summary(self) -> SessionSummary:
        """Return a complete SessionSummary snapshot."""
        elapsed = datetime.now() - self._start_time
        elapsed_minutes = elapsed.total_seconds() / 60

        return SessionSummary(
            start_time=self._start_time,
            elapsed=elapsed,
            total_signals=self._total_signals,
            total_orders_sent=self._total_sent,
            total_orders_filled=self._total_filled,
            total_orders_rejected=self._total_rejected,
            fill_rate=self._total_filled / self._total_sent if self._total_sent > 0 else 0.0,
            realized_pnl=self._realized_pnl,
            avg_ack_latency_ms=(
                sum(self._ack_latencies) / len(self._ack_latencies)
                if self._ack_latencies else 0.0
            ),
            avg_fill_latency_ms=(
                sum(self._fill_latencies) / len(self._fill_latencies)
                if self._fill_latencies else 0.0
            ),
            orders_per_minute=(
                self._total_sent / elapsed_minutes if elapsed_minutes > 0 else 0.0
            ),
            symbol_metrics=dict(self._symbol_metrics),
        )

    def print_summary(self) -> None:
        """Print the full session summary to stdout."""
        print("\n" + str(self.get_summary()))

    def print_dashboard(self) -> None:
        """Print a compact live dashboard line."""
        elapsed = datetime.now() - self._start_time
        elapsed_str = str(elapsed).split(".")[0]
        pnl_sign = "+" if self._realized_pnl >= 0 else ""
        fill_rate = (
            self._total_filled / self._total_sent
            if self._total_sent > 0 else 0.0
        )
        avg_fill_ms = (
            sum(self._fill_latencies) / len(self._fill_latencies)
            if self._fill_latencies else 0.0
        )

        print(
            f"[{elapsed_str}] "
            f"Signals={self._total_signals} | "
            f"Sent={self._total_sent} | "
            f"Filled={self._total_filled} ({fill_rate:.0%}) | "
            f"Rejected={self._total_rejected} | "
            f"PnL={pnl_sign}${self._realized_pnl:.2f} | "
            f"AvgFill={avg_fill_ms:.0f}ms"
        )

    # ------------------------------------------------------------------
    # Private – dashboard loop
    # ------------------------------------------------------------------

    async def _dashboard_loop(self) -> None:
        """Periodically print the live dashboard."""
        while True:
            try:
                await asyncio.sleep(self._config.dashboard_interval)
                self.print_dashboard()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Dashboard error: {exc}")

    # ------------------------------------------------------------------
    # Private – CSV persistence
    # ------------------------------------------------------------------

    _CSV_FIELDS = [
        "timestamp", "symbol", "action", "quantity",
        "limit_price", "fill_price", "fill_latency_ms",
        "ack_latency_ms", "realized_pnl", "order_id",
    ]

    def _open_csv(self) -> None:
        """Create the fills CSV file and write the header row."""
        log_dir = Path(self._config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = log_dir / f"fills_{date_str}.csv"

        self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._CSV_FIELDS)
        self._csv_writer.writeheader()

        logger.info(f"Fill log opened: {csv_path}")

    def _write_csv_row(self, record: ExecutionRecord) -> None:
        """Append a single fill record to the CSV."""
        try:
            self._csv_writer.writerow({
                "timestamp":       datetime.now().isoformat(),
                "symbol":          record.symbol,
                "action":          record.action,
                "quantity":        record.quantity,
                "limit_price":     record.limit_price,
                "fill_price":      record.fill_price,
                "fill_latency_ms": f"{record.fill_latency_ms:.2f}" if record.fill_latency_ms else "",
                "ack_latency_ms":  f"{record.ack_latency_ms:.2f}"  if record.ack_latency_ms  else "",
                "realized_pnl":    record.order_info.pnl if record.order_info else "",
                "order_id":        record.order_info.order_id if record.order_info else "",
            })
            self._csv_file.flush()   # Flush on every row so data is not lost on crash
        except Exception as exc:
            logger.error(f"CSV write error: {exc}")

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PerformanceTracker("
            f"filled={self._total_filled}, "
            f"pnl=${self._realized_pnl:+.2f})"
        )

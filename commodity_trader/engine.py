"""
Commodity Trading Engine — Professional HFT Redesign.

Key improvements over v1:
  - Fill-price-based PnL tracking (not mid-price at submit time).
  - Trailing stop that follows the peak unrealised P&L.
  - Single exit guard (_exit_in_flight) prevents concurrent exit tasks.
  - Limit-then-market order fallback — tries to get price improvement first.
  - Session circuit breaker: halts entries after max_daily_loss_usd.
  - Risk manager receives actual prices for correct notional checks.
  - Dashboard printed every 30s via asyncio background task.
"""
from __future__ import annotations

import asyncio
import logging
import signal as _signal
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

from ibkr.config import IBKRConfig
from ibkr.connection import IBKRConnection
from ibkr.logging_config import setup_logging
from ibkr.market_data import MarketDataManager, Quote
from ibkr.positions import PositionManager
from ibkr.risk import RiskManager
from ibkr.trading import OrderManager, OrderAction, OrderStatus

from commodity_trader.config import CommodityTraderConfig
from commodity_trader.strategy import (
    CommoditySignalDetector, Signal, SignalType, PositionState
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Basket fill record
# ---------------------------------------------------------------------------

@dataclass
class BasketLeg:
    """Tracks the fill price of one order leg."""
    symbol:     str
    action:     str           # "BUY" or "SELL"
    quantity:   int
    fill_price: float = 0.0
    filled:     bool  = False


@dataclass
class BasketRecord:
    """Aggregates fill prices across all three legs of one basket trade."""
    legs:       Dict[str, BasketLeg] = field(default_factory=dict)
    entry_time: float                = field(default_factory=time.monotonic)

    @property
    def all_filled(self) -> bool:
        return all(leg.filled for leg in self.legs.values())

    def net_cost(self) -> float:
        """Net cash outflow. Positive = net long (bought more than sold)."""
        total = 0.0
        for leg in self.legs.values():
            sign = 1 if leg.action == "BUY" else -1
            total += sign * leg.fill_price * leg.quantity
        return total


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class CommodityEngine:
    """
    Top-level orchestrator for the commodity HFT bot.

    Startup sequence:
        1. Connect to IBKR.
        2. Subscribe AGQ, UGL, UCO market data.
        3. Wire quote callback → detector → signal → execution.

    Risk controls:
        - Per-order risk check with real price (not dummy 1.0).
        - Trailing stop tracks peak unrealised P&L; triggers market exit.
        - Hard stop loss below entry.
        - Session circuit breaker halts entries after max daily loss.
    """

    def __init__(self, ibkr_cfg: IBKRConfig, trader_cfg: CommodityTraderConfig) -> None:
        self.ibkr_cfg   = ibkr_cfg
        self.trader_cfg = trader_cfg

        # IBKR components (initialised in start())
        self.connection: Optional[IBKRConnection]   = None
        self.mkt_data:   Optional[MarketDataManager] = None
        self.order_mgr:  Optional[OrderManager]      = None
        self.pos_mgr:    Optional[PositionManager]   = None
        self.risk_mgr:   Optional[RiskManager]       = None

        # Strategy detector
        self.detector = CommoditySignalDetector(trader_cfg)

        # Session accounting
        self._session_realised_pnl:    float = 0.0
        self._session_trades:          int   = 0
        self._session_start:           datetime = datetime.now()

        # Position tracking
        self._entry_basket: Optional[BasketRecord] = None   # fills at entry
        self._exit_basket:  Optional[BasketRecord] = None   # fills at exit

        # Trailing stop tracking
        self._peak_unrealised_bps:  float = -float("inf")
        self._trailing_stop_active: bool  = False

        # Guard: only one exit routine may be in flight at a time
        self._exit_in_flight: bool = False

        # Engine state
        self._running    = False
        self._stop_event = asyncio.Event()

        # Background dashboard task
        self._dashboard_task: Optional[asyncio.Task] = None

        logger.info(f"CommodityEngine created — {trader_cfg}")

    # ------------------------------------------------------------------
    # Public – lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        setup_logging(log_dir="commodity_logs", log_level=logging.INFO)
        logger.info("=" * 60)
        logger.info("  COMMODITY HFT ENGINE STARTING")
        logger.info("=" * 60)

        self.connection = IBKRConnection(self.ibkr_cfg)
        await self.connection.connect()
        ib = self.connection.ib

        self.order_mgr = OrderManager(ib)
        self.pos_mgr   = PositionManager(ib)
        self.risk_mgr  = RiskManager(self.trader_cfg.to_risk_limits())

        self.mkt_data = MarketDataManager(ib)
        for sym in self.trader_cfg.symbols:
            await self.mkt_data.subscribe(sym)
            logger.info(f"  Subscribed: {sym}")

        self.mkt_data.subscribe_quotes(self._on_quote)
        self._running = True

        self._dashboard_task = asyncio.create_task(
            self._dashboard_loop(), name="commodity_dashboard"
        )

        warmup_secs = int(
            self.trader_cfg.indicators.bar_duration_secs *
            (self.trader_cfg.indicators.rsi_period + self.trader_cfg.indicators.ema_slow)
        )
        logger.info("=" * 60)
        logger.info("  ENGINE LIVE  --  trading AGQ / UGL / UCO")
        logger.info(f"  Indicator warm-up: ~{warmup_secs}s before first signal")
        logger.info("=" * 60)

    async def stop(self) -> None:
        if not self._running:
            return
        logger.info("Stopping Commodity Engine...")
        self._running = False
        self._stop_event.set()

        if self._dashboard_task:
            self._dashboard_task.cancel()

        if self.mkt_data:
            self.mkt_data.unsubscribe_all()

        if self.connection:
            await self.connection.disconnect()

        self._print_session_summary()
        logger.info("Engine stopped cleanly.")

    async def run(self, duration_seconds: Optional[float] = None) -> None:
        self._register_signal_handlers()
        await self.start()

        try:
            if duration_seconds is not None:
                await asyncio.wait_for(self._stop_event.wait(), timeout=duration_seconds)
            else:
                await self._stop_event.wait()
        except asyncio.TimeoutError:
            pass
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    # ------------------------------------------------------------------
    # Private – hot-path quote callback
    # ------------------------------------------------------------------

    def _on_quote(self, quote: Quote) -> None:
        if not self._running or not quote.is_tradeable:
            return

        signal = self.detector.on_quote(
            quote.symbol,
            quote.mid,
            volume=quote.volume,
            spread_bps=quote.spread_bps,
        )

        # ── Trailing stop / hard stop checks on every tradeable tick ───
        if self.detector.state in (PositionState.LONG_GOLD, PositionState.LONG_OIL):
            if not self._exit_in_flight:
                self._check_stops()

        # ── Entry / exit signals from detector ─────────────────────────
        if signal is not None and not self._exit_in_flight:
            asyncio.create_task(self._handle_signal(signal))

    # ------------------------------------------------------------------
    # Private – stop monitoring (trailing stop & hard stop)
    # ------------------------------------------------------------------

    def _check_stops(self) -> None:
        """
        Evaluate trailing stop and hard stop against current unrealised PnL.
        Called on every tick while in a position.

        PnL is computed from actual FILL prices (stored in _entry_basket),
        *not* from live mid prices.  This eliminates the fill-price bug in v1.
        """
        if self._entry_basket is None or not self._entry_basket.all_filled:
            return

        prices = self.detector.get_current_prices()
        if not all(s in prices for s in ("AGQ", "UGL", "UCO")):
            return

        pc  = self.trader_cfg.position
        qty = pc.order_size

        # Net current mark-to-market value (same sign convention as entry)
        state = self.detector.state
        if state == PositionState.LONG_GOLD:
            #  LONG AGQ, LONG UGL, SHORT UCO
            current_val = (prices["AGQ"] + prices["UGL"] - prices["UCO"]) * qty
        else:
            #  LONG UCO, SHORT AGQ, SHORT UGL
            current_val = (prices["UCO"] - prices["AGQ"] - prices["UGL"]) * qty

        # Entry cost from actual fills
        entry_val = self._entry_basket.net_cost()

        unrealised_pnl = current_val - entry_val

        # Exposure denominator for bps calculation
        abs_exposure = (prices["AGQ"] + prices["UGL"] + prices["UCO"]) * qty
        if abs_exposure <= 0:
            return
        unrealised_bps = (unrealised_pnl / abs_exposure) * 10_000

        # ── Trailing stop logic ──────────────────────────────────────────
        if unrealised_bps > self._peak_unrealised_bps:
            self._peak_unrealised_bps = unrealised_bps

        if unrealised_bps >= pc.trailing_stop_bps:
            self._trailing_stop_active = True

        if self._trailing_stop_active:
            trail_level = self._peak_unrealised_bps - pc.trail_distance_bps
            if unrealised_bps <= trail_level:
                logger.info(
                    f"TRAILING STOP — peak={self._peak_unrealised_bps:.1f} bps, "
                    f"current={unrealised_bps:.1f} bps, "
                    f"trail_level={trail_level:.1f} bps. Exiting."
                )
                self._trigger_exit("trailing_stop")
                return

        # ── Hard stop loss ───────────────────────────────────────────────
        if unrealised_bps <= -pc.stop_loss_bps:
            logger.warning(
                f"HARD STOP LOSS — {unrealised_bps:.1f} bps ≤ "
                f"-{pc.stop_loss_bps} bps. Exiting NOW."
            )
            self._trigger_exit("hard_stop")

    def _trigger_exit(self, reason: str) -> None:
        """Safely schedule exit once, guarded against concurrent calls."""
        if not self._exit_in_flight:
            self._exit_in_flight = True
            self.detector.set_state(PositionState.EXITING)
            asyncio.create_task(self._execute_exit(reason=reason))

    # ------------------------------------------------------------------
    # Private – signal handling (entry and indicator-driven exit)
    # ------------------------------------------------------------------

    async def _handle_signal(self, signal: Signal) -> None:
        pc = self.trader_cfg.position

        if signal.type == SignalType.ENTER_LONG_GOLD:
            # Session circuit breaker
            if -self._session_realised_pnl >= pc.max_daily_loss_usd:
                logger.warning(
                    f"CIRCUIT BREAKER — session loss "
                    f"${-self._session_realised_pnl:.2f} exceeds limit. "
                    f"No new entries."
                )
                return
            self.detector.set_state(PositionState.LONG_GOLD)
            await self._execute_entry(
                agq_action=OrderAction.BUY,
                ugl_action=OrderAction.BUY,
                uco_action=OrderAction.SELL,
                signal=signal,
            )

        elif signal.type == SignalType.ENTER_LONG_OIL:
            if -self._session_realised_pnl >= pc.max_daily_loss_usd:
                logger.warning("CIRCUIT BREAKER active — skipping entry.")
                return
            self.detector.set_state(PositionState.LONG_OIL)
            await self._execute_entry(
                agq_action=OrderAction.SELL,
                ugl_action=OrderAction.SELL,
                uco_action=OrderAction.BUY,
                signal=signal,
            )

        elif signal.type == SignalType.EXIT:
            if not self._exit_in_flight:
                self._exit_in_flight = True
                self.detector.set_state(PositionState.EXITING)
                await self._execute_exit(reason="indicator_exit")

    # ------------------------------------------------------------------
    # Private – order execution
    # ------------------------------------------------------------------

    async def _execute_entry(
        self,
        agq_action: OrderAction,
        ugl_action: OrderAction,
        uco_action: OrderAction,
        signal: Signal,
    ) -> None:
        """
        Place limit orders for all three legs simultaneously.
        If a leg is not filled within limit_timeout_secs, fall back to market.
        """
        pc     = self.trader_cfg.position
        qty    = pc.order_size
        prices = signal.prices

        # Reset trailing stop state for new position
        self._peak_unrealised_bps  = -float("inf")
        self._trailing_stop_active = False

        basket = BasketRecord()

        legs_cfg = [
            ("AGQ", agq_action, prices.get("AGQ", 0.0)),
            ("UGL", ugl_action, prices.get("UGL", 0.0)),
            ("UCO", uco_action, prices.get("UCO", 0.0)),
        ]

        for sym, act, mid in legs_cfg:
            if mid <= 0:
                logger.error(f"No valid price for {sym}; aborting entry.")
                self.detector.set_state(PositionState.FLAT)
                return
            chk = self.risk_mgr.check_order(sym, str(act.value), qty, mid)
            if not chk.approved:
                logger.error(f"Risk check DENIED {sym}: {chk.reason}. Aborting entry.")
                self.detector.set_state(PositionState.FLAT)
                return
            basket.legs[sym] = BasketLeg(sym, str(act.value), qty)

        logger.info(
            f"ENTRY — "
            f"AGQ {agq_action.value}, UGL {ugl_action.value}, UCO {uco_action.value} "
            f"× {qty} shares"
        )

        # Place limit orders concurrently, each with an offset from mid
        async def _place_limit_then_market(sym: str, act: OrderAction, mid_price: float) -> None:
            offset = pc.limit_offset
            lmt = round(mid_price - offset if act == OrderAction.BUY else mid_price + offset, 4)
            try:
                order_info = await self.order_mgr.place_limit_order(
                    sym, qty, act, limit_price=lmt
                )
                # Wait for fill
                deadline = time.monotonic() + pc.limit_timeout_secs
                while time.monotonic() < deadline:
                    await asyncio.sleep(0.25)
                    current = self.order_mgr.orders.get(order_info.order_id)
                    if current and current.status == OrderStatus.FILLED:
                        basket.legs[sym].fill_price = current.avg_fill_price
                        basket.legs[sym].filled = True
                        self.risk_mgr.record_fill(sym, str(act.value), qty, current.avg_fill_price)
                        logger.info(f"  Limit filled: {act.value} {qty} {sym} @ ${current.avg_fill_price:.4f}")
                        return

                # Timeout — cancel limit and fall back to market
                logger.warning(f"Limit timeout for {sym} — falling back to market order.")
                await self.order_mgr.cancel_order(order_info.order_id)

            except Exception as exc:
                logger.error(f"Limit order error for {sym}: {exc} — falling back to market.")

            # Market fallback
            try:
                mkt_info = await self.order_mgr.place_market_order(sym, qty, act)
                
                deadline = time.monotonic() + 15.0  # allow 15s for market order
                fill_px = 0.0
                while time.monotonic() < deadline:
                    await asyncio.sleep(0.25)
                    current = self.order_mgr.orders.get(mkt_info.order_id)
                    if current and current.status == OrderStatus.FILLED:
                        fill_px = current.avg_fill_price
                        break
                        
                if fill_px == 0.0:
                    current = self.order_mgr.orders.get(mkt_info.order_id)
                    if current and current.avg_fill_price > 0:
                        fill_px = current.avg_fill_price
                    else:
                        fill_px = self.detector.get_current_prices().get(sym, mid_price)
                    logger.warning(f"Market order for {sym} incomplete after 15s. Using ${fill_px:.4f} for PnL tracking.")

                basket.legs[sym].fill_price = fill_px
                basket.legs[sym].filled = True
                self.risk_mgr.record_fill(sym, str(act.value), qty, fill_px)
                logger.info(f"  Market filled: {act.value} {qty} {sym} @ ${fill_px:.4f}")
            except Exception as exc:
                logger.error(f"Market fallback failed for {sym}: {exc}")

        await asyncio.gather(*[
            _place_limit_then_market(sym, act, prices.get(sym, 0.0))
            for sym, act in [("AGQ", agq_action), ("UGL", ugl_action), ("UCO", uco_action)]
        ])

        self._entry_basket = basket
        self._session_trades += 1
        logger.info(f"Entry complete. Basket fully {'filled' if basket.all_filled else 'PARTIAL'}.")

    async def _execute_exit(self, reason: str = "signal") -> None:
        """
        Close all three legs with market orders simultaneously.
        Records realised P&L from fill prices.
        """
        pc  = self.trader_cfg.position
        qty = pc.order_size

        state = self.detector.state  # was EXITING by the time we get here
        # Determine what the position WAS before EXITING was set
        # (We infer from the entry basket action directions)
        if self._entry_basket is None:
            logger.warning("No entry basket recorded — cannot determine exit directions.")
            self._reset_position()
            return

        exit_actions: Dict[str, OrderAction] = {}
        for sym, leg in self._entry_basket.legs.items():
            exit_actions[sym] = OrderAction.SELL if leg.action == "BUY" else OrderAction.BUY

        logger.info(
            f"EXIT ({reason}) — "
            + ", ".join(f"{v.value} {sym}" for sym, v in exit_actions.items())
        )

        exit_basket = BasketRecord()
        for sym, act in exit_actions.items():
            exit_basket.legs[sym] = BasketLeg(sym, str(act.value), qty)

        async def _place_exit_market(sym: str, act: OrderAction) -> None:
            try:
                mkt_info = await self.order_mgr.place_market_order(sym, qty, act)
                
                deadline = time.monotonic() + 15.0
                fill_px = 0.0
                while time.monotonic() < deadline:
                    await asyncio.sleep(0.25)
                    current = self.order_mgr.orders.get(mkt_info.order_id)
                    if current and current.status == OrderStatus.FILLED:
                        fill_px = current.avg_fill_price
                        break
                        
                if fill_px <= 0.0:
                    current = self.order_mgr.orders.get(mkt_info.order_id)
                    if current and current.avg_fill_price > 0:
                        fill_px = current.avg_fill_price
                    else:
                        fill_px = self.detector.get_current_prices().get(sym, 0.0)
                    logger.warning(f"Exit order for {sym} incomplete after 15s. Using ${fill_px:.4f} for PnL.")

                exit_basket.legs[sym].fill_price = fill_px
                exit_basket.legs[sym].filled = True
                self.risk_mgr.record_fill(sym, str(act.value), qty, fill_px)
                logger.info(f"  Exit filled: {act.value} {qty} {sym} @ ${fill_px:.4f}")
            except Exception as exc:
                logger.error(f"Exit market order failed for {sym}: {exc}")

        await asyncio.gather(*[
            _place_exit_market(sym, act) for sym, act in exit_actions.items()
        ])

        # ── Realised P&L ─────────────────────────────────────────────────
        self._exit_basket = exit_basket
        pnl = self._compute_realised_pnl(self._entry_basket, exit_basket)
        self._session_realised_pnl += pnl
        self.risk_mgr.record_pnl(pnl)

        logger.info(
            f"Position closed. Realised P&L: ${pnl:+.2f} | "
            f"Session total: ${self._session_realised_pnl:+.2f}"
        )

        self._reset_position()

    @staticmethod
    def _compute_realised_pnl(entry: BasketRecord, exit_b: BasketRecord) -> float:
        """
        P&L = Σ(exit_fill × qty × sign_exit) - Σ(entry_fill × qty × sign_entry)
        where sign = +1 for BUY, -1 for SELL.
        """
        pnl = 0.0
        for sym in entry.legs:
            e_leg = entry.legs[sym]
            x_leg = exit_b.legs.get(sym)
            if x_leg is None or not e_leg.filled or not x_leg.filled:
                continue
            e_cost  = (1 if e_leg.action == "BUY" else -1) * e_leg.fill_price * e_leg.quantity
            x_proc  = (1 if x_leg.action == "BUY" else -1) * x_leg.fill_price * x_leg.quantity
            pnl    += -(e_cost + x_proc)   # net proceeds - net cost
        return pnl

    def _reset_position(self) -> None:
        self._entry_basket         = None
        self._exit_basket          = None
        self._peak_unrealised_bps  = -float("inf")
        self._trailing_stop_active = False
        self._exit_in_flight       = False
        self.detector.set_state(PositionState.FLAT)

    # ------------------------------------------------------------------
    # Private – dashboard
    # ------------------------------------------------------------------

    async def _dashboard_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(30)
                self._print_dashboard()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Dashboard error: {exc}")

    def _print_dashboard(self) -> None:
        elapsed = datetime.now() - self._session_start
        elapsed_str = str(elapsed).split(".")[0]
        rsi_snap = self.detector.get_rsi_snapshot()
        rsi_str = " | ".join(
            f"{s} RSI={v:.1f}" for s, v in rsi_snap.items() if v is not None
        ) or "warming up…"
        logger.info(
            f"[Dashboard {elapsed_str}] "
            f"State={self.detector.state.value} | "
            f"Trades={self._session_trades} | "
            f"Session P&L=${self._session_realised_pnl:+.2f} | "
            f"{rsi_str}"
        )

    def _print_session_summary(self) -> None:
        elapsed = datetime.now() - self._session_start
        risk = self.risk_mgr.summary() if self.risk_mgr else {}
        print("\n" + "═" * 56)
        print("  COMMODITY ENGINE — SESSION SUMMARY")
        print("═" * 56)
        print(f"  Runtime        : {str(elapsed).split('.')[0]}")
        print(f"  Trades         : {self._session_trades}")
        print(f"  Realised P&L   : ${self._session_realised_pnl:+.2f}")
        print(f"  Orders sent    : {risk.get('orders_sent', 'n/a')}")
        print(f"  Orders rejected: {risk.get('orders_rejected', 'n/a')}")
        print("═" * 56)

    # ------------------------------------------------------------------
    # Private – OS signal handlers
    # ------------------------------------------------------------------

    def _register_signal_handlers(self) -> None:
        loop = asyncio.get_event_loop()
        def _shutdown(sig_name: str) -> None:
            logger.warning(f"Received {sig_name} — shutting down.")
            self._stop_event.set()

        for sig in (_signal.SIGINT, _signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda s=sig.name: _shutdown(s))
            except (NotImplementedError, OSError):
                pass

    def __repr__(self) -> str:
        status = "RUNNING" if self._running else "STOPPED"
        return (
            f"CommodityEngine("
            f"status={status}, "
            f"state={self.detector.state.value}, "
            f"pnl=${self._session_realised_pnl:+.2f})"
        )

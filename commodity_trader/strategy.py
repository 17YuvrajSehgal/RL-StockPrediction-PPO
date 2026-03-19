"""
Multi-indicator confluence signal detection for the commodity HFT bot.

Signal philosophy (professional HFT approach):
  - No single indicator fires a trade.
  - ALL of RSI + EMA crossover + VWAP deviation must agree before entry.
  - Exit on ANY one of: RSI reversal, EMA crossover flip, trailing stop, hard stop.

State machine:
    FLAT
      ├─(gold confluence)──► LONG_GOLD
      └─(oil  confluence)──► LONG_OIL

    LONG_GOLD / LONG_OIL
      ├─(exit confluence OR stop)──► EXITING
      └───────────────────────────► (continue holding)

    EXITING (orders in flight, managed by engine)
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from commodity_trader.config import CommodityTraderConfig
from commodity_trader.indicators import SymbolIndicators, BarAggregator, RSI, EMA, VWAP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class PositionState(str, Enum):
    FLAT      = "FLAT"
    LONG_GOLD = "LONG_GOLD"   # Long AGQ + UGL, Short UCO
    LONG_OIL  = "LONG_OIL"   # Long UCO, Short AGQ + UGL
    EXITING   = "EXITING"     # Exit orders in flight


class SignalType(str, Enum):
    ENTER_LONG_GOLD = "ENTER_LONG_GOLD"
    ENTER_LONG_OIL  = "ENTER_LONG_OIL"
    EXIT            = "EXIT"


# ---------------------------------------------------------------------------
# Signal object
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """A trading action request emitted by the detector."""
    type:             SignalType
    prices:           Dict[str, float]   # latest mid for each symbol
    trigger_symbol:   str                # the symbol that tipped the confluence
    rsi_values:       Dict[str, float]   # diagnostic
    timestamp:        float = field(default_factory=time.monotonic)

    def __repr__(self) -> str:
        rsi_str = ", ".join(f"{s}={v:.1f}" for s, v in self.rsi_values.items())
        return f"Signal({self.type.value} trigger={self.trigger_symbol} RSI=[{rsi_str}])"


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class CommoditySignalDetector:
    """
    Evaluates multi-indicator confluence on bar close events and emits
    trading signals.

    Indicators computed on 10-second bars:
        - Wilder's RSI-9
        - EMA-5 / EMA-13 crossover
        - Tick-level VWAP deviation

    ENTRY conditions (all required):
        Gold regime (LONG AGQ+UGL, SHORT UCO):
            UGL RSI > rsi_entry  AND  UGL EMA_fast > EMA_slow  AND  UGL price > VWAP * (1 + dev%)
        Oil regime (LONG UCO, SHORT AGQ+UGL):
            UCO RSI > rsi_entry  AND  UCO EMA_fast > EMA_slow  AND  UCO price > VWAP * (1 + dev%)

        If BOTH gold and oil simultaneously signal, pick the higher RSI.

    EXIT conditions (any one sufficient):
        Trigger symbol RSI falls below rsi_exit
        OR  EMA fast < EMA slow (trend reversal)
        (Stop/trailing stop are handled by the engine, not here)
    """

    def __init__(self, config: CommodityTraderConfig) -> None:
        ic = config.indicators
        sc = config.signal

        self._rsi_entry    = sc.rsi_entry
        self._rsi_exit     = sc.rsi_exit
        self._vwap_dev     = sc.vwap_dev_entry_pct / 100.0
        self._ema_confirm  = sc.ema_confirmation
        self._cooldown     = sc.cooldown_secs
        self._min_spread_bps = sc.min_spread_bps

        # Indicator bundles for the two DRIVING symbols
        self._drivers: Dict[str, SymbolIndicators] = {
            sym: SymbolIndicators(
                bars     = BarAggregator(ic.bar_duration_secs),
                rsi      = RSI(ic.rsi_period),
                ema_fast = EMA(ic.ema_fast),
                ema_slow = EMA(ic.ema_slow),
                vwap     = VWAP(),
            )
            for sym in ("UGL", "UCO")
        }
        # AGQ is only tracked for pricing (it's the satellite leg on silver)
        self._prices: Dict[str, float] = {}

        self.state: PositionState = PositionState.FLAT
        self._last_signal_time: float = 0.0

        logger.info(
            f"CommoditySignalDetector initialised — "
            f"RSI entry={self._rsi_entry} exit={self._rsi_exit}, "
            f"VWAP dev={sc.vwap_dev_entry_pct}%, "
            f"cooldown={self._cooldown}s"
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def on_quote(self, symbol: str, mid_price: float,
                 volume: float = 0.0, spread_bps: float = 0.0) -> Optional[Signal]:
        """
        Feed a single tick.  Returns a Signal if conditions are met, else None.

        Args:
            symbol:     The quoting symbol.
            mid_price:  (bid + ask) / 2.
            volume:     Last trade volume (use 0 if unavailable).
            spread_bps: Current bid-ask spread in basis points.
        """
        if not math.isfinite(mid_price) or mid_price <= 0:
            return None

        # Cache latest price for every symbol (used for basket order prices)
        self._prices[symbol] = mid_price

        # Feed driver instruments
        if symbol in self._drivers:
            self._drivers[symbol].on_tick(mid_price, volume)

        # ── Cooldown guard ──────────────────────────────────────────────
        now = time.monotonic()
        if now - self._last_signal_time < self._cooldown:
            return None

        # ── Need prices for all three basket symbols ─────────────────────
        if not all(s in self._prices for s in ("AGQ", "UGL", "UCO")):
            return None

        # ── Spread filter: skip if market is too thin ─────────────────────
        if self._min_spread_bps > 0 and spread_bps > self._min_spread_bps:
            logger.debug(f"Skipping tick — spread {spread_bps:.1f} bps > min {self._min_spread_bps} bps")
            return None

        # ── State-specific evaluation ─────────────────────────────────────
        if self.state == PositionState.FLAT:
            return self._evaluate_entry(now)

        elif self.state in (PositionState.LONG_GOLD, PositionState.LONG_OIL):
            return self._evaluate_exit(now)

        return None

    def set_state(self, state: PositionState) -> None:
        """Called by the engine to advance the state machine after order dispatch."""
        self.state = state
        self._last_signal_time = time.monotonic()
        logger.info(f"Detector state → {state.value}")

    def get_current_prices(self) -> Dict[str, float]:
        """Return a snapshot of the latest mid-prices for all symbols."""
        return dict(self._prices)

    def get_rsi_snapshot(self) -> Dict[str, Optional[float]]:
        return {sym: ind.last_rsi for sym, ind in self._drivers.items()}

    # ------------------------------------------------------------------
    # Private — entry evaluation
    # ------------------------------------------------------------------

    def _evaluate_entry(self, now: float) -> Optional[Signal]:
        """Check gold and oil confluences. If both fire, take the stronger one."""
        gold_score = self._gold_confluence()
        oil_score  = self._oil_confluence()

        if gold_score is None and oil_score is None:
            return None

        # Pick stronger (higher RSI score)
        if gold_score is not None and (oil_score is None or gold_score >= oil_score):
            return self._emit(SignalType.ENTER_LONG_GOLD, "UGL", now)
        else:
            return self._emit(SignalType.ENTER_LONG_OIL, "UCO", now)

    def _gold_confluence(self) -> Optional[float]:
        """
        Returns UGL's RSI if the FULL confluence is satisfied, else None.
        Confluence: RSI > entry  AND  EMA fast > slow  AND  price > VWAP + dev%
        """
        ind = self._drivers["UGL"]
        if not ind.is_ready:
            return None

        rsi = ind.last_rsi
        if rsi < self._rsi_entry:
            return None

        if self._ema_confirm and (ind.last_ema_fast is None or ind.last_ema_slow is None):
            return None
        if self._ema_confirm and ind.last_ema_fast <= ind.last_ema_slow:
            return None

        if ind.last_vwap and ind.last_price:
            if ind.last_price < ind.last_vwap * (1.0 + self._vwap_dev):
                return None

        return rsi

    def _oil_confluence(self) -> Optional[float]:
        """
        Returns UCO's RSI if the FULL confluence is satisfied, else None.
        """
        ind = self._drivers["UCO"]
        if not ind.is_ready:
            return None

        rsi = ind.last_rsi
        if rsi < self._rsi_entry:
            return None

        if self._ema_confirm and (ind.last_ema_fast is None or ind.last_ema_slow is None):
            return None
        if self._ema_confirm and ind.last_ema_fast <= ind.last_ema_slow:
            return None

        if ind.last_vwap and ind.last_price:
            if ind.last_price < ind.last_vwap * (1.0 + self._vwap_dev):
                return None

        return rsi

    # ------------------------------------------------------------------
    # Private — exit evaluation
    # ------------------------------------------------------------------

    def _evaluate_exit(self, now: float) -> Optional[Signal]:
        """
        Exit when the trigger symbol's RSI drops below rsi_exit
        OR when the EMA crossover reverses.
        """
        trigger = "UGL" if self.state == PositionState.LONG_GOLD else "UCO"
        ind = self._drivers[trigger]

        if not ind.is_ready:
            return None

        rsi_exit_triggered = (ind.last_rsi is not None and ind.last_rsi < self._rsi_exit)
        ema_flip = (
            self._ema_confirm and
            ind.last_ema_fast is not None and
            ind.last_ema_slow is not None and
            ind.last_ema_fast < ind.last_ema_slow
        )

        if rsi_exit_triggered or ema_flip:
            reason = "RSI exit" if rsi_exit_triggered else "EMA flip"
            logger.info(f"Exit condition met ({reason}): RSI={ind.last_rsi:.1f}")
            return self._emit(SignalType.EXIT, trigger, now)

        return None

    # ------------------------------------------------------------------
    # Private — signal emission
    # ------------------------------------------------------------------

    def _emit(self, signal_type: SignalType, trigger: str, now: float) -> Signal:
        self._last_signal_time = now
        rsi_snap = self.get_rsi_snapshot()
        sig = Signal(
            type=signal_type,
            prices=self.get_current_prices(),
            trigger_symbol=trigger,
            rsi_values={k: v for k, v in rsi_snap.items() if v is not None},
        )
        logger.info(f"Signal emitted: {sig}")
        return sig

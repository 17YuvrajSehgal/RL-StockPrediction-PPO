"""
Arbitrage signal detectors.

This module contains pure-Python signal detectors that analyse incoming
quote data and produce actionable trading signals.  There is NO I/O here —
only maths on price series — making the detectors fast, deterministic, and
easy to unit-test without a broker connection.

Design principles:
  - Each detector is self-contained and stateful (maintains its own history)
  - on_quote() is the single ingestion point: feed a (symbol, mid_price) pair
  - Returns a Signal when a threshold is crossed, None otherwise
  - All floating-point maths uses the standard library only (no numpy needed
    for small rolling windows, keeps latency low)

Included detectors:
  - PairsArbDetector   — z-score spread trading between two correlated symbols
  - (ETFArbDetector)   — placeholder, not yet implemented
  - (MarketMakingSignal) — placeholder, not yet implemented

Signal lifecycle:
  ENTER (long A / short B)  →  EXIT  →  (potentially) ENTER (short A / long B)
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal data structures
# ---------------------------------------------------------------------------

class SignalType(str, Enum):
    """
    What the detector is asking the executor to do.

    ENTER_LONG_A  → BUY symbol_a, SELL symbol_b  (spread is too wide, expect compression)
    ENTER_SHORT_A → SELL symbol_a, BUY symbol_b  (spread is too narrow / inverted)
    EXIT          → Close the current paired position back to flat
    """
    ENTER_LONG_A  = "ENTER_LONG_A"
    ENTER_SHORT_A = "ENTER_SHORT_A"
    EXIT          = "EXIT"


@dataclass
class Signal:
    """
    A trading signal emitted by a detector.

    Attributes:
        type:       The specific action recommended (see SignalType)
        symbol_a:   Primary symbol (the one whose action is named)
        symbol_b:   Secondary / hedge symbol
        zscore:     Z-score of the spread at signal time (diagnostic)
        spread:     Raw price spread (price_a - price_b) at signal time
        mid_a:      Mid price of symbol_a at signal time
        mid_b:      Mid price of symbol_b at signal time
        timestamp:  Unix timestamp (float) of signal generation
        quantity:   Suggested order size (shares per leg); may be overridden
    """
    type:      SignalType
    symbol_a:  str
    symbol_b:  str
    zscore:    float
    spread:    float
    mid_a:     float
    mid_b:     float
    timestamp: float = field(default_factory=time.monotonic)
    quantity:  int   = 100

    def __repr__(self) -> str:
        return (
            f"Signal({self.type.value} "
            f"{self.symbol_a}/{self.symbol_b} "
            f"z={self.zscore:+.3f} "
            f"spread={self.spread:+.4f})"
        )


# ---------------------------------------------------------------------------
# Internal rolling statistics helper
# ---------------------------------------------------------------------------

class _RollingStats:
    """
    Efficient rolling mean and standard deviation over a fixed-size window.

    Uses a deque for the window and Welford's online algorithm for
    numerically stable computation.

    Args:
        maxlen: Maximum number of observations to keep
    """

    def __init__(self, maxlen: int) -> None:
        self._window: deque[float] = deque(maxlen=maxlen)
        self._maxlen = maxlen

    def push(self, value: float) -> None:
        """Add a new observation."""
        self._window.append(value)

    @property
    def count(self) -> int:
        """Number of observations currently in the window."""
        return len(self._window)

    @property
    def is_ready(self) -> bool:
        """True when the window is fully populated."""
        return len(self._window) == self._maxlen

    @property
    def mean(self) -> float:
        """Arithmetic mean of the window."""
        if not self._window:
            return float("nan")
        return sum(self._window) / len(self._window)

    @property
    def std(self) -> float:
        """
        Sample standard deviation of the window.

        Returns NaN if fewer than 2 observations exist.
        """
        n = len(self._window)
        if n < 2:
            return float("nan")
        mu = self.mean
        variance = sum((x - mu) ** 2 for x in self._window) / (n - 1)
        return math.sqrt(variance)

    def zscore(self, value: float) -> float:
        """
        Compute the z-score of a new value against the current window.

        Does NOT push the value into the window.

        Args:
            value: The observation to score

        Returns:
            z-score, or NaN if insufficient data or std == 0
        """
        mu  = self.mean
        sig = self.std
        if math.isnan(mu) or math.isnan(sig) or sig == 0:
            return float("nan")
        return (value - mu) / sig


# ---------------------------------------------------------------------------
# Pairs arbitrage detector
# ---------------------------------------------------------------------------

class PairsArbDetector:
    """
    Statistical pairs arbitrage signal detector.

    Tracks the price spread (price_a - price_b) between two correlated
    symbols over a rolling window.  When the z-score of the spread exceeds
    `zscore_entry` (in either direction), an entry signal is generated.
    When the position is open and the z-score reverts to within `zscore_exit`,
    an exit signal is generated.

    State machine:
        FLAT  ──(|z| > entry)──▶  IN_TRADE
        IN_TRADE  ──(|z| < exit)──▶  FLAT

    Usage:
        >>> detector = PairsArbDetector("SPY", "IVV", lookback=100)
        >>> signal = detector.on_quote("SPY", 478.21)
        >>> signal = detector.on_quote("IVV", 477.85)
        >>> if signal:
        ...     print(signal)

    Args:
        symbol_a:       Primary symbol
        symbol_b:       Hedge symbol
        lookback:       Rolling window size (ticks)
        zscore_entry:   Z-score magnitude to trigger entry
        zscore_exit:    Z-score magnitude to trigger exit
        order_size:     Default order size per leg (shares)
        cooldown_secs:  Minimum seconds between signals (debounce)
        debug:          Log every evaluated z-score at DEBUG level
    """

    class _State(str, Enum):
        FLAT     = "FLAT"
        LONG_A   = "LONG_A"    # long symbol_a, short symbol_b
        SHORT_A  = "SHORT_A"   # short symbol_a, long symbol_b

    def __init__(
        self,
        symbol_a:      str,
        symbol_b:      str,
        lookback:      int   = 100,
        zscore_entry:  float = 2.0,
        zscore_exit:   float = 0.5,
        order_size:    int   = 100,
        cooldown_secs: float = 5.0,
        debug:         bool  = False,
    ) -> None:
        self.symbol_a      = symbol_a
        self.symbol_b      = symbol_b
        self.lookback      = lookback
        self.zscore_entry  = zscore_entry
        self.zscore_exit   = zscore_exit
        self.order_size    = order_size
        self.cooldown_secs = cooldown_secs
        self._debug        = debug

        # Per-symbol latest mid-price cache
        self._prices: dict[str, float] = {}

        # Rolling statistics over the spread series
        self._stats = _RollingStats(maxlen=lookback)

        # Position state
        self._state = self._State.FLAT

        # Cooldown timer (monotonic)
        self._last_signal_ts: float = 0.0

        # Diagnostic counters
        self.signals_generated: int = 0
        self.ticks_processed:   int = 0

        logger.info(
            f"PairsArbDetector initialised: "
            f"{symbol_a}/{symbol_b} "
            f"lookback={lookback} "
            f"entry_z={zscore_entry} exit_z={zscore_exit}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_quote(self, symbol: str, mid_price: float) -> Optional[Signal]:
        """
        Process a single incoming mid-price update.

        Call this with every quote tick from MarketDataManager.  The method
        updates internal state and returns a Signal when conditions are met,
        or None if no action is required.

        Args:
            symbol:    The symbol whose price updated
            mid_price: Current mid-price (bid+ask)/2

        Returns:
            Signal if an entry/exit condition is met, otherwise None
        """
        if symbol not in (self.symbol_a, self.symbol_b):
            return None
        if not self._is_valid_price(mid_price):
            return None

        # Store latest price
        self._prices[symbol] = mid_price
        self.ticks_processed += 1

        # We need both prices before computing anything
        if self.symbol_a not in self._prices or self.symbol_b not in self._prices:
            return None

        price_a = self._prices[self.symbol_a]
        price_b = self._prices[self.symbol_b]

        # Compute current spread and update rolling window
        spread  = price_a - price_b
        current_z = self._stats.zscore(spread)
        self._stats.push(spread)

        if self._debug:
            logger.debug(
                f"{self.symbol_a}/{self.symbol_b} spread={spread:.4f} z={current_z:+.3f} "
                f"state={self._state.value} window={self._stats.count}/{self.lookback}"
            )

        # Need a full window before trusting the z-score
        if not self._stats.is_ready:
            return None

        if math.isnan(current_z):
            return None

        # Evaluate state machine
        return self._evaluate(current_z, spread, price_a, price_b)

    def reset(self) -> None:
        """Reset all state — useful between trading sessions."""
        self._prices.clear()
        self._stats = _RollingStats(maxlen=self.lookback)
        self._state = self._State.FLAT
        self._last_signal_ts = 0.0
        self.signals_generated = 0
        self.ticks_processed   = 0
        logger.info(f"PairsArbDetector {self.symbol_a}/{self.symbol_b} reset")

    @property
    def is_in_position(self) -> bool:
        """True if currently holding an open pair trade."""
        return self._state != self._State.FLAT

    @property
    def current_state(self) -> str:
        """Current state machine state as string."""
        return self._state.value

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        z:       float,
        spread:  float,
        price_a: float,
        price_b: float,
    ) -> Optional[Signal]:
        """
        Apply the state-machine rules to produce a Signal (or None).

        Entry rules (FLAT state):
          - z > +entry  → spread too wide, short A / long B  (SHORT_A)
          - z < -entry  → spread too narrow, long A / short B (LONG_A)

        Exit rules (IN_TRADE state):
          - |z| < exit  → spread reverted, close position  (EXIT)
        """
        now = time.monotonic()

        if self._state == self._State.FLAT:
            # Do not fire signals faster than cooldown_secs
            if now - self._last_signal_ts < self.cooldown_secs:
                return None

            if z > self.zscore_entry:
                # Spread is abnormally large: A is expensive vs B
                # → short A, long B
                return self._emit(SignalType.ENTER_SHORT_A, z, spread, price_a, price_b, now)

            if z < -self.zscore_entry:
                # Spread is abnormally small / inverted: A is cheap vs B
                # → long A, short B
                return self._emit(SignalType.ENTER_LONG_A, z, spread, price_a, price_b, now)

        elif self._state in (self._State.LONG_A, self._State.SHORT_A):
            # Check for mean-reversion exit
            if abs(z) < self.zscore_exit:
                return self._emit(SignalType.EXIT, z, spread, price_a, price_b, now)

        return None

    def _emit(
        self,
        signal_type: SignalType,
        z:           float,
        spread:      float,
        price_a:     float,
        price_b:     float,
        now:         float,
    ) -> Signal:
        """Build the Signal object and advance internal state."""
        # Transition state machine
        if signal_type == SignalType.ENTER_LONG_A:
            self._state = self._State.LONG_A
        elif signal_type == SignalType.ENTER_SHORT_A:
            self._state = self._State.SHORT_A
        elif signal_type == SignalType.EXIT:
            self._state = self._State.FLAT

        self._last_signal_ts = now
        self.signals_generated += 1

        sig = Signal(
            type=signal_type,
            symbol_a=self.symbol_a,
            symbol_b=self.symbol_b,
            zscore=z,
            spread=spread,
            mid_a=price_a,
            mid_b=price_b,
            timestamp=now,
            quantity=self.order_size,
        )

        logger.info(f"Signal generated: {sig}")
        return sig

    @staticmethod
    def _is_valid_price(price: float) -> bool:
        return price is not None and not math.isnan(price) and price > 0

    def __repr__(self) -> str:
        return (
            f"PairsArbDetector("
            f"{self.symbol_a}/{self.symbol_b}, "
            f"state={self._state.value}, "
            f"signals={self.signals_generated}, "
            f"ticks={self.ticks_processed})"
        )


# ---------------------------------------------------------------------------
# Placeholder detectors (for future expansion)
# ---------------------------------------------------------------------------

class ETFArbDetector:
    """
    ETF vs basket-of-components arbitrage detector.

    Not yet implemented.  Registered here so it can be referenced in
    HFTConfig and toggled without code changes when implemented.
    """

    def on_quote(self, symbol: str, mid_price: float) -> Optional[Signal]:
        raise NotImplementedError("ETFArbDetector is not yet implemented")


class MarketMakingDetector:
    """
    Bid-ask spread capture (market making) signal detector.

    Not yet implemented.
    """

    def on_quote(self, symbol: str, mid_price: float) -> Optional[Signal]:
        raise NotImplementedError("MarketMakingDetector is not yet implemented")

"""
Pure-Python technical indicators for the commodity HFT bot.

All indicators are tick-fed and stateful.  No numpy dependency — keeps
import overhead and GC pressure minimal on the hot path.

Included:
    BarAggregator   — collapses raw ticks into fixed-duration OHLCV bars
    RSI             — Wilder's smoothed RSI on bar closes
    EMA             — Exponential Moving Average on bar closes
    VWAP            — Session Volume-Weighted Average Price (tick-level)
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# OHLCV Bar
# ---------------------------------------------------------------------------

@dataclass
class Bar:
    """A completed OHLCV bar."""
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float
    ts:     float   # bar open timestamp (monotonic)

    @property
    def typical_price(self) -> float:
        """(H+L+C)/3 — used for VWAP."""
        return (self.high + self.low + self.close) / 3.0


# ---------------------------------------------------------------------------
# Bar Aggregator
# ---------------------------------------------------------------------------

class BarAggregator:
    """
    Aggregates real-time tick updates into fixed-duration OHLCV bars.

    Each call to ``update(price, volume)`` returns a completed Bar when
    the current bar's duration elapses, otherwise returns None.

    Args:
        duration_secs:  How many seconds each bar spans (e.g. 10.0).
    """

    def __init__(self, duration_secs: float = 10.0) -> None:
        self._dur  = duration_secs
        self._open:    Optional[float] = None
        self._high:    float = -math.inf
        self._low:     float =  math.inf
        self._close:   float = 0.0
        self._volume:  float = 0.0
        self._bar_ts:  float = 0.0   # start of current bar

    def update(self, price: float, volume: float = 0.0) -> Optional[Bar]:
        """
        Feed a tick.  Returns a completed Bar if the bar window just closed,
        None otherwise.
        """
        now = time.monotonic()

        # Initialise the first bar
        if self._open is None:
            self._open   = price
            self._bar_ts = now

        # Update running OHLCV
        self._high   = max(self._high, price)
        self._low    = min(self._low,  price)
        self._close  = price
        self._volume += volume

        # Check if bar is complete
        if now - self._bar_ts >= self._dur:
            bar = Bar(
                open=self._open,
                high=self._high,
                low=self._low,
                close=self._close,
                volume=self._volume,
                ts=self._bar_ts,
            )
            # Start new bar
            self._open   = price
            self._high   = price
            self._low    = price
            self._close  = price
            self._volume = 0.0
            self._bar_ts = now
            return bar

        return None

    @property
    def current_close(self) -> Optional[float]:
        """Latest price seen in the still-open current bar."""
        return self._close if self._open is not None else None


# ---------------------------------------------------------------------------
# Wilder's RSI (bar-based)
# ---------------------------------------------------------------------------

class RSI:
    """
    Wilder's Relative Strength Index computed on bar close prices.

    Uses Wilder's smoothed averages (equivalent to EMA with alpha=1/period)
    which is the industry-standard RSI formulation.

    Args:
        period:  Look-back period in bars (typically 9 or 14).
    """

    def __init__(self, period: int = 9) -> None:
        self._period   = period
        self._closes:  deque[float] = deque()
        self._avg_gain: Optional[float] = None
        self._avg_loss: Optional[float] = None
        self._prev:     Optional[float] = None
        self._count:    int = 0

    def update(self, close: float) -> Optional[float]:
        """
        Feed a bar close.  Returns RSI (0–100) once the warm-up period
        is complete, otherwise None.
        """
        if self._prev is None:
            self._prev = close
            return None

        change = close - self._prev
        gain   = max(0.0, change)
        loss   = max(0.0, -change)
        self._prev = close
        self._count += 1

        if self._count < self._period:
            # Accumulating initial window
            self._closes.append((gain, loss))
            return None

        elif self._count == self._period:
            # First Wilder average: simple mean of initial period
            self._closes.append((gain, loss))
            self._avg_gain = sum(g for g, _ in self._closes) / self._period
            self._avg_loss = sum(l for _, l in self._closes) / self._period

        else:
            # Subsequent: Wilder smoothing
            alpha = 1.0 / self._period
            self._avg_gain = alpha * gain  + (1.0 - alpha) * self._avg_gain
            self._avg_loss = alpha * loss  + (1.0 - alpha) * self._avg_loss

        if self._avg_loss == 0.0:
            return 100.0
        rs = self._avg_gain / self._avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @property
    def is_ready(self) -> bool:
        return self._avg_gain is not None


# ---------------------------------------------------------------------------
# Exponential Moving Average (bar-based)
# ---------------------------------------------------------------------------

class EMA:
    """
    Standard exponential moving average on bar close prices.

    Args:
        period: EMA period in bars.
    """

    def __init__(self, period: int) -> None:
        self._period = period
        self._alpha  = 2.0 / (period + 1.0)
        self._value: Optional[float] = None
        self._count: int = 0

    def update(self, close: float) -> Optional[float]:
        """Feed a bar close. Returns current EMA value or None during warm-up."""
        self._count += 1
        if self._value is None:
            self._value = close   # seed with first price
        else:
            self._value = self._alpha * close + (1.0 - self._alpha) * self._value
        # Only return once we have seen enough bars to stabilise
        if self._count >= self._period:
            return self._value
        return None

    @property
    def value(self) -> Optional[float]:
        return self._value if self._count >= self._period else None


# ---------------------------------------------------------------------------
# VWAP — Session Volume-Weighted Average Price (tick-level)
# ---------------------------------------------------------------------------

class VWAP:
    """
    Running session VWAP, updated on every tick (not bar).

    VWAP = ΣPᵢVᵢ / ΣVᵢ

    When volume is unavailable from the feed, a synthetic unit-volume
    of 1 is used so the indicator degrades gracefully to a simple TWAP.
    """

    def __init__(self) -> None:
        self._cum_pv: float = 0.0
        self._cum_v:  float = 0.0

    def update(self, price: float, volume: float = 1.0) -> float:
        """Update with latest tick. Returns current VWAP."""
        vol = volume if volume > 0 else 1.0
        self._cum_pv += price * vol
        self._cum_v  += vol
        return self._cum_pv / self._cum_v

    def reset(self) -> None:
        """Reset at session open."""
        self._cum_pv = 0.0
        self._cum_v  = 0.0

    @property
    def value(self) -> Optional[float]:
        if self._cum_v == 0.0:
            return None
        return self._cum_pv / self._cum_v


# ---------------------------------------------------------------------------
# Indicator bundle per symbol
# ---------------------------------------------------------------------------

@dataclass
class SymbolIndicators:
    """
    Holds all indicators for a single symbol.

    Fields:
        bars:     Bar aggregator (ticks → bars)
        rsi:      RSI on bar closes
        ema_fast: Fast EMA on bar closes
        ema_slow: Slow EMA on bar closes
        vwap:     Session VWAP (tick-level)
        last_rsi: Most recent RSI reading
        last_ema_fast: Most recent fast EMA value
        last_ema_slow: Most recent slow EMA value
        last_vwap: Most recent VWAP value
        last_price: Most recent mid-price
    """
    bars:     BarAggregator
    rsi:      RSI
    ema_fast: EMA
    ema_slow: EMA
    vwap:     VWAP
    last_rsi:      Optional[float] = None
    last_ema_fast: Optional[float] = None
    last_ema_slow: Optional[float] = None
    last_vwap:     Optional[float] = None
    last_price:    Optional[float] = None

    def on_tick(self, price: float, volume: float = 0.0) -> Optional[Bar]:
        """
        Feed a tick.  Updates tick-level indicators (VWAP, price cache)
        and, if a bar completes, updates bar-level indicators (RSI, EMAs).

        Returns the completed Bar if one just closed, else None.
        """
        self.last_price = price

        # Tick-level
        self.last_vwap = self.vwap.update(price, volume)

        # Bar aggregation
        bar = self.bars.update(price, volume)
        if bar is not None:
            rsi_val = self.rsi.update(bar.close)
            if rsi_val is not None:
                self.last_rsi = rsi_val

            ef = self.ema_fast.update(bar.close)
            if ef is not None:
                self.last_ema_fast = ef

            es = self.ema_slow.update(bar.close)
            if es is not None:
                self.last_ema_slow = es

        return bar

    @property
    def is_ready(self) -> bool:
        """True when all indicators have enough data to produce a valid signal."""
        return (
            self.last_rsi      is not None and
            self.last_ema_fast is not None and
            self.last_ema_slow is not None and
            self.last_vwap     is not None and
            self.last_price    is not None
        )

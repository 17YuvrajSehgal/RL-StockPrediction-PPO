"""
Unit tests for hft/signals.py — PairsArbDetector.

Tests are standalone (no broker required) and exercise:
  - Rolling stats correctness
  - Z-score threshold crossing → correct signal type
  - State machine transitions (FLAT → LONG_A → FLAT, FLAT → SHORT_A → FLAT)
  - Cooldown debounce prevents back-to-back signals
  - Window warm-up suppresses premature signals

Run from project root:
    python -m pytest hft/tests/test_signals.py -v
"""

from __future__ import annotations

import math
import time
import sys
import os

# Make sure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from hft.signals import (
    PairsArbDetector,
    SignalType,
    _RollingStats,
)


# ---------------------------------------------------------------------------
# _RollingStats unit tests
# ---------------------------------------------------------------------------

class TestRollingStats:
    def test_empty_window_returns_nan(self):
        stats = _RollingStats(maxlen=10)
        assert math.isnan(stats.mean)
        assert math.isnan(stats.std)

    def test_single_element_std_is_nan(self):
        stats = _RollingStats(maxlen=10)
        stats.push(5.0)
        assert not math.isnan(stats.mean)
        assert math.isnan(stats.std)

    def test_mean_and_std_correctness(self):
        stats = _RollingStats(maxlen=5)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            stats.push(v)
        assert stats.is_ready
        assert abs(stats.mean - 3.0) < 1e-9
        # Sample std of [1,2,3,4,5] = sqrt(2.5) ≈ 1.5811
        assert abs(stats.std - math.sqrt(2.5)) < 1e-6

    def test_rolling_eviction(self):
        stats = _RollingStats(maxlen=3)
        for v in [1.0, 2.0, 3.0, 100.0]:  # 100 evicts 1
            stats.push(v)
        assert stats.count == 3
        assert abs(stats.mean - (2.0 + 3.0 + 100.0) / 3) < 1e-9

    def test_zscore_of_mean_is_zero(self):
        stats = _RollingStats(maxlen=4)
        for v in [1.0, 2.0, 3.0, 4.0]:
            stats.push(v)
        z = stats.zscore(stats.mean)
        assert abs(z) < 1e-9

    def test_zscore_nan_when_std_zero(self):
        stats = _RollingStats(maxlen=3)
        for _ in range(3):
            stats.push(5.0)
        z = stats.zscore(5.0)
        assert math.isnan(z)


# ---------------------------------------------------------------------------
# PairsArbDetector unit tests
# ---------------------------------------------------------------------------

class TestPairsArbDetector:

    def _make_detector(
        self,
        lookback: int = 10,
        zscore_entry: float = 2.0,
        zscore_exit: float = 0.5,
        cooldown_secs: float = 0.0,   # disable cooldown for fast tests
    ) -> PairsArbDetector:
        return PairsArbDetector(
            symbol_a="AAA",
            symbol_b="BBB",
            lookback=lookback,
            zscore_entry=zscore_entry,
            zscore_exit=zscore_exit,
            order_size=100,
            cooldown_secs=cooldown_secs,
        )

    def _warm_up(self, detector: PairsArbDetector, spread: float = 0.0, n: int = 12) -> None:
        """Feed n ticks with a fixed spread to populate the rolling window."""
        for i in range(n):
            # Add alternating noise so std is not exactly zero (which causes NaN z-scores)
            noise = (0.1 if i % 2 == 0 else -0.1)
            detector.on_quote("AAA", 100.0 + spread + noise)
            detector.on_quote("BBB", 100.0)

    # --- Warm-up phase ---

    def test_no_signal_before_window_full(self):
        det = self._make_detector(lookback=10)
        # Feed fewer than lookback ticks
        for _ in range(8):
            det.on_quote("AAA", 100.0)
            sig = det.on_quote("BBB", 100.0)
        assert sig is None

    def test_no_signal_for_unknown_symbol(self):
        det = self._make_detector()
        sig = det.on_quote("ZZZZ", 50.0)
        assert sig is None

    # --- Entry signals ---

    def test_enter_long_a_when_spread_below_entry(self):
        """spread << mean → z < -entry → ENTER_LONG_A"""
        det = self._make_detector(lookback=10, zscore_entry=2.0, cooldown_secs=0.0)
        self._warm_up(det, spread=0.0, n=11)   # window filled, spread ≈ 0
        # Now push a very low spread (A cheap vs B)
        det.on_quote("BBB", 100.0)
        sig = det.on_quote("AAA", 90.0)   # spread = 90 - 100 = -10, extreme negative z
        assert sig is not None
        assert sig.type == SignalType.ENTER_LONG_A

    def test_enter_short_a_when_spread_above_entry(self):
        """spread >> mean → z > +entry → ENTER_SHORT_A"""
        det = self._make_detector(lookback=10, zscore_entry=2.0, cooldown_secs=0.0)
        self._warm_up(det, spread=0.0, n=11)
        det.on_quote("BBB", 100.0)
        sig = det.on_quote("AAA", 110.0)   # spread = +10, extreme positive z
        assert sig is not None
        assert sig.type == SignalType.ENTER_SHORT_A

    def test_no_entry_signal_when_z_below_threshold(self):
        """Small spread move → z within ±entry → no signal."""
        det = self._make_detector(lookback=20, zscore_entry=5.0, cooldown_secs=0.0)
        self._warm_up(det, spread=0.0, n=21)
        det.on_quote("BBB", 100.0)
        sig = det.on_quote("AAA", 100.01)   # tiny move, z << 5
        assert sig is None

    # --- Exit signals ---

    def test_exit_signal_after_mean_reversion(self):
        """After ENTER_LONG_A, return spread to near-mean → EXIT."""
        det = self._make_detector(lookback=10, zscore_entry=2.0, zscore_exit=0.5, cooldown_secs=0.0)
        self._warm_up(det, spread=0.0, n=11)

        # Trigger entry
        det.on_quote("BBB", 100.0)
        sig = det.on_quote("AAA", 90.0)
        assert sig is not None and sig.type == SignalType.ENTER_LONG_A

        # Mean-revert (spread back to ~0)
        for i in range(10):
            noise = (0.1 if i % 2 == 0 else -0.1)
            det.on_quote("BBB", 100.0)
            exit_sig = det.on_quote("AAA", 100.0 + noise)
            if exit_sig:
                break

        assert exit_sig is not None
        assert exit_sig.type == SignalType.EXIT

    def test_state_resets_to_flat_after_exit(self):
        """After EXIT, detector should be FLAT."""
        det = self._make_detector(lookback=10, zscore_entry=2.0, zscore_exit=0.5, cooldown_secs=0.0)
        self._warm_up(det, spread=0.0, n=11)

        det.on_quote("BBB", 100.0)
        det.on_quote("AAA", 90.0)   # entry

        for i in range(15):
            noise = (0.1 if i % 2 == 0 else -0.1)
            det.on_quote("BBB", 100.0)
            det.on_quote("AAA", 100.0 + noise)

        assert det.current_state == "FLAT" or det.is_in_position is False

    # --- Cooldown ---

    def test_cooldown_suppresses_rapid_signals(self):
        """Two consecutive signals within cooldown window → second is suppressed."""
        det = self._make_detector(lookback=10, zscore_entry=2.0, cooldown_secs=10.0)
        self._warm_up(det, spread=0.0, n=11)

        # First extreme tick → should signal
        det.on_quote("BBB", 100.0)
        sig1 = det.on_quote("AAA", 90.0)

        # Reset state manually so we can test a second entry attempt
        det._state = det._State.FLAT   # noqa: SLF001

        # Immediately try a second signal (cooldown not elapsed)
        det.on_quote("BBB", 100.0)
        sig2 = det.on_quote("AAA", 90.0)

        # sig1 fires, sig2 is suppressed by cooldown
        assert sig1 is not None
        assert sig2 is None

    # --- Signal content ---

    def test_signal_fields_are_populated(self):
        det = self._make_detector(lookback=10, zscore_entry=2.0, cooldown_secs=0.0)
        self._warm_up(det, spread=0.0, n=11)
        det.on_quote("BBB", 100.0)
        sig = det.on_quote("AAA", 90.0)
        assert sig is not None
        assert sig.symbol_a == "AAA"
        assert sig.symbol_b == "BBB"
        assert sig.quantity == 100
        assert not math.isnan(sig.zscore)
        assert sig.zscore < 0   # negative z → long A

    # --- Reset ---

    def test_reset_clears_state(self):
        det = self._make_detector()
        self._warm_up(det, spread=0.0, n=11)
        det.reset()
        assert det.current_state == "FLAT"
        assert det.ticks_processed == 0
        assert det.signals_generated == 0

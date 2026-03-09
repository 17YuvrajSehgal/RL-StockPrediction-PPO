"""
Unit tests for ibkr/risk.py — RiskManager.

Tests are standalone (no broker required) and cover:
  - Order approval when all limits are satisfied
  - Rejection for each individual risk check
  - Daily loss halt triggers and persists
  - Position tracking after fill recording
  - Token-bucket rate limiter behaviour
  - Manual halt / resume

Run from project root:
    python -m pytest hft/tests/test_risk.py -v
"""

from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from ibkr.risk import RiskManager, RiskLimits, RejectionReason, TokenBucket


# ---------------------------------------------------------------------------
# TokenBucket tests
# ---------------------------------------------------------------------------

class TestTokenBucket:
    def test_full_bucket_allows_first_consume(self):
        bucket = TokenBucket(rate=10, capacity=10)
        assert bucket.consume() is True

    def test_empty_bucket_rejects(self):
        bucket = TokenBucket(rate=10, capacity=1)
        bucket.consume()          # drain
        assert bucket.consume() is False

    def test_refill_over_time(self):
        bucket = TokenBucket(rate=100, capacity=1)
        bucket.consume()          # drain to 0
        time.sleep(0.05)          # wait 50ms → refill ~5 tokens
        assert bucket.consume() is True


# ---------------------------------------------------------------------------
# RiskManager helpers
# ---------------------------------------------------------------------------

def _make_risk(
    max_orders_per_second: float  = 1000.0,   # effectively unlimited for most tests
    max_position_per_symbol: int  = 500,
    max_notional_per_symbol: float = 50_000.0,
    max_total_notional: float     = 100_000.0,
    max_daily_loss: float         = 1_000.0,
    min_order_quantity: int       = 1,
    max_order_quantity: int       = 500,
    readonly: bool                = False,
) -> RiskManager:
    limits = RiskLimits(
        max_orders_per_second=max_orders_per_second,
        max_position_per_symbol=max_position_per_symbol,
        max_notional_per_symbol=max_notional_per_symbol,
        max_total_notional=max_total_notional,
        max_daily_loss=max_daily_loss,
        min_order_quantity=min_order_quantity,
        max_order_quantity=max_order_quantity,
        readonly=readonly,
    )
    return RiskManager(limits)


# ---------------------------------------------------------------------------
# Basic approval
# ---------------------------------------------------------------------------

class TestRiskManagerApproval:
    def test_approves_valid_order(self):
        risk = _make_risk()
        result = risk.check_order("AAPL", "BUY", 100, 150.0)
        assert result.approved is True

    def test_approves_sell_order(self):
        risk = _make_risk()
        result = risk.check_order("MSFT", "SELL", 50, 300.0)
        assert result.approved is True


# ---------------------------------------------------------------------------
# Readonly
# ---------------------------------------------------------------------------

class TestReadonly:
    def test_readonly_rejects_all_orders(self):
        risk = _make_risk(readonly=True)
        result = risk.check_order("AAPL", "BUY", 1, 1.0)
        assert result.approved is False
        assert result.code == RejectionReason.READONLY_MODE


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class TestRateLimit:
    def test_rate_limit_rejection(self):
        # Cap at 1 order/s (burst=1), then immediately try a second
        risk = _make_risk(max_orders_per_second=1.0)
        result1 = risk.check_order("AAPL", "BUY", 1, 10.0)
        result2 = risk.check_order("AAPL", "BUY", 1, 10.0)
        assert result1.approved is True
        assert result2.approved is False
        assert result2.code == RejectionReason.RATE_LIMIT


# ---------------------------------------------------------------------------
# Quantity validation
# ---------------------------------------------------------------------------

class TestQuantityValidation:
    def test_zero_quantity_rejected(self):
        risk = _make_risk(min_order_quantity=1)
        result = risk.check_order("AAPL", "BUY", 0, 100.0)
        assert result.approved is False
        assert result.code == RejectionReason.INVALID_QUANTITY

    def test_quantity_above_max_rejected(self):
        risk = _make_risk(max_order_quantity=100)
        result = risk.check_order("AAPL", "BUY", 101, 100.0)
        assert result.approved is False
        assert result.code == RejectionReason.INVALID_QUANTITY

    def test_exact_max_quantity_approved(self):
        risk = _make_risk(max_order_quantity=100)
        result = risk.check_order("AAPL", "BUY", 100, 100.0)
        assert result.approved is True


# ---------------------------------------------------------------------------
# Price validation
# ---------------------------------------------------------------------------

class TestPriceValidation:
    def test_zero_price_rejected(self):
        risk = _make_risk()
        result = risk.check_order("AAPL", "BUY", 1, 0.0)
        assert result.approved is False
        assert result.code == RejectionReason.INVALID_PRICE

    def test_negative_price_rejected(self):
        risk = _make_risk()
        result = risk.check_order("AAPL", "BUY", 1, -5.0)
        assert result.approved is False
        assert result.code == RejectionReason.INVALID_PRICE


# ---------------------------------------------------------------------------
# Position limits
# ---------------------------------------------------------------------------

class TestPositionLimits:
    def test_position_limit_rejection(self):
        risk = _make_risk(max_position_per_symbol=100)
        # Record a fill that brings us to the limit
        risk.record_fill("AAPL", "BUY", 100, 150.0)
        # Next order would exceed it
        result = risk.check_order("AAPL", "BUY", 1, 150.0)
        assert result.approved is False
        assert result.code == RejectionReason.SYMBOL_EXPOSURE

    def test_sell_within_limit_approved(self):
        risk = _make_risk(max_position_per_symbol=100)
        risk.record_fill("AAPL", "BUY", 100, 150.0)
        # Selling reduces position — should be fine
        result = risk.check_order("AAPL", "SELL", 50, 150.0)
        assert result.approved is True


# ---------------------------------------------------------------------------
# Notional limits
# ---------------------------------------------------------------------------

class TestNotionalLimits:
    def test_symbol_notional_limit_rejected(self):
        risk = _make_risk(max_notional_per_symbol=1_000.0)
        # $1001 order at $100/share = 10.01 shares → exceeds $1000
        result = risk.check_order("AAPL", "BUY", 11, 100.0)
        assert result.approved is False
        assert result.code == RejectionReason.SYMBOL_EXPOSURE

    def test_total_notional_limit_rejected(self):
        risk = _make_risk(max_total_notional=500.0, max_notional_per_symbol=1_000.0)
        # Already have $400 in AAPL, try to add $200 in MSFT → exceeds $500 total
        risk.record_fill("AAPL", "BUY", 4, 100.0)   # $400
        result = risk.check_order("MSFT", "BUY", 2, 100.0)   # $200 more
        assert result.approved is False
        assert result.code == RejectionReason.PORTFOLIO_EXPOSURE


# ---------------------------------------------------------------------------
# Daily loss limit
# ---------------------------------------------------------------------------

class TestDailyLossLimit:
    def test_daily_loss_triggers_halt(self):
        risk = _make_risk(max_daily_loss=100.0)
        risk.record_pnl(-101.0)   # exceed limit
        result = risk.check_order("AAPL", "BUY", 1, 10.0)
        assert result.approved is False
        assert result.code == RejectionReason.DAILY_LOSS_LIMIT
        assert risk.is_halted is True

    def test_pnl_below_limit_still_trades(self):
        risk = _make_risk(max_daily_loss=100.0)
        risk.record_pnl(-99.0)
        result = risk.check_order("AAPL", "BUY", 1, 10.0)
        assert result.approved is True


# ---------------------------------------------------------------------------
# Manual halt / resume
# ---------------------------------------------------------------------------

class TestHaltResume:
    def test_manual_halt_blocks_orders(self):
        risk = _make_risk()
        risk.halt("test halt")
        result = risk.check_order("AAPL", "BUY", 1, 10.0)
        assert result.approved is False
        assert result.code == RejectionReason.TRADING_HALTED

    def test_resume_allows_orders(self):
        risk = _make_risk()
        risk.halt("test halt")
        risk.resume()
        result = risk.check_order("AAPL", "BUY", 1, 10.0)
        assert result.approved is True


# ---------------------------------------------------------------------------
# Fill recording
# ---------------------------------------------------------------------------

class TestFillRecording:
    def test_fill_updates_position(self):
        risk = _make_risk()
        risk.record_fill("AAPL", "BUY", 100, 150.0)
        assert risk.get_position("AAPL") == 100

    def test_sell_fill_reduces_position(self):
        risk = _make_risk()
        risk.record_fill("AAPL", "BUY", 100, 150.0)
        risk.record_fill("AAPL", "SELL", 50, 155.0)
        assert risk.get_position("AAPL") == 50

    def test_fill_updates_notional(self):
        risk = _make_risk()
        risk.record_fill("AAPL", "BUY", 10, 100.0)
        assert abs(risk.get_notional("AAPL") - 1_000.0) < 0.01

    def test_total_notional_aggregates_symbols(self):
        risk = _make_risk()
        risk.record_fill("AAPL", "BUY", 10, 100.0)   # $1000
        risk.record_fill("MSFT", "BUY", 5,  200.0)   # $1000
        assert abs(risk.total_notional - 2_000.0) < 0.01

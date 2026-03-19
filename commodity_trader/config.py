"""
Configuration for the Professional Commodity HFT Trading Bot.

All parameters are organised into composable frozen sub-configs so that
only the section you care about needs to be overridden.
"""
from __future__ import annotations
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Indicator parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IndicatorConfig:
    """
    Parameters for the technical indicators.

    Attributes:
        bar_duration_secs:  How many seconds of ticks form one OHLCV bar.
                            Shorter = more responsive; longer = less noise.
        rsi_period:         Number of bars for the RSI computation (Wilder's).
        ema_fast:           Fast EMA period (bars).
        ema_slow:           Slow EMA period (bars).
        vwap_reset_daily:   Always start a fresh VWAP at session open.
    """
    bar_duration_secs: float = 10.0    # 10-second bars
    rsi_period:        int   = 9       # RSI-9 on bars
    ema_fast:          int   = 5       # EMA-5 cross
    ema_slow:          int   = 13      # EMA-13 cross
    vwap_reset_daily:  bool  = True


# ---------------------------------------------------------------------------
# Signal / entry parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalConfig:
    """
    Confluence thresholds that ALL must be satisfied before a signal fires.

    Attributes:
        rsi_entry:          RSI must be ABOVE this to enter long on a symbol.
        rsi_exit:           RSI below this triggers exit in a live position.
        vwap_dev_entry_pct: Price must be THIS % above VWAP to confirm trend.
        ema_confirmation:   Require fast-EMA > slow-EMA for long; reversed for short.
        cooldown_secs:      Minimum seconds between any two entry signals.
        min_spread_bps:     Skip entry if the bid-ask spread is wider than this
                            (avoids toxic fills in thin markets).
    """
    rsi_entry:           float = 58.0
    rsi_exit:            float = 42.0
    vwap_dev_entry_pct:  float = 0.05   # 0.05% above VWAP to confirm trend
    ema_confirmation:    bool  = True
    cooldown_secs:       float = 3.0
    min_spread_bps:      float = 5.0    # skip if spread > 5 bps


# ---------------------------------------------------------------------------
# Position / risk parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PositionConfig:
    """
    Position sizing and risk controls.

    Attributes:
        order_size:             Shares per leg.
        stop_loss_bps:          Hard stop-loss on net basket P&L in basis points.
        trailing_stop_bps:      Trailing stop activates once unrealised gain
                                exceeds this many bps; then follows the peak.
        trail_distance_bps:     How far the trailing stop sits below the peak.
        limit_offset:           Price improvement from mid for limit orders ($).
        limit_timeout_secs:     Fall back to market order if limit not filled
                                within this many seconds.
        max_daily_loss_usd:     Session circuit breaker — halt new entries if
                                realised losses exceed this amount.
    """
    order_size:           int   = 500
    stop_loss_bps:        float = 15.0   # 0.15% hard stop
    trailing_stop_bps:    float = 20.0   # activate trailing stop above 0.20% gain
    trail_distance_bps:   float = 10.0   # trail 0.10% below the peak
    limit_offset:         float = 0.01   # $0.01 price improvement on limit entry
    limit_timeout_secs:   float = 8.0    # fall back to market after 8 s
    max_daily_loss_usd:   float = 500.0  # halt trading after $500 session loss


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CommodityTraderConfig:
    """
    Master configuration for the commodity HFT bot.

    Symbols traded:
        AGQ  — ProShares Ultra Silver (2× leveraged silver)
        UGL  — ProShares Ultra Gold   (2× leveraged gold)
        UCO  — ProShares Ultra DJ-AIG Crude Oil (2× leveraged crude)

    Regimes:
        LONG_GOLD:  Buy AGQ + UGL, Sell UCO  (when gold/silver trending up)
        LONG_OIL:   Buy UCO, Sell AGQ + UGL  (when crude trending up)
    """
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    signal:     SignalConfig    = field(default_factory=SignalConfig)
    position:   PositionConfig  = field(default_factory=PositionConfig)

    symbols: tuple = ("AGQ", "UGL", "UCO")

    def to_risk_limits(self):
        """Map position config onto ibkr.risk.RiskLimits."""
        from ibkr.risk import RiskLimits
        # Estimate worst-case notional: 500 shares × ~$100/share × 3 legs
        per_sym_notional = self.position.order_size * 200.0
        return RiskLimits(
            max_orders_per_second=45.0,
            max_position_per_symbol=self.position.order_size * 2,
            max_notional_per_symbol=per_sym_notional,
            max_total_notional=per_sym_notional * 3,
            max_daily_loss=self.position.max_daily_loss_usd,
            min_order_quantity=1,
            max_order_quantity=self.position.order_size,
        )

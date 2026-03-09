"""
HFT system configuration.

All tunable parameters live here, organised into composable sub-configs.
Each sub-config is a frozen dataclass so parameters cannot be mutated
after construction — this prevents accidental runtime changes.

Hierarchy:
  HFTConfig
   ├── PairsArbConfig   — statistical pairs arbitrage parameters
   ├── RiskConfig       — position / loss / rate limits (feeds into ibkr.risk)
   └── ExecutorConfig   — order submission behaviour

To customise, override only the fields you care about:

    >>> config = HFTConfig(
    ...     pairs_arb=PairsArbConfig(
    ...         zscore_entry=2.5,
    ...         order_size=50,
    ...     ),
    ...     risk=RiskConfig(max_daily_loss=250.0),
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PairsArbConfig:
    """
    Parameters for the Statistical Pairs Arbitrage strategy.

    The strategy monitors the z-score of the price spread between correlated
    pairs. When the z-score exceeds `zscore_entry` (in either direction) an
    entry is triggered; once the spread mean-reverts to within `zscore_exit`
    the position is closed.

    Attributes:
        pairs:          List of (symbol_a, symbol_b) pairs to trade.
                        The first symbol is the "primary" leg.
        zscore_entry:   Threshold to OPEN a position (default 2.0 σ).
                        Higher = fewer but more confident trades.
        zscore_exit:    Threshold to CLOSE a position (default 0.5 σ).
                        Lower = closer to mean-reversion; higher = faster exit.
        lookback:       Rolling window size in ticks used to compute the
                        spread's mean and standard deviation.
        order_size:     Shares per leg per trade signal.
        use_limit:      If True, submit limit orders; if False, use market.
        limit_offset:   Price improvement offset from the mid in dollars.
                        BUY price  = mid - limit_offset
                        SELL price = mid + limit_offset
        min_spread_bps: Minimum bid-ask spread (in basis points) below which
                        trades are skipped to avoid toxic fills.
        cooldown_secs:  Minimum seconds between consecutive signals for the
                        same pair (prevents signal chatter).
    """
    pairs: tuple[tuple[str, str], ...] = (
        ("SPY",  "SPX"),    # S&P 500 ETF (SPY) vs Index (SPX)
    )
    zscore_entry:    float = 2.0
    zscore_exit:     float = 0.5
    lookback:        int   = 100     # ticks
    order_size:      int   = 100     # shares per leg
    use_limit:       bool  = True
    limit_offset:    float = 0.01    # $0.01 improvement from mid
    min_spread_bps:  float = 0.0     # skip trades only if positive
    cooldown_secs:   float = 5.0     # seconds between signals per pair


@dataclass(frozen=True)
class RiskConfig:
    """
    Risk management limits (maps directly onto ibkr.risk.RiskLimits).

    With a $100k paper account the defaults below allow up to
    $20k notional per symbol and full account utilisation.

    Attributes:
        max_orders_per_second:    IBKR enforces 50/s; stay safely below.
        max_position_per_symbol:  Max absolute shares per symbol (both legs).
        max_notional_per_symbol:  Max USD notional allocated to any one symbol.
        max_total_notional:       Max USD notional across the entire portfolio.
        max_daily_loss:           Auto-halt when session loss exceeds this.
        min_order_quantity:       Smallest sensible order size.
        max_order_quantity:       Largest single order (prevents fat-finger).
    """
    max_orders_per_second:   float = 45.0
    max_position_per_symbol: int   = 2_000
    max_notional_per_symbol: float = 20_000.0
    max_total_notional:      float = 100_000.0    # full $100k paper account
    max_daily_loss:          float = 1_000.0      # 1% of account
    min_order_quantity:      int   = 1
    max_order_quantity:      int   = 1_000


@dataclass(frozen=True)
class ExecutorConfig:
    """
    Order execution behaviour.

    Attributes:
        queue_max_size:       Max pending orders in the internal queue.
                              Excess orders are dropped with a warning.
        order_ack_timeout:    Seconds to wait for order acknowledgment
                              before considering it stale.
        fill_timeout:         Seconds to wait for a limit order fill before
                              cancelling and moving on.
        cancel_on_shutdown:   If True, cancel all open orders on stop().
    """
    queue_max_size:    int   = 500
    order_ack_timeout: float = 2.0
    fill_timeout:      float = 30.0
    cancel_on_shutdown: bool = True


@dataclass(frozen=True)
class LoggingConfig:
    """
    Logging and persistence settings.

    Attributes:
        log_dir:         Directory for all HFT log files.
        save_fills_csv:  Write every fill to a CSV for post-session analysis.
        dashboard_interval: Seconds between dashboard prints to stdout.
        debug_signals:   Emit DEBUG log for every signal evaluated (verbose).
    """
    log_dir:             str   = "hft_logs"
    save_fills_csv:      bool  = True
    dashboard_interval:  float = 10.0    # seconds
    debug_signals:       bool  = False


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HFTConfig:
    """
    Master configuration for the HFT arbitrage engine.

    Compose sub-configs to build a complete configuration object.  All
    sub-configs have sensible defaults so the simplest usage is:

        >>> config = HFTConfig()   # ready to go with $100k paper defaults

    Or override specific sub-configs:

        >>> config = HFTConfig(
        ...     pairs_arb=PairsArbConfig(zscore_entry=2.5),
        ...     risk=RiskConfig(max_daily_loss=500.0),
        ... )

    Attributes:
        pairs_arb:              Pairs arbitrage strategy parameters.
        risk:                   Risk management limits.
        executor:               Order executor behaviour.
        logging:                Logging and persistence settings.
        enable_pairs_arb:       Master toggle for the pairs arb strategy.
        enable_etf_arb:         ETF basket arb (not yet implemented).
        enable_market_making:   Market making strategy (not yet implemented).
    """
    # Strategy sub-configs
    pairs_arb: PairsArbConfig = field(default_factory=PairsArbConfig)
    risk:      RiskConfig     = field(default_factory=RiskConfig)
    executor:  ExecutorConfig = field(default_factory=ExecutorConfig)
    logging:   LoggingConfig  = field(default_factory=LoggingConfig)

    # Strategy toggles
    enable_pairs_arb:      bool = True
    enable_etf_arb:        bool = False   # placeholder
    enable_market_making:  bool = False   # placeholder

    def all_symbols(self) -> list[str]:
        """Return a deduplicated list of all symbols across enabled strategies."""
        symbols: set[str] = set()
        if self.enable_pairs_arb:
            for a, b in self.pairs_arb.pairs:
                symbols.add(a)
                symbols.add(b)
        return sorted(symbols)

    def to_risk_limits(self):
        """
        Convert the risk sub-config into an ibkr.risk.RiskLimits object.

        Returns:
            ibkr.risk.RiskLimits instance populated from self.risk
        """
        from ibkr.risk import RiskLimits
        return RiskLimits(
            max_orders_per_second=self.risk.max_orders_per_second,
            max_position_per_symbol=self.risk.max_position_per_symbol,
            max_notional_per_symbol=self.risk.max_notional_per_symbol,
            max_total_notional=self.risk.max_total_notional,
            max_daily_loss=self.risk.max_daily_loss,
            min_order_quantity=self.risk.min_order_quantity,
            max_order_quantity=self.risk.max_order_quantity,
        )

    def __repr__(self) -> str:
        enabled = [
            s for s, flag in [
                ("pairs_arb", self.enable_pairs_arb),
                ("etf_arb",   self.enable_etf_arb),
                ("market_making", self.enable_market_making),
            ] if flag
        ]
        return (
            f"HFTConfig("
            f"strategies={enabled}, "
            f"pairs={len(self.pairs_arb.pairs)}, "
            f"max_daily_loss={self.risk.max_daily_loss})"
        )

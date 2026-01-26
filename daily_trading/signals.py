"""
Trading signal generation from trained PPO models.

This module provides professional-grade signal generation for trading decisions:
- Clear BUY/SELL/HOLD signals with confidence scoring
- Human-readable reasoning for each signal
- JSON export for integration with trading systems
- Proper model loading with observation normalization

Design Principles:
    - Signals are deterministic for reproducibility
    - Confidence reflects model certainty
    - All signals include audit trail information
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from daily_trading.config import SignalConfig, PathConfig
from daily_trading.utils import DataLoader, TradingCalendar

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from swing_trading import SwingTradingEnv, SwingTradingConfig


@dataclass
class TradingSignal:
    """
    Trading signal output from model prediction.
    
    This is the primary output of the signal generation system.
    It contains all information needed to make and audit trading decisions.
    
    Attributes:
        ticker: Stock ticker symbol
        signal_date: Date the signal is FOR (next trading day)
        generated_at: Timestamp when signal was generated
        signal: Trading action (BUY, SELL, HOLD)
        target_weight: Raw model output [-1, 1]
        confidence: Confidence score [0, 1]
        current_price: Last known price when signal generated
        reasoning: Human-readable explanation
        model_path: Path to model used for generation
        data_as_of: Most recent data date used
    
    Example:
        >>> signal = TradingSignal(
        ...     ticker="AAPL",
        ...     signal_date=date(2026, 1, 27),
        ...     generated_at=datetime.now(),
        ...     signal="BUY",
        ...     target_weight=0.65,
        ...     confidence=0.72,
        ...     current_price=185.50,
        ...     reasoning="Strong bullish signal with 65% long target",
        ...     model_path=Path("models/AAPL/2026-01-24/model.zip"),
        ...     data_as_of=date(2026, 1, 24)
        ... )
        >>> print(f"Signal: {signal.signal} with {signal.confidence:.0%} confidence")
    """
    
    ticker: str
    signal_date: date
    generated_at: datetime
    signal: Literal["BUY", "SELL", "HOLD"]
    target_weight: float
    confidence: float
    current_price: float
    reasoning: str
    model_path: Path
    data_as_of: date
    
    # Optional fields for post-hoc analysis
    actual_return: Optional[float] = None
    actual_direction: Optional[Literal["UP", "DOWN", "FLAT"]] = None
    was_correct: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "ticker": self.ticker,
            "signal_date": self.signal_date.isoformat(),
            "generated_at": self.generated_at.isoformat(),
            "signal": self.signal,
            "target_weight": round(self.target_weight, 4),
            "confidence": round(self.confidence, 4),
            "current_price": round(self.current_price, 2),
            "reasoning": self.reasoning,
            "model_path": str(self.model_path),
            "data_as_of": self.data_as_of.isoformat(),
            "actual_return": round(self.actual_return, 6) if self.actual_return else None,
            "actual_direction": self.actual_direction,
            "was_correct": self.was_correct,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TradingSignal:
        """Create from dictionary."""
        return cls(
            ticker=data["ticker"],
            signal_date=date.fromisoformat(data["signal_date"]),
            generated_at=datetime.fromisoformat(data["generated_at"]),
            signal=data["signal"],
            target_weight=data["target_weight"],
            confidence=data["confidence"],
            current_price=data["current_price"],
            reasoning=data["reasoning"],
            model_path=Path(data["model_path"]),
            data_as_of=date.fromisoformat(data["data_as_of"]),
            actual_return=data.get("actual_return"),
            actual_direction=data.get("actual_direction"),
            was_correct=data.get("was_correct"),
        )
    
    def save(self, path: Path) -> None:
        """Save signal to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> TradingSignal:
        """Load signal from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"TradingSignal({self.ticker} | {self.signal_date} | "
            f"{self.signal} @ {self.confidence:.0%} confidence | "
            f"Target: {self.target_weight:+.2f})"
        )
    
    def format_report(self) -> str:
        """Format as human-readable report."""
        lines = [
            "=" * 60,
            f"TRADING SIGNAL: {self.ticker}",
            "=" * 60,
            f"Signal Date:    {self.signal_date} (next trading day)",
            f"Generated:      {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "-" * 60,
            f"Signal:         {self.signal}",
            f"Confidence:     {self.confidence:.1%}",
            f"Target Weight:  {self.target_weight:+.2%}",
            f"Current Price:  ${self.current_price:.2f}",
            "-" * 60,
            f"Reasoning:      {self.reasoning}",
            "-" * 60,
            f"Model:          {self.model_path.name}",
            f"Data As Of:     {self.data_as_of}",
            "=" * 60,
        ]
        return "\n".join(lines)


class SignalGenerator:
    """
    Generate trading signals from trained PPO models.
    
    This class handles all aspects of signal generation:
    - Loading trained models with normalization stats
    - Building observation from recent market data
    - Running model inference
    - Interpreting output as actionable signal
    
    Attributes:
        config: Signal configuration
        paths: Path configuration
        env_config: Trading environment configuration
        data_loader: Data loading utility
        calendar: Trading calendar
    
    Example:
        >>> generator = SignalGenerator(signal_config, paths)
        >>> 
        >>> # Generate signal for next Monday
        >>> signal = generator.generate(
        ...     ticker="AAPL",
        ...     for_date=date(2026, 1, 27),
        ...     model_path=Path("models/AAPL/2026-01-24/model.zip")
        ... )
        >>> 
        >>> print(signal.format_report())
    """
    
    def __init__(
        self,
        config: SignalConfig,
        paths: PathConfig,
        env_config: Optional[SwingTradingConfig] = None,
    ):
        """
        Initialize signal generator.
        
        Args:
            config: Signal interpretation configuration
            paths: Path configuration
            env_config: Trading environment configuration
        """
        self.config = config
        self.paths = paths
        self.env_config = env_config or SwingTradingConfig()
        
        self.data_loader = DataLoader(paths.data_dir)
        self.calendar = TradingCalendar()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Cache for loaded models
        self._model_cache: Dict[str, tuple[PPO, VecNormalize]] = {}
    
    def generate(
        self,
        ticker: str,
        for_date: date,
        model_path: Optional[Path] = None,
        data_as_of: Optional[date] = None,
    ) -> TradingSignal:
        """
        Generate trading signal for specified date.
        
        Args:
            ticker: Stock ticker
            for_date: Date to generate signal for (should be next trading day)
            model_path: Path to trained model. If None, uses latest available.
            data_as_of: Data cutoff date. If None, uses previous trading day.
        
        Returns:
            TradingSignal with BUY/SELL/HOLD recommendation
        
        Raises:
            ValueError: If no model or insufficient data available
        """
        self.logger.info(f"Generating signal for {ticker} on {for_date}")
        
        # Resolve data cutoff date
        if data_as_of is None:
            data_as_of = self.calendar.previous_trading_day(for_date)
        
        # Find model if not specified
        if model_path is None:
            model_path = self._find_model(ticker, data_as_of)
            if model_path is None:
                raise ValueError(f"No model found for {ticker} as of {data_as_of}")
        
        self.logger.info(f"Using model: {model_path}")
        self.logger.info(f"Data as of: {data_as_of}")
        
        # Load model and create inference environment
        model, env = self._load_model(ticker, model_path, data_as_of)
        
        # Get current price
        df = self.data_loader.load_ticker(ticker, as_of_date=data_as_of)
        current_price = float(df['close'].iloc[-1])
        
        # Get observation and predict
        obs = env.reset()
        action, _ = model.predict(obs, deterministic=self.config.deterministic)
        
        # Extract weight
        target_weight = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        target_weight = np.clip(target_weight, -1.0, 1.0)
        
        # Interpret signal
        signal, confidence = self.config.interpret_weight(target_weight)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(signal, target_weight, confidence)
        
        # Create signal object
        trading_signal = TradingSignal(
            ticker=ticker,
            signal_date=for_date,
            generated_at=datetime.now(),
            signal=signal,
            target_weight=target_weight,
            confidence=confidence,
            current_price=current_price,
            reasoning=reasoning,
            model_path=model_path,
            data_as_of=data_as_of,
        )
        
        # Save signal
        signal_path = self.paths.signal_path(ticker, for_date)
        trading_signal.save(signal_path)
        self.logger.info(f"Signal saved to {signal_path}")
        
        return trading_signal
    
    def generate_batch(
        self,
        tickers: List[str],
        for_date: date,
    ) -> Dict[str, TradingSignal]:
        """
        Generate signals for multiple tickers.
        
        Args:
            tickers: List of stock tickers
            for_date: Date to generate signals for
        
        Returns:
            Dictionary mapping ticker to signal
        """
        signals = {}
        
        for ticker in tickers:
            try:
                signals[ticker] = self.generate(ticker, for_date)
            except Exception as e:
                self.logger.error(f"Failed to generate signal for {ticker}: {e}")
        
        return signals
    
    def _find_model(self, ticker: str, as_of_date: date) -> Optional[Path]:
        """Find most recent model for ticker."""
        ticker_dir = self.paths.models_dir / ticker
        
        if not ticker_dir.exists():
            return None
        
        # Get all dated directories
        model_dates = []
        for d in ticker_dir.iterdir():
            if d.is_dir():
                try:
                    model_date = date.fromisoformat(d.name)
                    if model_date <= as_of_date:
                        model_file = d / "model.zip"
                        if model_file.exists():
                            model_dates.append((model_date, model_file))
                except ValueError:
                    continue
        
        if not model_dates:
            return None
        
        # Return most recent
        model_dates.sort(key=lambda x: x[0], reverse=True)
        return model_dates[0][1]
    
    def _load_model(
        self,
        ticker: str,
        model_path: Path,
        data_as_of: date,
    ) -> tuple[PPO, VecNormalize]:
        """Load model and create inference environment."""
        cache_key = str(model_path)
        
        # Check cache
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        # Load data for environment
        df = self.data_loader.load_ticker(ticker, as_of_date=data_as_of)
        
        # Create environment with adjusted config
        from dataclasses import replace
        
        lookback = min(60, len(df) // 3)
        episode_length = min(126, len(df) - lookback - 10)
        
        env_cfg = replace(
            self.env_config.environment,
            lookback=lookback,
            episode_length=episode_length,
            random_start=False,
        )
        config = replace(self.env_config, environment=env_cfg)
        
        # Create and wrap environment
        env = SwingTradingEnv(df, config)
        vec_env = DummyVecEnv([lambda: env])
        
        # Load VecNormalize if available
        vecnorm_path = model_path.parent / "vecnormalize.pkl"
        if vecnorm_path.exists():
            vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
        else:
            vec_env = VecNormalize(
                vec_env,
                training=False,
                norm_obs=True,
                norm_reward=False,
            )
        
        # Load model
        model = PPO.load(model_path)
        
        # Cache for reuse
        self._model_cache[cache_key] = (model, vec_env)
        
        return model, vec_env
    
    def _generate_reasoning(
        self,
        signal: Literal["BUY", "SELL", "HOLD"],
        weight: float,
        confidence: float,
    ) -> str:
        """Generate human-readable reasoning for signal."""
        weight_pct = abs(weight) * 100
        
        if signal == "BUY":
            strength = "strong" if confidence > 0.7 else "moderate" if confidence > 0.4 else "weak"
            return (
                f"Model predicts {strength} bullish signal with {weight_pct:.0f}% long target. "
                f"Confidence: {confidence:.0%}. "
                f"Action: Consider entering or increasing long position."
            )
        
        elif signal == "SELL":
            strength = "strong" if confidence > 0.7 else "moderate" if confidence > 0.4 else "weak"
            return (
                f"Model predicts {strength} bearish signal with {weight_pct:.0f}% short target. "
                f"Confidence: {confidence:.0%}. "
                f"Action: Consider exiting long or entering short position."
            )
        
        else:  # HOLD
            return (
                f"Model predicts neutral/uncertain market conditions with {weight_pct:.0f}% target. "
                f"Confidence: {confidence:.0%}. "
                f"Action: Hold current position, avoid new entries."
            )
    
    def clear_cache(self) -> None:
        """Clear model cache."""
        self._model_cache.clear()


def get_next_trading_day_signal(
    ticker: str,
    paths: Optional[PathConfig] = None,
    signal_config: Optional[SignalConfig] = None,
) -> TradingSignal:
    """
    Convenience function to get signal for next trading day.
    
    Args:
        ticker: Stock ticker
        paths: Path configuration
        signal_config: Signal configuration
    
    Returns:
        TradingSignal for next trading day
    """
    from datetime import date
    
    paths = paths or PathConfig()
    signal_config = signal_config or SignalConfig()
    calendar = TradingCalendar()
    
    # Find next trading day
    today = date.today()
    next_day = calendar.next_trading_day(today)
    
    # Generate signal
    generator = SignalGenerator(signal_config, paths)
    return generator.generate(ticker, next_day)

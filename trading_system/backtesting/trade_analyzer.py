"""
Trade analysis module for detailed trade metrics and insights.

Provides comprehensive analysis of trading performance including:
- Win/loss analysis
- Trade quality scoring
- Holding period analysis
- Risk metrics per trade
- Statistical analysis
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from trading_system.backtesting.backtest_engine import Trade, BacktestResult


class TradeAnalyzer:
    """
    Analyze trading performance in detail.
    
    Provides:
    - Win rate and profit factor
    - Average win/loss analysis
    - Trade quality metrics
    - Risk-adjusted performance
    - Statistical insights
    
    Example:
        >>> analyzer = TradeAnalyzer(backtest_result)
        >>> metrics = analyzer.analyze()
        >>> print(f"Win Rate: {metrics['win_rate']:.2%}")
    """
    
    def __init__(self, result: BacktestResult):
        """
        Initialize trade analyzer.
        
        Args:
            result: BacktestResult from backtesting engine
        """
        self.result = result
        self.trades = result.trades
        self.df = result.to_dataframe()
    
    def analyze(self) -> Dict[str, any]:
        """
        Perform comprehensive trade analysis.
        
        Returns:
            Dictionary of analysis metrics
        """
        if len(self.trades) == 0:
            return self._empty_analysis()
        
        metrics = {}
        
        # Basic statistics
        metrics.update(self._basic_stats())
        
        # Win/loss analysis
        metrics.update(self._win_loss_analysis())
        
        # Holding period analysis
        metrics.update(self._holding_period_analysis())
        
        # Risk metrics
        metrics.update(self._risk_metrics())
        
        # Trade quality
        metrics.update(self._trade_quality())
        
        return metrics
    
    def _empty_analysis(self) -> Dict[str, any]:
        """Return empty analysis for no trades."""
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
        }
    
    def _basic_stats(self) -> Dict[str, any]:
        """Calculate basic trading statistics."""
        return {
            'total_trades': len(self.trades),
            'long_trades': len([t for t in self.trades if t.direction == 'long']),
            'short_trades': len([t for t in self.trades if t.direction == 'short']),
            'total_pnl': sum(t.pnl for t in self.trades),
            'total_costs': sum(t.costs for t in self.trades),
        }
    
    def _win_loss_analysis(self) -> Dict[str, any]:
        """Analyze winning and losing trades."""
        winners = [t for t in self.trades if t.is_winner()]
        losers = [t for t in self.trades if not t.is_winner()]
        
        win_rate = len(winners) / len(self.trades) if self.trades else 0.0
        
        total_wins = sum(t.pnl for t in winners)
        total_losses = abs(sum(t.pnl for t in losers))
        
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        avg_win = np.mean([t.pnl for t in winners]) if winners else 0.0
        avg_loss = np.mean([t.pnl for t in losers]) if losers else 0.0
        
        avg_win_pct = np.mean([t.pnl_pct for t in winners]) if winners else 0.0
        avg_loss_pct = np.mean([t.pnl_pct for t in losers]) if losers else 0.0
        
        # Largest win/loss
        largest_win = max([t.pnl for t in winners]) if winners else 0.0
        largest_loss = min([t.pnl for t in losers]) if losers else 0.0
        
        # Consecutive wins/losses
        consecutive_wins, consecutive_losses = self._consecutive_streaks()
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'total_wins': total_wins,
            'total_losses': total_losses,
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses,
        }
    
    def _consecutive_streaks(self) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses."""
        if not self.trades:
            return 0, 0
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in self.trades:
            if trade.is_winner():
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        return max_wins, max_losses
    
    def _holding_period_analysis(self) -> Dict[str, any]:
        """Analyze holding periods."""
        if not self.trades:
            return {}
        
        holding_periods = [t.holding_period for t in self.trades]
        
        return {
            'avg_holding_period': np.mean(holding_periods),
            'median_holding_period': np.median(holding_periods),
            'min_holding_period': np.min(holding_periods),
            'max_holding_period': np.max(holding_periods),
        }
    
    def _risk_metrics(self) -> Dict[str, any]:
        """Calculate risk metrics."""
        if not self.trades:
            return {}
        
        # MAE and MFE analysis
        mae_values = [t.mae for t in self.trades]
        mfe_values = [t.mfe for t in self.trades]
        
        # Expectancy
        win_rate = len([t for t in self.trades if t.is_winner()]) / len(self.trades)
        avg_win = np.mean([t.pnl for t in self.trades if t.is_winner()]) if win_rate > 0 else 0
        avg_loss = np.mean([t.pnl for t in self.trades if not t.is_winner()]) if win_rate < 1 else 0
        
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # Kelly criterion
        if avg_loss != 0:
            kelly = (win_rate / abs(avg_loss)) - ((1 - win_rate) / avg_win) if avg_win > 0 else 0
        else:
            kelly = 0
        
        return {
            'avg_mae': np.mean(mae_values),
            'avg_mfe': np.mean(mfe_values),
            'expectancy': expectancy,
            'kelly_criterion': kelly,
        }
    
    def _trade_quality(self) -> Dict[str, any]:
        """Assess trade quality."""
        if not self.trades:
            return {}
        
        # Efficiency: how much of MFE was captured
        efficiencies = []
        for trade in self.trades:
            if trade.mfe > 0:
                efficiency = trade.pnl / trade.mfe
                efficiencies.append(efficiency)
        
        avg_efficiency = np.mean(efficiencies) if efficiencies else 0.0
        
        # Risk/reward ratio
        rr_ratios = []
        for trade in self.trades:
            if trade.mae < 0:
                rr = trade.pnl / abs(trade.mae)
                rr_ratios.append(rr)
        
        avg_rr = np.mean(rr_ratios) if rr_ratios else 0.0
        
        return {
            'avg_efficiency': avg_efficiency,
            'avg_risk_reward': avg_rr,
        }
    
    def get_best_trades(self, n: int = 5) -> List[Trade]:
        """Get top N best trades by P&L."""
        return sorted(self.trades, key=lambda t: t.pnl, reverse=True)[:n]
    
    def get_worst_trades(self, n: int = 5) -> List[Trade]:
        """Get top N worst trades by P&L."""
        return sorted(self.trades, key=lambda t: t.pnl)[:n]
    
    def get_trade_summary_table(self) -> pd.DataFrame:
        """
        Generate summary table of all trades.
        
        Returns:
            DataFrame with trade details
        """
        if self.df.empty:
            return pd.DataFrame()
        
        # Select key columns
        columns = [
            'entry_date', 'exit_date', 'direction', 'entry_price', 'exit_price',
            'shares', 'pnl', 'pnl_pct', 'costs', 'holding_period', 'winner'
        ]
        
        summary = self.df[columns].copy()
        
        # Format for display
        summary['pnl'] = summary['pnl'].apply(lambda x: f"${x:,.2f}")
        summary['pnl_pct'] = summary['pnl_pct'].apply(lambda x: f"{x:.2%}")
        summary['costs'] = summary['costs'].apply(lambda x: f"${x:.2f}")
        
        return summary

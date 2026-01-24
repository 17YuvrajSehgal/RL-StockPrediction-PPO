"""
Professional visualization module for backtesting results.

Creates interactive charts using Plotly for:
- Price charts with entry/exit markers
- Equity curves
- Drawdown charts
- Trade distribution plots
- Performance dashboards
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trading_system.backtesting.backtest_engine import BacktestResult, Trade


class BacktestVisualizer:
    """
    Create professional interactive visualizations for backtest results.
    
    Features:
    - Interactive Plotly charts
    - Entry/exit markers on price chart
    - Equity curve with drawdown
    - Trade distribution analysis
    - Performance dashboards
    
    Example:
        >>> viz = BacktestVisualizer(backtest_result, price_data)
        >>> fig = viz.create_price_chart()
        >>> fig.write_html("price_chart.html")
    """
    
    def __init__(self, result: BacktestResult, price_data: pd.DataFrame):
        """
        Initialize visualizer.
        
        Args:
            result: BacktestResult from backtesting
            price_data: OHLCV DataFrame
        """
        self.result = result
        self.price_data = price_data
        self.trades = result.trades
    
    def create_price_chart(self) -> go.Figure:
        """
        Create price chart with entry/exit markers.
        
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=self.price_data.index,
            y=self.price_data['close'],
            mode='lines',
            name='Price',
            line=dict(color='#2E86AB', width=2),
            hovertemplate='<b>Date</b>: %{x}<br><b>Price</b>: $%{y:.2f}<extra></extra>'
        ))
        
        # Add entry markers
        long_entries = [t for t in self.trades if t.direction == 'long']
        short_entries = [t for t in self.trades if t.direction == 'short']
        
        if long_entries:
            fig.add_trace(go.Scatter(
                x=[t.entry_date for t in long_entries],
                y=[t.entry_price for t in long_entries],
                mode='markers',
                name='Long Entry',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color='#06D6A0',
                    line=dict(color='white', width=1)
                ),
                hovertemplate='<b>Long Entry</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ))
        
        if short_entries:
            fig.add_trace(go.Scatter(
                x=[t.entry_date for t in short_entries],
                y=[t.entry_price for t in short_entries],
                mode='markers',
                name='Short Entry',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color='#EF476F',
                    line=dict(color='white', width=1)
                ),
                hovertemplate='<b>Short Entry</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ))
        
        # Add exit markers
        exits = [t for t in self.trades if t.exit_date is not None]
        if exits:
            exit_colors = ['#06D6A0' if t.is_winner() else '#EF476F' for t in exits]
            fig.add_trace(go.Scatter(
                x=[t.exit_date for t in exits],
                y=[t.exit_price for t in exits],
                mode='markers',
                name='Exit',
                marker=dict(
                    symbol='x',
                    size=10,
                    color=exit_colors,
                    line=dict(color='white', width=1)
                ),
                text=[f"P&L: ${t.pnl:.2f}" for t in exits],
                hovertemplate='<b>Exit</b><br>Date: %{x}<br>Price: $%{y:.2f}<br>%{text}<extra></extra>'
            ))
        
        # Layout
        fig.update_layout(
            title='Price Chart with Trade Markers',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            template='plotly_white',
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def create_equity_curve(self) -> go.Figure:
        """
        Create equity curve with drawdown.
        
        Returns:
            Plotly Figure with equity and drawdown
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=('Portfolio Equity', 'Drawdown')
        )
        
        # Equity curve
        equity = self.result.equity_curve
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity.values,
                mode='lines',
                name='Equity',
                line=dict(color='#2E86AB', width=2),
                fill='tozeroy',
                fillcolor='rgba(46, 134, 171, 0.1)',
                hovertemplate='<b>Date</b>: %{x}<br><b>Equity</b>: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add initial capital line
        initial_capital = self.result.config['initial_capital']
        fig.add_hline(
            y=initial_capital,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Initial: ${initial_capital:,.0f}",
            row=1, col=1
        )
        
        # Calculate drawdown
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values * 100,
                mode='lines',
                name='Drawdown',
                line=dict(color='#EF476F', width=2),
                fill='tozeroy',
                fillcolor='rgba(239, 71, 111, 0.2)',
                hovertemplate='<b>Date</b>: %{x}<br><b>Drawdown</b>: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Layout
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        fig.update_layout(
            height=700,
            template='plotly_white',
            showlegend=False,
            hovermode='x unified'
        )
        
        return fig
    
    def create_trade_distribution(self) -> go.Figure:
        """
        Create trade P&L distribution chart.
        
        Returns:
            Plotly Figure
        """
        if not self.trades:
            return go.Figure()
        
        pnls = [t.pnl for t in self.trades]
        
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=pnls,
            nbinsx=30,
            name='Trade P&L',
            marker=dict(
                color=pnls,
                colorscale=[[0, '#EF476F'], [0.5, '#FFD166'], [1, '#06D6A0']],
                line=dict(color='white', width=1)
            ),
            hovertemplate='<b>P&L Range</b>: $%{x}<br><b>Count</b>: %{y}<extra></extra>'
        ))
        
        # Add vertical line at zero
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        # Layout
        fig.update_layout(
            title='Trade P&L Distribution',
            xaxis_title='Profit/Loss ($)',
            yaxis_title='Number of Trades',
            template='plotly_white',
            height=500,
            showlegend=False
        )
        
        return fig
    
    def create_performance_dashboard(self) -> go.Figure:
        """
        Create comprehensive performance dashboard.
        
        Returns:
            Plotly Figure with multiple subplots
        """
        metrics = self.result.metrics
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Win Rate',
                'Profit Factor',
                'Sharpe Ratio',
                'Max Drawdown'
            ),
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
                   [{'type': 'indicator'}, {'type': 'indicator'}]]
        )
        
        # Win Rate
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=metrics.get('win_rate', 0) * 100,
                title={'text': "Win Rate (%)"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#06D6A0"},
                    'steps': [
                        {'range': [0, 40], 'color': "#FFD166"},
                        {'range': [40, 60], 'color': "#2E86AB"},
                        {'range': [60, 100], 'color': "#06D6A0"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ),
            row=1, col=1
        )
        
        # Profit Factor
        pf = min(metrics.get('profit_factor', 0), 5)  # Cap at 5 for display
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=pf,
                title={'text': "Profit Factor"},
                gauge={
                    'axis': {'range': [0, 5]},
                    'bar': {'color': "#2E86AB"},
                    'steps': [
                        {'range': [0, 1], 'color': "#EF476F"},
                        {'range': [1, 2], 'color': "#FFD166"},
                        {'range': [2, 5], 'color': "#06D6A0"}
                    ],
                    'threshold': {
                        'line': {'color': "green", 'width': 4},
                        'thickness': 0.75,
                        'value': 1
                    }
                }
            ),
            row=1, col=2
        )
        
        # Sharpe Ratio
        sharpe = metrics.get('sharpe_ratio', 0)
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=sharpe,
                title={'text': "Sharpe Ratio"},
                delta={'reference': 1},
                number={'font': {'size': 60}},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=2, col=1
        )
        
        # Max Drawdown
        max_dd = abs(metrics.get('max_drawdown', 0)) * 100
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=max_dd,
                title={'text': "Max Drawdown (%)"},
                delta={'reference': 0, 'decreasing': {'color': "#06D6A0"}},
                number={'font': {'size': 60, 'color': '#EF476F'}},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def create_monthly_returns_heatmap(self) -> go.Figure:
        """
        Create monthly returns heatmap.
        
        Returns:
            Plotly Figure
        """
        equity = self.result.equity_curve
        
        # Calculate monthly returns
        monthly = equity.resample('ME').last()
        monthly_returns = monthly.pct_change().dropna()
        
        # Pivot to year x month
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values * 100
        })
        
        pivot = df.pivot(index='year', columns='month', values='return')
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            y=pivot.index,
            colorscale='RdYlGn',
            zmid=0,
            text=pivot.values,
            texttemplate='%{text:.1f}%',
            textfont={"size": 10},
            hovertemplate='<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title='Monthly Returns Heatmap (%)',
            xaxis_title='Month',
            yaxis_title='Year',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def save_all_charts(self, output_dir: str) -> None:
        """
        Save all charts to HTML files.
        
        Args:
            output_dir: Directory to save charts
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating charts in {output_dir}...")
        
        # Price chart
        fig = self.create_price_chart()
        fig.write_html(output_path / "price_chart.html")
        print("  price_chart.html")
        
        # Equity curve
        fig = self.create_equity_curve()
        fig.write_html(output_path / "equity_curve.html")
        print("  equity_curve.html")
        
        # Trade distribution
        fig = self.create_trade_distribution()
        fig.write_html(output_path / "trade_distribution.html")
        print("  trade_distribution.html")
        
        # Performance dashboard
        fig = self.create_performance_dashboard()
        fig.write_html(output_path / "performance_dashboard.html")
        print("  performance_dashboard.html")
        
        # Monthly returns
        fig = self.create_monthly_returns_heatmap()
        fig.write_html(output_path / "monthly_returns.html")
        print("  monthly_returns.html")
        
        print(f"All charts saved to {output_dir}")

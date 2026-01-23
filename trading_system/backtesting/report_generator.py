"""
Professional report generator for backtest results.

Creates trade floor ready reports including:
- Executive summary
- Detailed trade logs
- HTML dashboards
- Performance reports
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from trading_system.backtesting.backtest_engine import BacktestResult
from trading_system.backtesting.trade_analyzer import TradeAnalyzer
from trading_system.backtesting.visualizer import BacktestVisualizer


class ReportGenerator:
    """
    Generate professional reports for backtest results.
    
    Creates:
    - Executive summary (HTML)
    - Detailed trade log (CSV/Excel)
    - Interactive dashboard (HTML)
    - Performance report (HTML)
    
    Example:
        >>> generator = ReportGenerator(result, price_data)
        >>> generator.generate_all_reports("reports/aapl_backtest")
    """
    
    def __init__(self, result: BacktestResult, price_data: pd.DataFrame):
        """
        Initialize report generator.
        
        Args:
            result: BacktestResult from backtesting
            price_data: OHLCV DataFrame
        """
        self.result = result
        self.price_data = price_data
        self.analyzer = TradeAnalyzer(result)
        self.visualizer = BacktestVisualizer(result, price_data)
    
    def generate_executive_summary(self, output_path: str) -> None:
        """
        Generate executive summary HTML report.
        
        Args:
            output_path: Path to save HTML file
        """
        metrics = self.analyzer.analyze()
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Executive Summary</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 32px;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-label {{
            color: #666;
            font-size: 14px;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            color: #333;
        }}
        .metric-value.positive {{
            color: #06D6A0;
        }}
        .metric-value.negative {{
            color: #EF476F;
        }}
        .section {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #333;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .recommendation {{
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
            font-size: 16px;
        }}
        .recommendation.deploy {{
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            color: #155724;
        }}
        .recommendation.review {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            color: #856404;
        }}
        .recommendation.reject {{
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
            color: #721c24;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Backtest Executive Summary</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Model: {self.result.config.get('model_path', 'N/A')}</p>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Total Return</div>
            <div class="metric-value {'positive' if metrics.get('total_pnl', 0) > 0 else 'negative'}">
                {metrics.get('total_pnl', 0):+,.2f}$
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value">
                {metrics.get('win_rate', 0):.1%}
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value {'positive' if self.result.metrics.get('sharpe_ratio', 0) > 0 else 'negative'}">
                {self.result.metrics.get('sharpe_ratio', 0):.2f}
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value negative">
                {self.result.metrics.get('max_drawdown', 0):.2%}
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Profit Factor</div>
            <div class="metric-value {'positive' if metrics.get('profit_factor', 0) > 1 else 'negative'}">
                {min(metrics.get('profit_factor', 0), 99.9):.2f}
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Total Trades</div>
            <div class="metric-value">
                {metrics.get('total_trades', 0)}
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Performance Summary</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Initial Capital</td>
                <td>${self.result.config.get('initial_capital', 0):,.2f}</td>
            </tr>
            <tr>
                <td>Final Equity</td>
                <td>${self.result.metrics.get('final_equity', 0):,.2f}</td>
            </tr>
            <tr>
                <td>Total Return</td>
                <td>{self.result.metrics.get('total_return', 0):.2%}</td>
            </tr>
            <tr>
                <td>Winning Trades</td>
                <td>{metrics.get('winning_trades', 0)} ({metrics.get('win_rate', 0):.1%})</td>
            </tr>
            <tr>
                <td>Losing Trades</td>
                <td>{metrics.get('losing_trades', 0)}</td>
            </tr>
            <tr>
                <td>Average Win</td>
                <td>${metrics.get('avg_win', 0):,.2f}</td>
            </tr>
            <tr>
                <td>Average Loss</td>
                <td>${metrics.get('avg_loss', 0):,.2f}</td>
            </tr>
            <tr>
                <td>Expectancy</td>
                <td>${metrics.get('expectancy', 0):,.2f}</td>
            </tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Top 5 Best Trades</h2>
        <table>
            <tr>
                <th>Date</th>
                <th>Direction</th>
                <th>Entry</th>
                <th>Exit</th>
                <th>P&L</th>
            </tr>
            {''.join(self._format_trade_row(t) for t in self.analyzer.get_best_trades(5))}
        </table>
    </div>
    
    <div class="section">
        <h2>⚠️ Top 5 Worst Trades</h2>
        <table>
            <tr>
                <th>Date</th>
                <th>Direction</th>
                <th>Entry</th>
                <th>Exit</th>
                <th>P&L</th>
            </tr>
            {''.join(self._format_trade_row(t) for t in self.analyzer.get_worst_trades(5))}
        </table>
    </div>
    
    {self._generate_recommendation(metrics)}
    
</body>
</html>
"""
        
        Path(output_path).write_text(html, encoding="utf-8", errors="replace")


        print(f"Executive summary saved to {output_path}")
    
    def _format_trade_row(self, trade) -> str:
        """Format a trade as HTML table row."""
        return f"""
            <tr>
                <td>{trade.entry_date.strftime('%Y-%m-%d')}</td>
                <td>{trade.direction.upper()}</td>
                <td>${trade.entry_price:.2f}</td>
                <td>${trade.exit_price:.2f}</td>
                <td style="color: {'#06D6A0' if trade.pnl > 0 else '#EF476F'}; font-weight: bold;">
                    ${trade.pnl:,.2f}
                </td>
            </tr>
        """
    
    def _generate_recommendation(self, metrics: dict) -> str:
        """Generate deployment recommendation."""
        win_rate = metrics.get('win_rate', 0)
        profit_factor = metrics.get('profit_factor', 0)
        sharpe = self.result.metrics.get('sharpe_ratio', 0)
        max_dd = abs(self.result.metrics.get('max_drawdown', 0))
        
        # Decision logic
        if (win_rate > 0.5 and profit_factor > 1.5 and sharpe > 1.0 and max_dd < 0.20):
            rec_class = "deploy"
            rec_text = "DEPLOY - Model shows strong performance across all metrics. Recommend paper trading for 2 weeks before live deployment."
        elif (win_rate > 0.45 and profit_factor > 1.0 and sharpe > 0.5):
            rec_class = "review"
            rec_text = "⚠️ <strong>REVIEW</strong> - Model shows promise but needs further validation. Consider extended backtesting or parameter tuning."
        else:
            rec_class = "reject"
            rec_text = "❌ <strong>REJECT</strong> - Model does not meet minimum performance criteria. Recommend retraining with different parameters or features."
        
        return f"""
    <div class="recommendation {rec_class}">
        <h3>Recommendation</h3>
        <p>{rec_text}</p>
    </div>
"""
    
    def generate_trade_log(self, output_path: str) -> None:
        """
        Generate detailed trade log CSV.
        
        Args:
            output_path: Path to save CSV file
        """
        df = self.result.to_dataframe()
        df.to_csv(output_path, index=False)
        print(f"Trade log saved to {output_path}")
    
    def generate_all_reports(self, output_dir: str) -> None:
        """
        Generate all reports and charts.
        
        Args:
            output_dir: Directory to save all outputs
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"GENERATING REPORTS")
        print(f"{'='*70}")
        print(f"Output directory: {output_dir}\n")
        
        # Executive summary
        self.generate_executive_summary(str(output_path / "executive_summary.html"))
        
        # Trade log
        self.generate_trade_log(str(output_path / "trades.csv"))
        
        # Charts
        charts_dir = output_path / "charts"
        self.visualizer.save_all_charts(str(charts_dir))
        
        print(f"\n{'='*70}")
        print(f"ALL REPORTS GENERATED")
        print(f"{'='*70}")
        print(f"\nOpen these files to view results:")
        print(f"  {output_path / 'executive_summary.html'}")
        print(f"  {charts_dir / 'price_chart.html'}")
        print(f"  {charts_dir / 'equity_curve.html'}")
        print(f"  {output_path / 'trades.csv'}")
        print(f"{'='*70}\n")

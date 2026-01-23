# Backtesting System - Quick Start Guide

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install plotly kaleido
```

### 2. Run Backtest

```bash
python backtest_model.py \
    --model stock_trading_logs/models/AAPL_production_20260123_043341_final \
    --ticker AAPL \
    --start-date 2024-11-01 \
    --end-date 2026-01-22
```

### 3. View Results

The backtest will generate:
- `reports/<ticker>_backtest_<timestamp>/executive_summary.html` - Open in browser
- `reports/<ticker>_backtest_<timestamp>/charts/` - Interactive charts
- `reports/<ticker>_backtest_<timestamp>/trades.csv` - Trade log

## 📊 What You'll Get

### Executive Summary
Professional HTML report with:
- ✅ Key performance metrics (Win Rate, Sharpe, Drawdown)
- 📈 Performance summary table
- 🎯 Top 5 best/worst trades
- 💡 Deployment recommendation

### Interactive Charts
1. **Price Chart** - Entry/exit markers with P&L labels
2. **Equity Curve** - Portfolio value over time with drawdown
3. **Trade Distribution** - P&L histogram
4. **Performance Dashboard** - Gauges for key metrics
5. **Monthly Returns** - Heatmap of monthly performance

### Trade Log (CSV)
Complete record of all trades:
- Entry/exit dates and prices
- Direction (long/short)
- P&L and costs
- Holding period
- MAE/MFE (risk metrics)

## 📋 Command Options

```bash
python backtest_model.py --help
```

**Key Options:**
- `--model` - Path to trained model (required)
- `--ticker` - Stock ticker (required)
- `--start-date` - Start date (YYYY-MM-DD)
- `--end-date` - End date (YYYY-MM-DD)
- `--initial-capital` - Starting capital (default: $100,000)
- `--transaction-cost` - Cost rate (default: 0.001 = 10 bps)
- `--output-dir` - Custom output directory
- `--no-charts` - Skip chart generation (faster)

## 🎯 Example Use Cases

### Validation Period Backtest
```bash
python backtest_model.py \
    --model stock_trading_logs/models/AAPL_production_20260123_043341_final \
    --ticker AAPL \
    --start-date 2024-11-01 \
    --end-date 2026-01-22 \
    --output-dir reports/aapl_validation
```

### Full Historical Backtest
```bash
python backtest_model.py \
    --model stock_trading_logs/models/AAPL_production_20260123_043341_final \
    --ticker AAPL \
    --output-dir reports/aapl_full_history
```

### High Capital Backtest
```bash
python backtest_model.py \
    --model stock_trading_logs/models/AAPL_production_20260123_043341_final \
    --ticker AAPL \
    --initial-capital 1000000 \
    --output-dir reports/aapl_1M_capital
```

### Quick Test (No Charts)
```bash
python backtest_model.py \
    --model stock_trading_logs/models/AAPL_production_20260123_043341_final \
    --ticker AAPL \
    --no-charts
```

## 📈 Understanding the Results

### Key Metrics

**Win Rate**: Percentage of profitable trades
- ✅ Good: > 50%
- ⚠️ Review: 40-50%
- ❌ Poor: < 40%

**Profit Factor**: Total wins / Total losses
- ✅ Good: > 1.5
- ⚠️ Review: 1.0-1.5
- ❌ Poor: < 1.0

**Sharpe Ratio**: Risk-adjusted returns
- ✅ Good: > 1.0
- ⚠️ Review: 0.5-1.0
- ❌ Poor: < 0.5

**Max Drawdown**: Largest peak-to-trough decline
- ✅ Good: < 15%
- ⚠️ Review: 15-25%
- ❌ Poor: > 25%

### Trade Quality Metrics

**Expectancy**: Average $ per trade
- Positive = profitable system
- Higher is better

**MAE (Maximum Adverse Excursion)**: Worst drawdown during trade
- Shows risk per trade
- Lower is better

**MFE (Maximum Favorable Excursion)**: Best profit during trade
- Shows profit potential
- Higher is better

**Efficiency**: Captured profit / MFE
- How much of potential profit was captured
- Higher is better (> 0.5 is good)

## 🎨 Chart Interpretation

### Price Chart
- **Green triangles** ▲ = Long entries
- **Red triangles** ▼ = Short entries
- **X markers** = Exits (green=win, red=loss)
- Hover for details

### Equity Curve
- **Blue line** = Portfolio value
- **Red area** = Drawdown periods
- **Gray dashed line** = Initial capital

### Performance Dashboard
- **Gauges** show metric ranges
- **Green zones** = good performance
- **Red zones** = poor performance

## 🚨 Deployment Decision

The executive summary includes a recommendation:

**✅ DEPLOY** - Strong performance, ready for paper trading
- Win rate > 50%
- Profit factor > 1.5
- Sharpe > 1.0
- Max DD < 20%

**⚠️ REVIEW** - Promising but needs validation
- Meets some criteria
- Consider extended testing

**❌ REJECT** - Does not meet criteria
- Recommend retraining

## 📁 Output Structure

```
reports/AAPL_backtest_20260123_051200/
├── executive_summary.html    # Main report (open this first)
├── trades.csv                 # Complete trade log
└── charts/
    ├── price_chart.html       # Interactive price chart
    ├── equity_curve.html      # Equity + drawdown
    ├── trade_distribution.html # P&L histogram
    ├── performance_dashboard.html # Metrics gauges
    └── monthly_returns.html   # Monthly heatmap
```

## 💡 Tips

1. **Always backtest on validation data first** (data not used in training)
2. **Compare to buy-and-hold** - Is the model better than just holding?
3. **Check trade frequency** - Too many trades = high costs
4. **Review worst trades** - Understand failure modes
5. **Monitor drawdown periods** - How long to recover?
6. **Validate assumptions** - Are transaction costs realistic?

## 🔧 Troubleshooting

**Error: "No data file found"**
- Check ticker name matches file in `yf_data/`
- Ensure data file exists

**Error: "Insufficient data"**
- Date range too short
- Need at least 150 days for default settings

**Charts not displaying**
- Install plotly: `pip install plotly kaleido`
- Check browser allows local HTML files

**Model not loading**
- Verify model path is correct
- Don't include `.zip` extension
- Check VecNormalize file exists

## 📞 Next Steps

After backtesting:

1. **Review Results** - Check all metrics and charts
2. **Analyze Trades** - Understand entry/exit logic
3. **Compare Periods** - Train vs validation performance
4. **Paper Trade** - Deploy to paper account for 2 weeks
5. **Monitor Live** - Track real-time performance
6. **Iterate** - Retrain if needed

---

**Ready to backtest?** Run the command above and open the executive summary HTML! 🚀

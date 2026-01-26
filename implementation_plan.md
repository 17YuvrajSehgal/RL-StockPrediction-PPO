# Daily Retraining & Walk-Forward Prediction System

## Problem Statement

Build a production-grade system that:
1. **Retrains PPO models daily** with the latest market data
2. **Generates trading signals** (BUY/SELL/HOLD) for the next trading day
3. **Validates the strategy** using walk-forward testing (train on day N, test on day N+1)

## User Review Required

> [!IMPORTANT]
> **Key Design Decisions Requiring Approval**
> 
> Before implementation, please confirm these strategic choices:
> 
> 1. **Continuous Learning vs. Full Retrain**: Should we fine-tune the existing model daily (faster, ~5 min/day) or train from scratch (slower, ~30-60 min/day but more robust)?
> 
> 2. **Signal Thresholds**: The model outputs a weight ∈ [-1, 1]. Proposed thresholds:
>    - `weight > 0.3` → **BUY** (go long 30%+ of portfolio)
>    - `weight < -0.3` → **SELL** (go short or exit long)
>    - `-0.3 ≤ weight ≤ 0.3` → **HOLD** (maintain current position)
> 
> 3. **Walk-Forward Window**: How much historical data for training?
>    - Proposed: 2 years rolling window (keeps model adaptive to regime changes)
>    - Alternative: Expanding window (all available history, may overfit to old patterns)

---

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DAILY TRADING SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐               │
│  │   Data Sync   │───▶│  Daily Train  │───▶│Signal Generate│               │
│  │  (fetch_yf)   │    │  (fine-tune)  │    │  (predict)    │               │
│  └───────────────┘    └───────────────┘    └───────────────┘               │
│         │                    │                    │                        │
│         ▼                    ▼                    ▼                        │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                      Signal Output                              │       │
│  │  {ticker: "AAPL", date: "2026-01-27", signal: "BUY",           │       │
│  │   confidence: 0.72, target_weight: 0.65, reason: "..."}        │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                      WALK-FORWARD VALIDATION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Day 1     Day 2     Day 3     Day 4     Day 5                             │
│  ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐                           │
│  │Train│──▶│Test │   │Train│──▶│Test │   │Train│──▶...                     │
│  │T1-D1│   │D2   │   │T1-D2│   │D3   │   │T1-D3│                           │
│  └─────┘   └─────┘   └─────┘   └─────┘   └─────┘                           │
│                                                                             │
│  Aggregates: Accuracy, Sharpe, Max DD, Signal Distribution                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Proposed Changes

### New Files to Create

---

#### [NEW] [daily_trainer.py](file:///c:/workplace/RL-StockPrediction-PPO/daily_trainer.py)

Core daily retraining module:

```python
class DailyTrainer:
    """Handles incremental/daily model training."""
    
    def train_for_date(self, ticker: str, as_of_date: date, 
                       base_model_path: str = None) -> TrainResult:
        """Train model using data up to as_of_date."""
        
    def fine_tune(self, ticker: str, as_of_date: date,
                  base_model_path: str) -> TrainResult:
        """Quick fine-tune of existing model with recent data."""
```

Key features:
- Loads data up to specified date (no future data leakage)
- Supports both full training and fine-tuning
- Saves models with date-versioned paths
- Configurable training window (rolling vs expanding)

---

#### [NEW] [signal_generator.py](file:///c:/workplace/RL-StockPrediction-PPO/signal_generator.py)

Generates actionable trading signals:

```python
@dataclass
class TradingSignal:
    ticker: str
    signal_date: date        # Date signal is FOR (next trading day)
    generated_at: datetime   # When signal was generated
    signal: Literal["BUY", "SELL", "HOLD"]
    target_weight: float     # Model's raw output [-1, 1]
    confidence: float        # Derived from weight magnitude
    current_price: float     # Price at signal generation
    reasoning: str           # Human-readable explanation

class SignalGenerator:
    """Generate trading signals from trained model."""
    
    def generate_signal(self, ticker: str, model_path: str,
                        for_date: date) -> TradingSignal:
        """Generate signal for the next trading day."""
```

Key features:
- Clear signal interpretation (BUY/SELL/HOLD)
- Confidence scoring based on weight magnitude
- Human-readable reasoning for each signal
- JSON export for integration with trading systems

---

#### [NEW] [walk_forward_validator.py](file:///c:/workplace/RL-StockPrediction-PPO/walk_forward_validator.py)

Rigorous strategy validation:

```python
@dataclass  
class WalkForwardResult:
    start_date: date
    end_date: date
    total_signals: int
    correct_signals: int     # Signal direction matched actual return
    accuracy: float
    cumulative_return: float
    sharpe_ratio: float
    max_drawdown: float
    signal_distribution: Dict[str, int]  # {"BUY": 45, "SELL": 30, "HOLD": 25}
    daily_results: List[DayResult]

class WalkForwardValidator:
    """Walk-forward testing framework."""
    
    def validate(self, ticker: str, start_date: date, end_date: date,
                 training_window_days: int = 504,  # 2 years
                 timesteps_per_day: int = 50000) -> WalkForwardResult:
        """Run walk-forward validation over date range."""
```

Key features:
- Trains model up to day N, tests on day N+1
- Tracks signal accuracy (did the signal match actual market direction?)
- Computes cumulative performance metrics
- Generates detailed reports with per-day breakdowns
- Configurable training window and timesteps

---

#### [NEW] [daily_pipeline.py](file:///c:/workplace/RL-StockPrediction-PPO/daily_pipeline.py)

Orchestrates the full daily workflow:

```python
class DailyPipeline:
    """End-to-end daily trading signal pipeline."""
    
    def run_daily(self, ticker: str, 
                  target_date: date = None) -> PipelineResult:
        """
        Full daily pipeline:
        1. Sync latest market data
        2. Retrain/fine-tune model
        3. Generate signal for next trading day
        4. Save results and logs
        """
```

Key features:
- Automatic data sync from Yahoo Finance
- Handles weekends/holidays (finds next trading day)
- Model versioning with date stamps
- Comprehensive logging and audit trail

---

### Directory Structure After Implementation

```
RL-StockPrediction-PPO/
├── daily_trainer.py           # [NEW] Daily retraining logic
├── signal_generator.py        # [NEW] Signal generation  
├── walk_forward_validator.py  # [NEW] Walk-forward testing
├── daily_pipeline.py          # [NEW] End-to-end pipeline
│
├── daily_models/              # [NEW] Date-versioned models
│   └── AAPL/
│       ├── 2026-01-24/
│       │   ├── model.zip
│       │   └── vecnormalize.pkl
│       └── ...
│
├── signals/                   # [NEW] Generated signals
│   └── AAPL/
│       ├── 2026-01-27.json   # Signal for Monday
│       └── ...
│
├── walk_forward_results/      # [NEW] Validation results
│   └── AAPL_2025-01-01_to_2026-01-24.json
│
└── (existing files...)
```

---

## Verification Plan

### Automated Tests

1. **Unit Tests**: Test each component in isolation
   ```bash
   python -m pytest tests/test_daily_trainer.py
   python -m pytest tests/test_signal_generator.py
   python -m pytest tests/test_walk_forward.py
   ```

2. **Integration Test**: Run mini walk-forward on 1-week period
   ```bash
   python walk_forward_validator.py --ticker AAPL \
       --start 2025-12-01 --end 2025-12-07 \
       --timesteps 10000  # Fast test
   ```

3. **End-to-End Test**: Generate signal for Monday
   ```bash
   python daily_pipeline.py --ticker AAPL --target-date 2026-01-27
   ```

### Manual Verification

1. Review generated signals for reasonableness
2. Compare walk-forward results against buy-and-hold baseline
3. Validate no future data leakage in training pipeline

---

## Alternative Strategies Considered

### Strategy 1: Ensemble of Multiple Timeframes (More Robust)

Instead of a single model, train 3 models with different lookback windows:
- Short-term: 6-month training window (captures recent regime)
- Medium-term: 2-year training window (balanced)
- Long-term: 5-year training window (historical patterns)

Final signal = weighted vote of all 3 models.

**Pros**: More robust to regime changes, less sensitive to hyperparameters
**Cons**: 3x training time, more complexity

### Strategy 2: Bayesian Fine-Tuning (Faster Daily Updates)

Instead of retraining from scratch:
1. Train a base model on 5 years of data (one-time)
2. Daily: Fine-tune for 5,000 steps on last 60 days of data

**Pros**: Much faster daily updates (~5 min vs 30+ min)
**Cons**: May drift over time, needs periodic full retrain

### Strategy 3: Meta-Learning for Rapid Adaptation

Use MAML-style meta-learning to train a model that can quickly adapt to new market conditions with minimal data.

**Pros**: Theoretically superior adaptation
**Cons**: Significantly more complex, research-grade implementation

---

## Recommended Approach

I recommend **Strategy 2 (Bayesian Fine-Tuning)** for practical deployment:

1. **Initial Setup**: Train base model on full historical data (2020-2026)
2. **Daily Updates**: Fine-tune for ~50,000 steps on rolling 90-day window
3. **Weekly Full Retrain**: Every Sunday, do a full retrain to prevent drift

This gives us:
- Fast daily updates (~10-15 min on GPU)
- Regular full retrains to stay calibrated
- Good balance of adaptability and stability

---

## Questions for You

1. **Training preference**: Fine-tune daily (fast) or full retrain (thorough)?

2. **Signal thresholds**: Are the proposed ±0.3 thresholds reasonable, or would you prefer tighter/looser bounds?

3. **Ensemble approach**: Should I implement the multi-timeframe ensemble (Strategy 1) as an optional enhancement?

4. **Walk-forward period**: How many days of walk-forward testing would you like to run initially? (e.g., 30 days, 90 days, full year?)

5. **Production deployment**: Do you want this to run automatically on a schedule, or manually triggered?

#!/bin/bash
#SBATCH --job-name=ppo_tsla_trading       # Job name
#SBATCH --account=def-naser2
#SBATCH --time=02:00:00                    # Time limit (2 hours for full training)
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks=1                         # Number of tasks
#SBATCH --cpus-per-task=8                  # CPU cores per task
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/yuvraj17/stock_trading_logs/%x-%j.out    # Standard output
#SBATCH --error=/scratch/yuvraj17/stock_trading_logs/%x-%j.err     # Standard error
#SBATCH --mail-type=BEGIN,END,FAIL         # Email notifications
#SBATCH --mail-user=ys19rk@brocku.ca       # Your email

################################################################################
# TSLA Stock Trading - PPO Training Job Script
# 
# Training Configuration:
# - Ticker: TSLA (Tesla Inc.)
# - Data Period: 2021-01-23 to 2026-01-23 (5 years)
# - Training Split: 80% (2021-01-23 to ~2024-11)
# - Validation Split: 20% (~2024-11 to 2026-01-23)
# - Backtest Period: Recent data (2025-01-01 to 2026-01-23)
#
# Usage:
#   sbatch slurm_train_tsla.slurm
#
# Monitor:
#   squeue -u $USER
#   tail -f /scratch/yuvraj17/stock_trading_logs/ppo_tsla_trading-*.out
################################################################################

# Print job information
echo "==========================================="
echo "TSLA PPO Stock Trading Training"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "==========================================="

# Load required modules
echo "Loading modules..."
module load StdEnv/2023
module load python/3.10
module load cuda/12.2

# Print GPU information
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Set up environment
echo "Setting up environment..."
export PROJECT_DIR=/scratch/yuvraj17/RL-StockPrediction-PPO
export LOG_DIR=/scratch/yuvraj17/stock_trading_logs
export MODEL_DIR=/scratch/yuvraj17/stock_trading_logs/models
export TENSORBOARD_DIR=/scratch/yuvraj17/stock_trading_logs/tensorboard
export REPORT_DIR=/scratch/yuvraj17/stock_trading_logs/reports

# Create necessary directories
mkdir -p $LOG_DIR
mkdir -p $MODEL_DIR
mkdir -p $TENSORBOARD_DIR
mkdir -p $REPORT_DIR

# Navigate to project directory
cd $PROJECT_DIR || exit 1

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Verify installations
echo ""
echo "Python version:"
python --version
echo ""
echo "PyTorch version and CUDA availability:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU Count: {torch.cuda.device_count()}')"
echo ""

################################################################################
# Training Configuration
################################################################################

# Stock ticker to trade
TICKER="TSLA"

# Training mode: "production" (conservative) or "research" (aggressive)
MODE="production"

# Total training timesteps (2M for production)
TIMESTEPS=2000000

# Experiment name (will be used for model and log naming)
EXPERIMENT="${TICKER}_${MODE}_$(date +%Y%m%d_%H%M%S)"

# GPU ID (0 for first GPU)
GPU_ID=0

# Random seed for reproducibility
SEED=42

################################################################################
# Run Training
################################################################################

echo "==========================================="
echo "Starting PPO Training for TSLA"
echo "==========================================="
echo "Ticker: $TICKER"
echo "Mode: $MODE"
echo "Timesteps: $TIMESTEPS"
echo "Experiment: $EXPERIMENT"
echo "GPU: $GPU_ID"
echo "Data: yf_data/TSLA_1d_2021-01-23_to_2026-01-23.csv"
echo "==========================================="
echo ""

# Run training with all outputs logged
python -u train_production_ppo.py \
    --ticker $TICKER \
    --mode $MODE \
    --timesteps $TIMESTEPS \
    --gpu $GPU_ID \
    --seed $SEED \
    --experiment $EXPERIMENT \
    2>&1 | tee $LOG_DIR/${EXPERIMENT}_training.log

# Capture exit code
TRAIN_EXIT_CODE=$?

echo ""
echo "==========================================="
echo "Training Completed with Exit Code: $TRAIN_EXIT_CODE"
echo "==========================================="

################################################################################
# Run Backtest on Recent Data (2025-01-01 to 2026-01-23)
################################################################################

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "==========================================="
    echo "Running Backtest on Recent Data"
    echo "==========================================="
    echo "Period: 2025-01-01 to 2026-01-23"
    echo ""
    
    # Find the trained model
    MODEL_PATH=$(ls -t $MODEL_DIR/${TICKER}_${MODE}_*_final.zip 2>/dev/null | head -1 | sed 's/.zip$//')
    
    if [ -z "$MODEL_PATH" ]; then
        # Model might be in local models/ directory
        MODEL_PATH=$(ls -t models/${TICKER}_${MODE}_*_final.zip 2>/dev/null | head -1 | sed 's/.zip$//')
    fi
    
    if [ -n "$MODEL_PATH" ]; then
        echo "Using model: $MODEL_PATH"
        
        # Run backtest on recent 1 year of data
        python -u backtest_model.py \
            --model "$MODEL_PATH" \
            --ticker $TICKER \
            --start-date 2025-01-01 \
            --end-date 2026-01-23 \
            --output-dir $REPORT_DIR/${EXPERIMENT}_backtest \
            2>&1 | tee $LOG_DIR/${EXPERIMENT}_backtest.log
        
        BACKTEST_EXIT_CODE=$?
        echo "Backtest completed with exit code: $BACKTEST_EXIT_CODE"
    else
        echo "WARNING: Could not find trained model for backtest"
        BACKTEST_EXIT_CODE=1
    fi
else
    echo "Skipping backtest due to training failure"
    BACKTEST_EXIT_CODE=1
fi

################################################################################
# Post-Training Tasks
################################################################################

echo ""
echo "==========================================="
echo "Post-Training Tasks"
echo "==========================================="

# Copy models to scratch for persistence
if [ -d "models" ]; then
    echo "Copying models to $MODEL_DIR..."
    cp -r models/* $MODEL_DIR/
    echo "Models saved to: $MODEL_DIR"
fi

# Copy TensorBoard logs to scratch
if [ -d "runs" ]; then
    echo "Copying TensorBoard logs to $TENSORBOARD_DIR..."
    cp -r runs/* $TENSORBOARD_DIR/
    echo "TensorBoard logs saved to: $TENSORBOARD_DIR"
fi

# Copy backtest reports
if [ -d "reports" ]; then
    echo "Copying reports to $REPORT_DIR..."
    cp -r reports/* $REPORT_DIR/
    echo "Reports saved to: $REPORT_DIR"
fi

# Print summary
echo ""
echo "==========================================="
echo "Job Summary"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Ticker: $TICKER"
echo "Experiment: $EXPERIMENT"
echo "Training Exit Code: $TRAIN_EXIT_CODE"
echo "Backtest Exit Code: $BACKTEST_EXIT_CODE"
echo ""
echo "Output Locations:"
echo "  Models:      $MODEL_DIR"
echo "  TensorBoard: $TENSORBOARD_DIR"
echo "  Reports:     $REPORT_DIR"
echo "  Logs:        $LOG_DIR"
echo ""
echo "To view TensorBoard:"
echo "  tensorboard --logdir $TENSORBOARD_DIR --host 0.0.0.0 --port 6006"
echo ""
echo "End Time: $(date)"
echo "==========================================="

# Print resource usage
echo ""
echo "Resource Usage:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,AllocCPUS,State,ExitCode,Elapsed,MaxRSS,MaxVMSize

# Exit with training exit code
exit $TRAIN_EXIT_CODE

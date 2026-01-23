#!/bin/bash
#SBATCH --job-name=ppo_stock_trading      # Job name
#SBATCH --account=def-naser2             # Replace with your account
#SBATCH --time=00:30:00                    # Time limit (24 hours)
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks=1                         # Number of tasks
#SBATCH --cpus-per-task=8                  # CPU cores per task
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/yuvraj17/stock_trading_logs/%x-%j.out    # Standard output
#SBATCH --error=/scratch/yuvraj17/stock_trading_logs/%x-%j.err     # Standard error
#SBATCH --mail-type=BEGIN,END,FAIL         # Email notifications
#SBATCH --mail-user=ys19rk@brocku.ca       # Your email

################################################################################
# Production PPO Stock Trading - SLURM Job Script
# 
# This script runs enterprise-grade PPO training on Trillium cluster with:
# - GPU acceleration (CUDA)
# - Comprehensive logging
# - TensorBoard monitoring
# - Model checkpointing
# - Risk management
#
# Usage:
#   sbatch slurm_train_ppo.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f /scratch/yuvraj17/stock_trading_logs/<job-name>-<job-id>.out
################################################################################

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Load required modules
echo "Loading modules..."
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

# Create necessary directories
mkdir -p $LOG_DIR
mkdir -p $MODEL_DIR
mkdir -p $TENSORBOARD_DIR

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
TICKER="AAPL"

# Training mode: "production" (conservative) or "research" (aggressive)
MODE="production"

# Total training timesteps
TIMESTEPS=1000000

# Experiment name (will be used for model and log naming)
EXPERIMENT="${TICKER}_${MODE}_$(date +%Y%m%d_%H%M%S)"

# GPU ID (0 for first GPU)
GPU_ID=0

# Random seed for reproducibility
SEED=42

################################################################################
# Run Training
################################################################################

echo "=========================================="
echo "Starting PPO Training"
echo "=========================================="
echo "Ticker: $TICKER"
echo "Mode: $MODE"
echo "Timesteps: $TIMESTEPS"
echo "Experiment: $EXPERIMENT"
echo "GPU: $GPU_ID"
echo "=========================================="
echo ""

# Run training with all outputs logged
python train_production_ppo.py \
    --ticker $TICKER \
    --mode $MODE \
    --timesteps $TIMESTEPS \
    --gpu $GPU_ID \
    --seed $SEED \
    --experiment $EXPERIMENT \
    2>&1 | tee $LOG_DIR/${EXPERIMENT}_training.log

# Capture exit code
EXIT_CODE=$?

################################################################################
# Post-Training Tasks
################################################################################

echo ""
echo "=========================================="
echo "Training Completed"
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=========================================="

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

# Print summary
echo ""
echo "=========================================="
echo "Job Summary"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Experiment: $EXPERIMENT"
echo "Exit Code: $EXIT_CODE"
echo "Models: $MODEL_DIR"
echo "Logs: $LOG_DIR"
echo "TensorBoard: $TENSORBOARD_DIR"
echo ""
echo "To view TensorBoard:"
echo "  tensorboard --logdir $TENSORBOARD_DIR --host 0.0.0.0 --port 6006"
echo "=========================================="

# Print resource usage
echo ""
echo "Resource Usage:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,AllocCPUS,State,ExitCode,Elapsed,MaxRSS,MaxVMSize

# Exit with training exit code
exit $EXIT_CODE

#!/bin/bash
#SBATCH --job-name=bullinger_qwen
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
# Optional: choose a node type
# #SBATCH --nodelist=diufrd203    # L40S (48GB)

set -euo pipefail

echo "=== Node ==="
hostname
nvidia-smi || true
echo "============"

# Env
source ~/.bashrc
conda activate bullinger-mwe

# Torch with CUDA (adjust if your cluster uses a different CUDA runtime)
pip show torch >/dev/null 2>&1 || pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision --quiet
pip install -q transformers>=4.43.0 accelerate bitsandbytes pillow

# Cache to project dir (faster + avoids cluttering $HOME)
export HF_HOME="$PWD/.hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$TRANSFORMERS_CACHE"

# Run
python run_eval_qwen.py \
  --data-dir data_val \
  --hf-model Qwen/Qwen3-VL-8B-Instruct \
  --hf-device cuda \
  --eval-csv evaluation_qwen.csv

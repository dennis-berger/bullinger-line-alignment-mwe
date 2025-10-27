#!/bin/bash
#SBATCH --job-name=bullinger_qwen
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
# Prefer the L40S if available (48GB VRAM); V100 works with 4-bit:
#SBATCH --nodelist=diufrd203   # L40S
# #SBATCH --nodelist=diufrd204   # V100 32GB

set -euo pipefail
set -x

echo "=== Node ==="
hostname
nvidia-smi || true
echo "============"

# Robust conda activation
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  eval "$(/usr/bin/conda shell.bash hook)" || true
fi
conda activate bullinger-mwe

# HF cache under project
export HF_HOME="$PWD/.hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$TRANSFORMERS_CACHE"

# Reduce fragmentation (per error hint)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use local model if present
MODEL="./.hf/Qwen3-VL-8B-Instruct"
[ -d "$MODEL" ] || MODEL="Qwen/Qwen3-VL-8B-Instruct"

python run_eval_qwen.py \
  --data-dir data_val \
  --hf-model "$MODEL" \
  --hf-device cuda \
  --eval-csv evaluation_qwen.csv \
  --max-new-tokens 800

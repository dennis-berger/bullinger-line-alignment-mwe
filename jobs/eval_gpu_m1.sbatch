#!/bin/bash
#SBATCH --job-name=bullinger_qwen_m1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

set -euo pipefail
set -x

echo "=== Node ==="
hostname
nvidia-smi || true
echo "============"

# Make conda available
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: 'conda' not found. Load your conda module or install Miniconda."
  exit 1
fi
eval "$(conda shell.bash hook)"
conda activate bullinger-mwe

# HF cache in project dir
export HF_HOME="$PWD/.hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$TRANSFORMERS_CACHE"

# Reduce CUDA fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Prefer local snapshot if present, else hub
MODEL="./.hf/Qwen3-VL-8B-Instruct"
[ -d "$MODEL" ] || MODEL="Qwen/Qwen3-VL-8B-Instruct"

which python
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

python run_eval_m1.py \
  --data-dir data_val \
  --hf-model "$MODEL" \
  --hf-device cuda \
  --out-dir predictions_m1 \
  --eval-csv evaluation_qwen_m1.csv \
  --max-new-tokens 800

echo "Done."

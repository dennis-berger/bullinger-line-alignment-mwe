#!/bin/bash
#SBATCH --job-name=bull_qwen_m1_print
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G

set -euo pipefail
set -x

echo "=== Node ==="
hostname
nvidia-smi || true
echo "============"

# Conda
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: 'conda' not found."
  exit 1
fi
eval "$(conda shell.bash hook)"
conda activate bullinger-mwe

# HF cache in project dir
export HF_HOME="$PWD/.hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$TRANSFORMERS_CACHE"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="./.hf/Qwen3-VL-8B-Instruct"
[ -d "$MODEL" ] || MODEL="Qwen/Qwen3-VL-8B-Instruct"

which python
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

python run_eval_qwen_m1.py \
  --data-dir bullinger_print_dataset \
  --out-dir bullinger_print_predictions_m1 \
  --eval-csv bullinger_print_eval_qwen_m1.csv \
  --hf-model "$MODEL" \
  --hf-device cuda \
  --max-new-tokens 800

echo "Done."

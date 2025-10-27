#!/bin/bash
#SBATCH --job-name=bullinger_qwen
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
# Optional (pick one if you want to pin a node):
# #SBATCH --nodelist=diufrd203   # L40S 48GB
# #SBATCH --nodelist=diufrd204   # V100 32GB

set -euo pipefail
set -x

echo "=== Node ==="
hostname
nvidia-smi || true
echo "============"

# --- Try to make conda available in a non-interactive shell ---

# Try environment modules, ignore failures
if command -v module >/dev/null 2>&1; then
  module purge || true
  module load anaconda || true
  module load miniconda || true
  module load miniconda3 || true
fi

# If you previously ran `conda init bash`, sourcing ~/.bashrc is often enough
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"

# If conda still not on PATH, source common locations (use ~ to avoid /HOME vs /home mismatch)
if ! command -v conda >/dev/null 2>&1; then
  [ -f ~/miniconda3/etc/profile.d/conda.sh ] && source ~/miniconda3/etc/profile.d/conda.sh || true
  [ -f ~/anaconda3/etc/profile.d/conda.sh ]  && source ~/anaconda3/etc/profile.d/conda.sh  || true
fi

# Last-resort: try the shell hook if conda exists but isn't initialized
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" || true
else
  echo "ERROR: 'conda' command not found. Please load your conda module or install Miniconda in your home."
  exit 1
fi

# --- Activate env (create if missing) ---
conda activate bullinger-mwe || (conda create -n bullinger-mwe python=3.10 -y && conda activate bullinger-mwe)

# --- (Optional) Ensure deps already installed in the env (do this once on login node for speed) ---
# pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
# pip install "transformers>=4.43.0" accelerate bitsandbytes pillow huggingface_hub

# Use local Hugging Face cache under the project directory
export HF_HOME="$PWD/.hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$TRANSFORMERS_CACHE"

# Reduce CUDA fragmentation (helps on V100 32GB)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Prefer local model snapshot if you pre-downloaded it; otherwise use the hub id
MODEL="./.hf/Qwen3-VL-8B-Instruct"
[ -d "$MODEL" ] || MODEL="Qwen/Qwen3-VL-8B-Instruct"

# Sanity print
which python || true
python -c "import sys, os; print(sys.version); print('PYTORCH_CUDA_ALLOC_CONF=', os.environ.get('PYTORCH_CUDA_ALLOC_CONF','<unset>'))" || true

# Run
python run_eval_qwen.py \
  --data-dir data_val \
  --hf-model "$MODEL" \
  --hf-device cuda \
  --eval-csv evaluation_qwen.csv \
  --max-new-tokens 800

echo "Done."

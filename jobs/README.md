# FAITH Cluster Jobs

These Slurm job scripts reproduce the experiments on the **FAITH HPC cluster** at Uni Fribourg.

- `eval_gpu_qwen.sbatch`: runs `run_eval_qwen.py` on a single GPU (L40S/3080/V100)

Usage:

```bash
sbatch jobs/eval_gpu_qwen.sbatch
squeue -u $USER
tail -f logs/bullinger_qwen_*.out

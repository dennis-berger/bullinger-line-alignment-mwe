# FAITH Cluster Jobs

These Slurm job scripts reproduce the experiments on the **FAITH HPC cluster** at Uni Fribourg.

- `eval_gpu_qwen.sbatch`: runs `run_eval_qwen.py` on a single GPU (L40S/3080/V100)

## Usage:
For detailed documentation of the Faith HPC Cluster vistit: https://diuf-doc.unifr.ch/books/faith-hpc-cluster


Logging into the Faith Cluster

Before submitting jobs, connect to the cluster login node:
```bash
ssh <USERNAME>@diufrd200.unifr.ch
```

Submit jobs:
```bash
sbatch jobs/eval_gpu_qwen.sbatch
squeue -u $USER
tail -f logs/bullinger_qwen_*.out
```

Copy results to local machine:
```bashß
scp bergerd@diufrd200.unifr.ch:~/projects/bullinger-line-alignment-mwe/<evaluationfile> .
scp -r bergerd@diufrd200.unifr.ch:~/projects/bullinger-line-alignment-mwe/predictions ./predictions

```
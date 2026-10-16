#!/bin/bash
#SBATCH -C "gpu"
#SBATCH -A als
#SBATCH -q debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH -o /pscratch/sd/m/monika/metric_study/logs/benchmark_%j.out
#SBATCH -e /pscratch/sd/m/monika/metric_study/logs/benchmark_%j.err
#SBATCH --job-name=metric_study_benchmark
#
# Plan §4.4/§7: measure per-metric, per-resolution runtime and peak GPU memory
# BEFORE launching the full array, so the plan's §7 estimates are replaced with
# real numbers. `cli.py benchmark` is not implemented yet (plan §5.2 lists it as
# pending a checkpoint to benchmark reconstruction inference against, plan §9 Q1)
# -- this script is a placeholder for when it is.

set -euo pipefail
cd /pscratch/sd/m/monika/metric_study

module purge
module load cuda
module load python
source /global/homes/m/monika/miniconda3/etc/profile.d/conda.sh
conda activate /pscratch/sd/m/monika/conda/envs/metric_study

python -c "import torch; print('cuda available:', torch.cuda.is_available())"

python -u -m metric_study.cli benchmark \
    --config configs/metric_study.yaml \
    --section default \
    --output-dir /pscratch/sd/m/monika/metric_study/outputs/benchmark \
    --batch-sizes 8,32,64

EXIT_CODE=$?
echo "Benchmark job finished with exit code: $EXIT_CODE"
exit $EXIT_CODE

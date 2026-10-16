#!/bin/bash
#SBATCH -C "gpu"
#SBATCH -A als
#SBATCH -q debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH -o /pscratch/sd/m/monika/metric_study/logs/dry_run_%j.out
#SBATCH -e /pscratch/sd/m/monika/metric_study/logs/dry_run_%j.err
#SBATCH --job-name=metric_study_dry_run
#
# Plan §6.7: runs the exact production code path on the small sample dataset
# before the full 1.5M-image launch. Copy `sample_data/` to NERSC first (or point
# --data-dir at wherever it lands), since this script assumes a NERSC filesystem
# path rather than the local macOS path used during development.

set -euo pipefail
cd /pscratch/sd/m/monika/metric_study

module purge
module load python
source /global/homes/m/monika/miniconda3/etc/profile.d/conda.sh
conda activate /pscratch/sd/m/monika/conda/envs/metric_study

python -u -m metric_study.cli dry-run \
    --config configs/metric_study.yaml \
    --section dry_run \
    --data-dir /pscratch/sd/m/monika/metric_study/sample_data \
    --output-dir /pscratch/sd/m/monika/metric_study/outputs/dry_run

EXIT_CODE=$?
echo "Dry run finished with exit code: $EXIT_CODE"
exit $EXIT_CODE

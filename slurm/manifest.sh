#!/bin/bash
#SBATCH -C "cpu"
#SBATCH -A als
#SBATCH -q regular
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=64
#SBATCH --time=01:00:00
#SBATCH -o /pscratch/sd/m/monika/metric_study/logs/manifest_%j.out
#SBATCH -e /pscratch/sd/m/monika/metric_study/logs/manifest_%j.err
#SBATCH --job-name=metric_study_manifest
#
# Plan §7: manifest + mask pass over ~1.5M images. Same header style as the
# training repo's VAE_gpu_500_regular_512.sh (account/QOS placeholders --
# NERSC account/QOS/env path are open questions, plan §9 Q8).

set -euo pipefail
cd /pscratch/sd/m/monika/metric_study

module purge
module load python
source /global/homes/m/monika/miniconda3/etc/profile.d/conda.sh
conda activate /pscratch/sd/m/monika/conda/envs/metric_study  # NEW, dedicated env (plan §5.1)

DATA_DIR=/pscratch/sd/m/monika/bl733_data_png_test/   # confirm production path, plan §9 Q2
OUTPUT_DIR=/pscratch/sd/m/monika/metric_study/outputs/production

python -m metric_study.cli manifest --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?
echo "Manifest job finished with exit code: $EXIT_CODE"
exit $EXIT_CODE

#!/bin/bash
#SBATCH -C "gpu"
#SBATCH -A als
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --array=0-7            # placeholder: set to (n_chunks / n_nodes_per_task), see plan §6.2/§7
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH -o /pscratch/sd/m/monika/metric_study/logs/run_%A_%a.out
#SBATCH -e /pscratch/sd/m/monika/metric_study/logs/run_%A_%a.err
#SBATCH --job-name=metric_study_run
#
# Plan §6.2/§6.3: one node per array task, 4 ranks x 1 GPU each. Chunk-level output
# is idempotent (chunk=NNNNNN.parquet + .done marker), so resubmitting this exact
# array after a crash or preemption skips already-finished chunks automatically --
# no separate "resume" script needed.

set -euo pipefail
cd /pscratch/sd/m/monika/metric_study

module purge
module load cuda
module load python
source /global/homes/m/monika/miniconda3/etc/profile.d/conda.sh
conda activate /pscratch/sd/m/monika/conda/envs/metric_study

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=configs/metric_study.yaml
SECTION=production   # add a `production` section to metric_study.yaml before the full run
OUTPUT_DIR=/pscratch/sd/m/monika/metric_study/outputs/production

echo "Array task $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT"

# Each rank in this array task claims a disjoint slice of chunks. `cli.py run`
# currently processes chunks sequentially within one process (plan §6.2's
# GPU-batched, multi-rank version is a follow-up once `benchmark` numbers are in --
# see plans/metric_study.md §7); this srun line divides work across ranks by having
# each one call `run` with a distinct --limit/offset once that flag exists.
# Placeholder invocation until per-rank chunk sharding is added to cli.py:
srun --ntasks=4 --gpus-per-task=1 python -u -m metric_study.cli run \
    --config "$CONFIG" \
    --section "$SECTION" \
    --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?
echo "Run array task finished with exit code: $EXIT_CODE"
exit $EXIT_CODE

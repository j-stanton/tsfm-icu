#!/bin/bash
#SBATCH --job-name=tsfm-smoke-test
#SBATCH --partition=stampede
#SBATCH -w mscluster40
# ^ pinned to the node CONFIRMED working for GPU access (see
#   SETUP_ENVIRONMENT.md). Remove once confirmed to work identically on
#   other stampede nodes.
#
# SUBMIT THIS FROM ~/tsfm-icu-early-warning/zero_shot/ (cd there, then
# `sbatch run_smoke_test.slurm`) -- $SLURM_SUBMIT_DIR and the script
# references below assume that's where this file and its sibling pipeline
# files (yaib_data.py, zero_shot_pipeline.py, etc.) live.
#SBATCH --output=/home-mscluster/%u/tsfm-icu-early-warning/zero_shot/results/smoke-output-%N-%j.txt
#SBATCH --error=/home-mscluster/%u/tsfm-icu-early-warning/zero_shot/results/smoke-error-%N-%j.txt

# Runs, as a single batch job rather than an interactive srun session:
#   1. smoke_test.py             -- scaffolding checks (no data, no GPU work)
#   2. make_synthetic_yaib_cohort.py -- writes synthetic dyn/outc/sta.parquet
#   3. run_zero_shot.py --max-patients 12 -- a small end-to-end zero-shot
#      dry run against that synthetic cohort (this DOES download and run the
#      real ~800MB TimesFM checkpoint on first call -- expect this step to
#      take a few minutes, most of it the one-time download).
#
# All synthetic data and intermediate files are written under
# $SLURM_SUBMIT_DIR/smoke_test_data/, not /tmp -- since batch jobs on this
# cluster may not land on the same node each time, /tmp from an earlier
# interactive session wouldn't be visible here anyway; using a path under
# $SLURM_SUBMIT_DIR keeps everything in one place you can inspect after the
# job finishes.

sleep 20

echo "=== Useful Debugging Information ==="
echo "Date: "$(date)
echo "Username: "$USER
echo "Job ID: "$SLURM_JOB_ID
echo "Job Name: "$SLURM_JOB_NAME
echo "Home Directory: "$HOME
echo "Node: "$HOSTNAME
echo "Current Working Directory: "$PWD
echo "Job Submitted From Machine: "$SLURM_SUBMIT_HOST
echo "Job Submitted From Directory: "$SLURM_SUBMIT_DIR
echo "Number of Nodes: " $SLURM_JOB_NUM_NODES
echo "List of Nodes: "$SLURM_JOB_NODELIST
echo "Number of Tasks: "$SLURM_NTASKS
echo "======"

cd $SLURM_SUBMIT_DIR
mkdir -p results
mkdir -p smoke_test_data

source /etc/profile.d/lmod.sh
module load pytorch/pytorch-cuda12.9-ubuntu24.04-00

# Same non-interactive apptainer+conda pattern as run_env_check.slurm --
# see SETUP_ENVIRONMENT.md for why this specific sequence is needed (the
# module's own `conda-activate` wrapper only works from an interactive
# shell; this is the batch-job equivalent).
#
# SIF_PATH: same caveat as run_env_check.slurm -- inferred, not yet
# directly confirmed against this exact module version. Check with
# `module show pytorch/pytorch-cuda12.9-ubuntu24.04-00` if this fails with
# a "file not found" error on the .sif path specifically (as opposed to
# failing inside the python scripts themselves, which would be a
# different, unrelated problem).
SIF_PATH="/usr/local/containers/pytorch/pytorch-cuda12.9-ubuntu24.04-00.sif"

apptainer exec --nv "$SIF_PATH" bash -c "
    source /opt/conda/etc/profile.d/conda.sh
    conda activate tsfm-adapter

    set -e  # stop at the first failing step rather than silently
            # continuing to the next one with bad/missing inputs

    echo '--- Step 1: smoke_test.py ---'
    cd $SLURM_SUBMIT_DIR
    python3 smoke_test.py

    echo ''
    echo '--- Step 2: make_synthetic_yaib_cohort.py ---'
    python3 -c \"
from make_synthetic_yaib_cohort import make_synthetic_cohort
make_synthetic_cohort('$SLURM_SUBMIT_DIR/smoke_test_data/synthetic_yaib_cohort')
print('synthetic cohort written OK')
\"

    echo ''
    echo '--- Step 3: build per-task subdirs for the driver script ---'
    for task_dir in mortality24 aki sepsis; do
        mkdir -p '$SLURM_SUBMIT_DIR/smoke_test_data/synthetic_yaib_root/'\$task_dir
        cp '$SLURM_SUBMIT_DIR/smoke_test_data/synthetic_yaib_cohort/'*.parquet \
           '$SLURM_SUBMIT_DIR/smoke_test_data/synthetic_yaib_root/'\$task_dir/
    done

    echo ''
    echo '--- Step 4: run_zero_shot.py (small dry run, max 5 patients) ---'
    echo '(first call downloads the real TimesFM checkpoint, ~800MB -- expect this to take a few minutes)'
    python3 run_zero_shot.py \
        --cohort-root '$SLURM_SUBMIT_DIR/smoke_test_data/synthetic_yaib_root' \
        --max-patients 12 \
        --output '$SLURM_SUBMIT_DIR/results/smoke_test_phase2_results.json'
"

echo ""
echo "Done. Check results/smoke_test_phase2_results.json for the dry-run output table."

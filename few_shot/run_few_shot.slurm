#!/bin/bash
#SBATCH --job-name=tsfm-few-shot-full
#SBATCH --partition=stampede
#SBATCH -w mscluster40
#SBATCH --time=48:00:00
#SBATCH --output=/home-mscluster/%u/tsfm-icu-early-warning/few_shot/results/full-%j-output-%N.txt
#SBATCH --error=/home-mscluster/%u/tsfm-icu-early-warning/few_shot/results/full-%j-error-%N.txt
# Full few-shot run for one task and one selection strategy. Submit from few_shot/.


TASK="$1"
STRATEGY="$2"
if [[ -z "$TASK" || -z "$STRATEGY" ]]; then
    echo "ERROR: usage: sbatch run_few_shot.slurm <mortality|aki|sepsis> <random|similarity>"
    exit 1
fi
if [[ "$TASK" != "mortality" && "$TASK" != "aki" && "$TASK" != "sepsis" ]]; then
    echo "ERROR: unknown task '$TASK' -- expected mortality, aki, or sepsis"
    exit 1
fi
if [[ "$STRATEGY" != "random" && "$STRATEGY" != "similarity" ]]; then
    echo "ERROR: unknown strategy '$STRATEGY' -- expected random or similarity"
    exit 1
fi

sleep 20

echo "=== Useful Debugging Information ==="
echo "Date: "$(date)
echo "Username: "$USER
echo "Job ID: "$SLURM_JOB_ID
echo "Job Name: "$SLURM_JOB_NAME
echo "Task: "$TASK
echo "Strategy: "$STRATEGY
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

source /etc/profile.d/lmod.sh
module load pytorch/pytorch-cuda12.9-ubuntu24.04-00

SIF_PATH="/usr/local/containers/pytorch/pytorch-cuda12.9-ubuntu24.04-00.sif"

REPO_ROOT="$SLURM_SUBMIT_DIR/.."
COHORT_ROOT="$REPO_ROOT/external/YAIB-cohorts/data"
SPLIT_FILE="$REPO_ROOT/splits/shared_split.csv"

apptainer exec --nv "$SIF_PATH" bash -c "
    source /opt/conda/etc/profile.d/conda.sh
    conda activate tsfm-adapter

    set -e

    export PYTHONUNBUFFERED=1

    cd $SLURM_SUBMIT_DIR

    echo '--- Confirming real data paths exist before running ---'
    test -f '$COHORT_ROOT/mortality24/mimic/dyn.parquet' && echo 'mortality24/mimic: OK' || { echo 'MISSING: mortality24/mimic/dyn.parquet'; exit 1; }
    test -f '$COHORT_ROOT/aki/mimic/dyn.parquet' && echo 'aki/mimic: OK' || { echo 'MISSING: aki/mimic/dyn.parquet'; exit 1; }
    test -f '$COHORT_ROOT/sepsis/mimic/dyn.parquet' && echo 'sepsis/mimic: OK' || { echo 'MISSING: sepsis/mimic/dyn.parquet'; exit 1; }
    test -f '$SPLIT_FILE' && echo 'shared_split.csv: OK' || { echo 'MISSING: shared_split.csv'; exit 1; }

    echo ''
    echo '--- Phase 3 FULL run: task=$TASK, strategy=$STRATEGY, k in {1,5,10,20}, 5 seeds, NO caps ---'
    python3 -u run_few_shot.py \
        --cohort-root '$COHORT_ROOT' \
        --split-assignment '$SPLIT_FILE' \
        --tasks $TASK \
        --strategies $STRATEGY \
        --output '$SLURM_SUBMIT_DIR/results/phase3_full_${TASK}_${STRATEGY}.json'
"

echo ""
echo "Done. Check results/phase3_full_${TASK}_${STRATEGY}.json for the results table."
echo "Check the job log above for [DEGENERATE PROBE] / [UNCALIBRATED] flags per cell --"
echo "both are EXPECTED (see this file's header comment) and not errors."
echo ""
echo "If this job times out before finishing: the log (now with real progress"
echo "lines, unlike job 61982) will show you exactly which (approach/k/seed)"
echo "cell it was on -- narrow further from there (e.g. a single --ks value"
echo "per job) rather than just raising --time again blindly."

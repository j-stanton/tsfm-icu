#!/bin/bash
#SBATCH --job-name=tsfm-few-shot-dry-run
#SBATCH --partition=stampede
#SBATCH -w mscluster40
#SBATCH --output=/home-mscluster/%u/tsfm-icu-early-warning/few_shot/results/dry-run-output-%N-%j.txt
#SBATCH --error=/home-mscluster/%u/tsfm-icu-early-warning/few_shot/results/dry-run-error-%N-%j.txt
# Small few-shot dry run (10 patients).


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

source /etc/profile.d/lmod.sh
module load pytorch/pytorch-cuda12.9-ubuntu24.04-00

# Same non-interactive apptainer+conda pattern as phase2's .slurm jobs --
# see SETUP_ENVIRONMENT.md for why this specific sequence is needed.
SIF_PATH="/usr/local/containers/pytorch/pytorch-cuda12.9-ubuntu24.04-00.sif"

REPO_ROOT="$SLURM_SUBMIT_DIR/.."
COHORT_ROOT="$REPO_ROOT/external/YAIB-cohorts/data"
SPLIT_FILE="$REPO_ROOT/splits/shared_split.csv"

apptainer exec --nv "$SIF_PATH" bash -c "
    source /opt/conda/etc/profile.d/conda.sh
    conda activate tsfm-adapter

    set -e

    cd $SLURM_SUBMIT_DIR

    echo '--- Confirming real data paths exist before running ---'
    test -f '$COHORT_ROOT/mortality24/mimic/dyn.parquet' && echo 'mortality24/mimic: OK' || { echo 'MISSING: mortality24/mimic/dyn.parquet'; exit 1; }
    test -f '$SPLIT_FILE' && echo 'shared_split.csv: OK' || { echo 'MISSING: shared_split.csv'; exit 1; }

    echo ''
    echo '--- Phase 3 dry run: run_few_shot.py (mortality, k in {1,5}, random strategy, 2 seeds) ---'
    python3 run_few_shot.py \
        --cohort-root '$COHORT_ROOT' \
        --split-assignment '$SPLIT_FILE' \
        --tasks mortality \
        --ks 1 5 \
        --strategies random \
        --n-seeds 2 \
        --max-patients 10 \
        --max-train-pool 200 \
        --output '$SLURM_SUBMIT_DIR/results/phase3_dry_run.json'
"

echo ""
echo "Done. Check results/phase3_dry_run.json for AUROC/AUPRC/ECE per (task,approach,strategy,k)."
echo "Check the printed [DEGENERATE PROBE] / [UNCALIBRATED] flags in the job log above --"
echo "[UNCALIBRATED] is EXPECTED and fine (k<=20 essentially never has 5+ per class,"
echo "so Approach A's probe uses LogisticRegression's raw sigmoid, not Platt scaling --"
echo "see forecast_to_label.py's ProbeClassifier); [DEGENERATE PROBE] means at least"
echo "one seed/query hit a fully single-class demo draw (more likely at k=1) and fell"
echo "back to a neutral 0.5 score for that cell -- worth checking how often this fires"
echo "before trusting a specific (task,k=1,strategy) cell's Approach-A numbers."

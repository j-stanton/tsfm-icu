#!/bin/bash
#SBATCH --job-name=tsfm-few-shot-sim
#SBATCH --partition=stampede
#SBATCH -w mscluster40
#SBATCH --time=46:00:00
#SBATCH --output=/home-mscluster/%u/tsfm-icu-early-warning/few_shot/results/sim-%j-output-%N.txt
#SBATCH --error=/home-mscluster/%u/tsfm-icu-early-warning/few_shot/results/sim-%j-error-%N.txt
# Similarity-based few-shot run on mortality, capped at 500 test patients.


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
    test -f '$SPLIT_FILE' && echo 'shared_split.csv: OK' || { echo 'MISSING: shared_split.csv'; exit 1; }

    echo ''
    echo '--- Phase 3 CAPPED similarity run: task=mortality, strategy=similarity, k in {1,5}, 5 seeds, max-patients=500 ---'
    echo '--- --resume is set: if results/phase3_similarity_capped_mortality.json already has'
    echo '--- completed (task,approach,strategy,k) cells from an earlier attempt, they are'
    echo '--- skipped rather than re-run (see run_few_shot.py --resume help text).'
    python3 -u run_few_shot.py \
        --cohort-root '$COHORT_ROOT' \
        --split-assignment '$SPLIT_FILE' \
        --tasks mortality \
        --ks 1 5 \
        --strategies similarity \
        --approaches probe \
        --n-seeds 5 \
        --max-patients 500 \
        --resume \
        --output '$SLURM_SUBMIT_DIR/results/phase3_similarity_capped_mortality.json'
"

echo ""
echo "Done. Check results/phase3_similarity_capped_mortality.json for the results table."
echo "This is a SCOPED result (max-patients=500, ks={1,5} only) -- not directly"
echo "comparable in CI width to phase3_full_mortality_random.json's full-5762-patient"
echo "numbers, but DOES give a fair, same-sized, same-seeds comparison of"
echo "strategy=similarity vs strategy=random's AUROC/AUPRC/ECE on the same slice."
echo "If you want that direct comparison, rerun strategy=random with the same"
echo "--max-patients 500 --ks 1 5 flags for an apples-to-apples pair."
echo ""
echo "NOTE (see few_shot_pipeline.py's _fit_probe_on_demos / run_few_shot_evaluation):"
echo "Approach A's probe is trained on ZERO-SHOT forecast features of the"
echo "demonstrations but scored on ICL-CONDITIONED forecast features of the query --"
echo "a real train/inference feature-distribution mismatch, not a crash-level bug."
echo "This affects probe numbers under BOTH strategies, but is worth keeping in mind"
echo "if this job's k=5 AUROC looks unexpectedly weak or below-chance, the way"
echo "aki/random's k=5/k=10 did (see phase3_full_aki_random.json)."

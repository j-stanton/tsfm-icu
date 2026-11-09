#!/bin/bash
#SBATCH --job-name=tsfm-real-dry-run
#SBATCH --partition=stampede
#SBATCH -w mscluster40
#SBATCH --output=/home-mscluster/%u/tsfm-icu-early-warning/zero_shot/results/real-dry-run-output-%N-%j.txt
#SBATCH --error=/home-mscluster/%u/tsfm-icu-early-warning/zero_shot/results/real-dry-run-error-%N-%j.txt
# Zero-shot dry run on the real cohorts with small patient caps.


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

    set -e  # stop at the first failing step rather than silently
            # continuing with bad/missing inputs

    cd $SLURM_SUBMIT_DIR

    echo '--- Confirming real data paths exist before running ---'
    test -f '$COHORT_ROOT/mortality24/mimic/dyn.parquet' && echo 'mortality24/mimic: OK' || { echo 'MISSING: mortality24/mimic/dyn.parquet'; exit 1; }
    test -f '$COHORT_ROOT/aki/mimic/dyn.parquet' && echo 'aki/mimic: OK' || { echo 'MISSING: aki/mimic/dyn.parquet'; exit 1; }
    test -f '$COHORT_ROOT/sepsis/mimic/dyn.parquet' && echo 'sepsis/mimic: OK' || { echo 'MISSING: sepsis/mimic/dyn.parquet'; exit 1; }
    test -f '$SPLIT_FILE' && echo 'shared_split.csv: OK' || { echo 'MISSING: shared_split.csv'; exit 1; }

    echo ''
    echo '--- Real-data dry run: run_zero_shot.py --max-patients 20 --max-calibration-patients 50 ---'
    echo '(first call downloads the real TimesFM checkpoint if not already cached, ~800MB)'
    python3 run_zero_shot.py \
        --cohort-root '$COHORT_ROOT' \
        --split-assignment '$SPLIT_FILE' \
        --tasks mortality aki sepsis \
        --max-patients 20 \
        --max-calibration-patients 50 \
        --output '$SLURM_SUBMIT_DIR/results/real_data_dry_run.json'
"

echo ""
echo "Done. Check results/real_data_dry_run.json for AUROC/AUPRC/ECE per task+approach."
echo "Sanity check before trusting these numbers: AUROC should NOT be exactly"
echo "1.000 and identical across all 5 seeds the way the earlier SYNTHETIC"
echo "smoke test's numbers were (that was an artifact of the synthetic data's"
echo "deterministic label-to-signal construction, not a real result)."

#!/bin/bash
#SBATCH --job-name=tsfm-zero-shot-full
#SBATCH --partition=stampede
# No -w pin: the diagnostic job (58204) queued faster on an unpinned
# request and landed on mscluster38, confirming GPU access isn't uniquely
# tied to mscluster40 -- pinning here would only add unnecessary queue
# wait for a job that's already going to take a while to run.
#
# SUBMIT THIS FROM ~/tsfm-icu-early-warning/zero_shot/ (cd there, then
# `sbatch run_zero_shot.slurm`).
#SBATCH --output=/home-mscluster/%u/tsfm-icu-early-warning/zero_shot/results/full-run-output-%N-%j.txt
#SBATCH --error=/home-mscluster/%u/tsfm-icu-early-warning/zero_shot/results/full-run-error-%N-%j.txt

# Runs run_zero_shot.py against REAL YAIB-cohorts output and the
# REAL shared_split.csv, WITHOUT --max-patients / --max-calibration-patients
# -- i.e. the full val/test splits for mortality, and the full set of
# sampled patient-hours (yaib_data.ROLLING_HOURS_PER_PATIENT=3 per
# patient, across every patient in each split) for aki/sepsis.
#
# WHY NO CAPS THIS TIME: the earlier --max-patients 20 /
# --max-calibration-patients 50 dry run got AKI's probe AUROC down to
# 0.098 -- confirmed via a per-patient diagnostic (diagnose_aki_probe.py,
# job 58204) to be a genuine small-sample/class-imbalance artifact, NOT a
# code bug (the true positives all scored near-zero, consistent with an
# overfit logistic regression fit on very few positive examples out of
# only 50 calibration patient-hours for a rare, sparse-onset outcome).
# Removing the caps uses the FULL calibration split, where even a low
# per-patient-hour positive rate yields a much larger absolute number of
# positive examples -- expected to resolve the artifact without any
# stratified-sampling workaround, since the underlying issue was sample
# size, not pipeline logic.
#
# EXPECTED RUNTIME: substantially longer than the capped dry run
# (minutes). Mortality alone has ~11,467 val+test patients; AKI/sepsis
# sample 3 hours x every patient in their (much larger, intersected)
# val+test splits. This is a genuine background job -- do not expect to
# babysit it interactively; check back via `squeue -u $USER` and the
# output file once it's had time to run.
#
# Otherwise identical prerequisites/mechanics to run_real_dry_run.slurm
# (memory-safe TimesFM loading, rolling-task dispatch, real-path
# preflight checks) -- see that file's own comments and
# zero_shot_pipeline.py/yaib_data.py for the fuller history if anything here
# needs revisiting.

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

    cd $SLURM_SUBMIT_DIR

    echo '--- Confirming real data paths exist before running ---'
    test -f '$COHORT_ROOT/mortality24/mimic/dyn.parquet' && echo 'mortality24/mimic: OK' || { echo 'MISSING: mortality24/mimic/dyn.parquet'; exit 1; }
    test -f '$COHORT_ROOT/aki/mimic/dyn.parquet' && echo 'aki/mimic: OK' || { echo 'MISSING: aki/mimic/dyn.parquet'; exit 1; }
    test -f '$COHORT_ROOT/sepsis/mimic/dyn.parquet' && echo 'sepsis/mimic: OK' || { echo 'MISSING: sepsis/mimic/dyn.parquet'; exit 1; }
    test -f '$SPLIT_FILE' && echo 'shared_split.csv: OK' || { echo 'MISSING: shared_split.csv'; exit 1; }

    echo ''
    echo '--- FULL real-data run: run_zero_shot.py (no patient caps) ---'
    echo '(first call downloads the real TimesFM checkpoint if not already cached, ~800MB)'
    python3 run_zero_shot.py \
        --cohort-root '$COHORT_ROOT' \
        --split-assignment '$SPLIT_FILE' \
        --tasks mortality aki sepsis \
        --output '$SLURM_SUBMIT_DIR/results/full_run_results.json'
"

echo ""
echo "Done. Check results/full_run_results.json for AUROC/AUPRC/ECE per task+approach."

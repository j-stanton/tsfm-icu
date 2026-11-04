#!/bin/bash
#SBATCH --job-name=tsfm-real-dry-run
#SBATCH --partition=stampede
#SBATCH -w mscluster40
# ^ pinned to the node CONFIRMED working for GPU access (see
#   SETUP_ENVIRONMENT.md). Remove once confirmed to work identically on
#   other stampede nodes.
#
# SUBMIT THIS FROM ~/tsfm-icu-early-warning/zero_shot/ (cd there, then
# `sbatch run_real_dry_run.slurm`) -- $SLURM_SUBMIT_DIR and the script
# references below assume that's where this file and its sibling pipeline
# files (yaib_data.py, zero_shot_pipeline.py, etc.) live.
#SBATCH --output=/home-mscluster/%u/tsfm-icu-early-warning/zero_shot/results/real-dry-run-output-%N-%j.txt
#SBATCH --error=/home-mscluster/%u/tsfm-icu-early-warning/zero_shot/results/real-dry-run-error-%N-%j.txt

# Runs run_zero_shot.py against REAL YAIB-cohorts output (mimic
# source, not mimic_demo) and the REAL shared_split.csv -- the same split
# file the classical baselines use (Section 3.2's patient-level split
# requirement). --max-patients 20 caps the eval/test split for a fast
# first check; --max-calibration-patients 50 SEPARATELY caps the
# calibration/val split (this distinction matters -- see
# zero_shot_pipeline.py's run_zero_shot_evaluation docstring: an earlier
# version of this job left the calibration split completely uncapped,
# which on the real mortality cohort meant running full TimesFM inference
# over all 5,705 val patients before ever reaching the small eval loop,
# making the job appear to hang and eventually fail with a misleading
# "0 usable patients" error that was actually a very-slow-but-working
# loop, not an empty split).
#
# PREREQUISITES (all confirmed present on this cluster as of 2026-09-14):
#   - Real YAIB-cohorts output at
#     ~/tsfm-icu-early-warning/external/YAIB-cohorts/data/{mortality24,aki,sepsis}/mimic/
#   - Real shared_split.csv at ~/tsfm-icu-early-warning/splits/shared_split.csv
#     (stay_id,split columns; confirmed via `head` against the real file)
#   - tsfm-adapter conda env with timesfm installed (see SETUP_ENVIRONMENT.md)
#
# This uses the SAME memory-safe TimesFM loading (per_core_batch_size=8,
# model loaded once per task/approach not once per seed) that fixed a real
# CUDA OOM during the earlier synthetic-data smoke test -- see
# timesfm_loader.py and run_zero_shot.py's own comments for that
# history if this OOMs again on the larger real cohort.
#
# AKI/SEPSIS ROLLING-TASK NOTE (added 2026-09-14, after this job's first
# real run crashed on AKI): AKI and sepsis's outc.parquet is NOT one row
# per patient the way mortality's is -- it's one row per (patient, hour)
# (confirmed: AKI's real outc.parquet has 2,216,091 rows, one sample
# patient alone had 142 hourly rows), matching Section 3.2/3.4's own
# "rolling-window tasks (sepsis, AKI)" framing. yaib_data.py/
# zero_shot_pipeline.py now detect and handle this automatically
# (task_kind="rolling"), sampling yaib_data.ROLLING_HOURS_PER_PATIENT (3)
# hours per patient rather than evaluating every hour of every patient
# (which would be both computationally impractical and would violate the
# patient-level-independence assumption behind Section 3.4's bootstrap
# CI). --max-patients/--max-calibration-patients for these two tasks caps
# total PATIENT-HOUR pairs, not patient count directly -- see
# zero_shot_pipeline.py's _get_candidate_ids for the exact mechanics.

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

# Same non-interactive apptainer+conda pattern as run_smoke_test.slurm /
# run_env_check.slurm -- see SETUP_ENVIRONMENT.md for why this specific
# sequence is needed (the module's own `conda-activate` wrapper only works
# from an interactive shell; this is the batch-job equivalent).
#
# SIF_PATH: same caveat as the other .slurm jobs -- inferred, not yet
# directly confirmed against this exact module version. Check with
# `module show pytorch/pytorch-cuda12.9-ubuntu24.04-00` if this fails with
# a "file not found" error on the .sif path specifically.
SIF_PATH="/usr/local/containers/pytorch/pytorch-cuda12.9-ubuntu24.04-00.sif"

# Real paths, confirmed on this cluster via `find`/`head` in this project's
# own troubleshooting history, and matching this project's own established
# $REPO_ROOT/external/YAIB-cohorts/data convention (used elsewhere in this
# pipeline's other scripts) -- not guessed.
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

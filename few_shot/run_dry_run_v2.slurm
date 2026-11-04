#!/bin/bash
#SBATCH --job-name=tsfm-few-shot-dry-run-v2
#SBATCH --partition=stampede
#SBATCH -w mscluster40
# ^ same pinned node phase2's jobs used, confirmed working for GPU access
#   (see SETUP_ENVIRONMENT.md). Remove once confirmed to work identically
#   on other stampede nodes.
#
# SUBMIT THIS FROM ~/tsfm-icu-early-warning/few_shot/ (cd there, then
# `sbatch run_dry_run_v2.slurm`) -- $SLURM_SUBMIT_DIR and the script
# references below assume that's where this file and its sibling Phase 3
# files (yaib_data.py, few_shot_pipeline.py, timesfm_icl.py, etc.) live.
#SBATCH --output=/home-mscluster/%u/tsfm-icu-early-warning/few_shot/results/dry-run-v2-output-%N-%j.txt
#SBATCH --error=/home-mscluster/%u/tsfm-icu-early-warning/few_shot/results/dry-run-v2-error-%N-%j.txt

# v2 of run_dry_run.slurm. The first dry run (job 61803) completed
# with NO crashes and correctly-behaving [DEGENERATE PROBE]/[UNCALIBRATED]
# flags, but every cell came back AUROC=nan/AUPRC=nan -- NOT a pipeline
# bug: sklearn's own warnings in that run's error log ("Only one class is
# present in y_true... No positive class found in y_true") show the
# --max-patients 10 eval slice was single-class (all label=0). Mortality's
# real positive rate is ~9-10% (Phase 2's full zero-shot run: n=5762,
# AUPRC=0.181), so 10 UNSTRATIFIED sequential test-split candidates easily
# land all-negative by chance -- run_few_shot_evaluation correctly returns
# nan rather than crashing when len(set(y_true))<=1 (matches Phase 2's
# zero-shot behaviour), but that makes a 10-patient eval slice too small
# to trust here. The only change in this version is a larger
# --max-patients, so the eval slice is big enough to actually contain
# both classes.
#
# Scope, otherwise identical to v1:
#   --tasks mortality           (cheapest task: fixed, not rolling)
#   --ks 1 5                    (smallest two k values -- also the ones
#                                 most likely to exercise ProbeClassifier's
#                                 small-sample uncalibrated fallback; see
#                                 forecast_to_label.py's
#                                 MIN_CLASS_COUNT_FOR_CALIBRATION)
#   --strategies random         (similarity strategy is more expensive --
#                                 it refits the probe PER QUERY, not once
#                                 per replicate; verify random first)
#   --n-seeds 2                 (Section 3.4 specifies 5 for real results;
#                                 2 is enough to confirm the seed loop and
#                                 bootstrap-CI pooling work end-to-end)
#   --max-patients 60           (RAISED from 10 -- at ~9-10% positive
#                                 rate, ~60 patients should yield roughly
#                                 5-6 positives, enough for roc_auc_score
#                                 to actually compute)
#   --max-train-pool 200        (unchanged -- caps the demonstration pool;
#                                 similarity selection isn't exercised at
#                                 --strategies random, but this keeps pool
#                                 construction itself fast regardless)
#
# k=1 is still expected to show [DEGENERATE PROBE] even with a larger eval
# slice -- that flag is about the DEMONSTRATION draw (a single demo can
# never have 2 classes), not the eval set, so raising --max-patients does
# not fix it. That is an expected, real finding about k=1's Approach-A
# reliability, not a bug to chase.
#
# If AUROC is STILL nan at --max-patients 60, that points to something
# more structural (e.g. the test split itself being smaller/more skewed
# than expected) worth checking directly against shared_split.csv rather
# than just raising the cap again blindly.

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

# Same real, confirmed paths as phase2's jobs -- REPO_ROOT goes up from
# few_shot/, landing at the same shared external/ and splits/ directories
# (both phases sit as siblings under the same repo root).
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
    echo '--- Phase 3 dry run v2: run_few_shot.py (mortality, k in {1,5}, random strategy, 2 seeds, max-patients 60) ---'
    python3 run_few_shot.py \
        --cohort-root '$COHORT_ROOT' \
        --split-assignment '$SPLIT_FILE' \
        --tasks mortality \
        --ks 1 5 \
        --strategies random \
        --n-seeds 2 \
        --max-patients 60 \
        --max-train-pool 200 \
        --output '$SLURM_SUBMIT_DIR/results/phase3_dry_run_v2.json'
"

echo ""
echo "Done. Check results/phase3_dry_run_v2.json for AUROC/AUPRC/ECE per (task,approach,strategy,k)."
echo "Expect AUROC/AUPRC to be real numbers now (not nan), given a larger eval slice."
echo "Check the printed [DEGENERATE PROBE] / [UNCALIBRATED] flags in the job log above --"
echo "[UNCALIBRATED] is EXPECTED and fine (k<=20 essentially never has 5+ per class,"
echo "so Approach A's probe uses LogisticRegression's raw sigmoid, not Platt scaling --"
echo "see forecast_to_label.py's ProbeClassifier); [DEGENERATE PROBE] is still expected"
echo "at k=1 specifically (a single demonstration can never span 2 classes) -- treat"
echo "that as a real finding about k=1, not something this larger eval slice fixes."

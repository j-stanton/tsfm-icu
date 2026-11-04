#!/bin/bash
#SBATCH --job-name=tsfm-few-shot-dry-run
#SBATCH --partition=stampede
#SBATCH -w mscluster40
# ^ same pinned node phase2's jobs used, confirmed working for GPU access
#   (see SETUP_ENVIRONMENT.md). Remove once confirmed to work identically
#   on other stampede nodes.
#
# SUBMIT THIS FROM ~/tsfm-icu-early-warning/few_shot/ (cd there, then
# `sbatch run_dry_run.slurm`) -- $SLURM_SUBMIT_DIR and the script
# references below assume that's where this file and its sibling Phase 3
# files (yaib_data.py, few_shot_pipeline.py, timesfm_icl.py, etc.) live --
# few_shot/ has its OWN copies of the shared modules, not a symlink/import
# into ../zero_shot/ (a deliberate project-structure decision, matching how
# zero_shot/ itself was originally set up as its own folder).
#SBATCH --output=/home-mscluster/%u/tsfm-icu-early-warning/few_shot/results/dry-run-output-%N-%j.txt
#SBATCH --error=/home-mscluster/%u/tsfm-icu-early-warning/few_shot/results/dry-run-error-%N-%j.txt

# Runs run_few_shot.py against REAL YAIB-cohorts output and the REAL
# shared_split.csv, but scoped to a SMALL slice of the full (k x strategy x
# approach x task) factorial -- per the proposal's own Phase 3 milestone
# note: "Dry-runs verify that the prompt format does not silently
# truncate." This is that verification step, not a results run.
#
# Scope deliberately kept small:
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
#   --max-patients 10           (small eval slice)
#   --max-train-pool 200        (caps the demonstration pool -- similarity
#                                 selection isn't exercised at --strategies
#                                 random, but this keeps pool construction
#                                 itself fast regardless)
#
# Once this comes back clean (no crashes, AUROC not NaN for every cell,
# check the printed [DEGENERATE PROBE]/[UNCALIBRATED] flags to understand
# which cells used the small-sample fallback), widen scope incrementally
# (more tasks, both strategies, full k sweep, full 5 seeds) rather than
# jumping straight to an uncapped full run -- same lesson phase2's own
# dry-run-before-full-run progression already established.
#
# EXPECTED RUNTIME: several minutes -- much smaller than phase2's dry run
# in total forecast-call count, but each few-shot query costs MORE than a
# zero-shot one (one extra forecast_batch call per demonstration, to build
# that demonstration's own probe features for approach="probe" -- see
# few_shot_pipeline.py's _fit_probe_on_demos).

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

# Same real, confirmed paths as phase2's jobs -- REPO_ROOT now goes up
# from few_shot/, not zero_shot/, but lands at the same shared external/ and
# splits/ directories (both phases sit as siblings under the same repo
# root, per the original "id rather do this phase in another folder"
# request).
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

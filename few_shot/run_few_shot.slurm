#!/bin/bash
#SBATCH --job-name=tsfm-few-shot-full
#SBATCH --partition=stampede
#SBATCH -w mscluster40
#SBATCH --time=48:00:00
# ^ stampede's real max walltime is 3-00:00:00 (confirmed via
# `sinfo -o "%P %l"`), so 48h is well within range. This is still a GUESS,
# now an informed one: the first full-run attempt (job 61982, task=
# mortality, BOTH strategies, full k-sweep, 5 seeds, no caps) ran the
# ENTIRE 24h walltime and was killed without finishing -- but its output
# file showed ZERO progress lines, so we don't actually know how close it
# got (see the PYTHONUNBUFFERED fix below for why). This version narrows
# scope to ONE strategy per job (see STRATEGY arg) specifically so you get
# a real, visible timing data point rather than another opaque timeout.
#
# SUBMIT THIS FROM ~/tsfm-icu-early-warning/few_shot/, with TASK and
# STRATEGY as two positional arguments:
#   sbatch run_few_shot.slurm mortality random
#   sbatch run_few_shot.slurm mortality similarity
#   sbatch run_few_shot.slurm aki random
#   sbatch run_few_shot.slurm aki similarity
#   sbatch run_few_shot.slurm sepsis random
#   sbatch run_few_shot.slurm sepsis similarity
# One job per (task, strategy) pair, not one job covering everything --
# strategy="similarity" refits Approach A's probe PER EVAL PATIENT (not
# once per replicate the way strategy="random" does -- see
# few_shot_pipeline.py's run_few_shot_evaluation docstring), which is a
# real, likely large, per-patient cost multiplier at higher k that job
# 61982 never got to demonstrate either way. Splitting by strategy means
# a similarity-strategy timeout doesn't also cost you random-strategy's
# (likely much faster) results, and the two can run in parallel.
#SBATCH --output=/home-mscluster/%u/tsfm-icu-early-warning/few_shot/results/full-%j-output-%N.txt
#SBATCH --error=/home-mscluster/%u/tsfm-icu-early-warning/few_shot/results/full-%j-error-%N.txt

# RECOMMENDED ORDER:
#   1. sbatch run_few_shot.slurm mortality random
#      Cheapest task (fixed task_kind, smaller eval set, probe-only -- no
#      rule approach) and the cheaper strategy. This is your first real,
#      visible timing data point -- watch its log fill in live via
#      `tail -f results/full-<jobid>-output-*.txt`, since the
#      PYTHONUNBUFFERED fix below means you should see
#      "[task/approach/strategy/k] starting" and per-100-patient
#      heartbeat lines appear as it runs, not just at the very end.
#   2. Once (1) finishes (or you've watched enough of it to estimate
#      total time), submit mortality/similarity, then aki/random,
#      sepsis/random, then the similarity variants of aki/sepsis last --
#      roughly cheapest-and-most-informative first.
#
# No patient/demo-pool caps here (uncapped --max-patients/--max-train-pool)
# -- this is the real results run, not a dry run. The dry runs (jobs
# 61803, 61805) already confirmed: no crashes on real data, k=1 reliably
# hits [DEGENERATE PROBE] (single demonstration can never span 2 classes
# -- expected, not a bug, and worth noting as a limitation for k=1's
# Approach-A numbers specifically), and k>=5 [UNCALIBRATED] is expected
# and by design (see forecast_to_label.py's
# ProbeClassifier.MIN_CLASS_COUNT_FOR_CALIBRATION) -- both flags will
# appear throughout this run's log; that is normal, not a sign something
# is wrong.

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
    # CRITICAL, added after job 61982 (the first full-run attempt) ran
    # the ENTIRE 24h walltime and was killed with its output file showing
    # ZERO progress lines -- not evidence the job was stuck, but evidence
    # we had no visibility into it either way. Python fully buffers
    # stdout once it's redirected to a file (as Slurm does for #SBATCH
    # --output) rather than a terminal, so print() calls queue up in a
    # buffer that's only flushed on a clean exit or when full -- a job
    # killed by SIGTERM at the walltime limit never gets that flush.
    # PYTHONUNBUFFERED=1 disables that buffering entirely. Combined with
    # run_few_shot.py's per-cell 'starting' line and
    # few_shot_pipeline.py's per-100-candidate heartbeat (both added
    # alongside this fix, both also using flush=True, belt-and-suspenders),
    # this run's log should show real-time progress instead of nothing
    # until the end.

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

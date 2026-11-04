#!/bin/bash
#SBATCH --job-name=train-baselines
#SBATCH --output=/home-mscluster/%u/tsfm-icu-early-warning/baselines/results/result-%N-%A_%a.txt
#SBATCH --error=/home-mscluster/%u/tsfm-icu-early-warning/baselines/results/errors-%N-%A_%a.txt
#SBATCH --partition=stampede
#SBATCH --array=5
#SBATCH --time=14:00:00

# Trains the three classical baselines (LGBMClassifier, LSTM, LogisticRegression)
# across the three real cohorts (mortality24, aki, sepsis) = 9 combinations,
# as a SLURM array -- one job per (task, model) pair, indexed 0-8.
#
# REVISION NOTE: an earlier version of this script used `uv run`, which is
# not installed on this cluster (confirmed: command not found, no root to
# install system-wide). This version uses `conda run -n yaib`, following
# this cluster's own documented pattern (see "Introductory HPC Examples --
# Anaconda With Local Environment"), with the yaib environment built via
# `pip install -e .` against external/YAIB/pyproject.toml (installed
# interactively via `srun --pty bash`, per that same doc's etiquette
# guidance -- NOT inside a batch job).
#
# ALSO NOTE: the yaib environment required polars-lts-cpu instead of
# standard polars -- standard polars requires AVX2/FMA/BMI1/BMI2/etc CPU
# instructions that at least one compute node on this cluster (mscluster23)
# does not have, causing an "Illegal instruction" crash. This is fixed at
# the package level (already installed into the yaib env), so it should
# hold regardless of which node this job lands on -- but if a future
# "Illegal instruction" crash reappears, this is the first thing to check.
#
# WHY apply_fixed_split.py MUST BE IMPORTED INSIDE THE SAME `conda run`
# CALL: the split patch is a monkeypatch on
# icu_benchmarks.data.split_process_data.make_single_split_polars, which
# only exists in memory for the lifetime of the Python process it's
# imported into. `conda run -n yaib python -c "..."` runs one Python
# process per array task, so the patch import and the training call must
# both happen inside that single `python -c` block -- there is no way to
# "pre-apply" the split in an earlier job and have this one inherit it.
# The same reasoning applies to fix_class_weight_bug.py, added below.
#
# WHY fix_class_weight_bug.py WAS ADDED: the first full 9-way run completed
# successfully, but LogisticRegression landed at near-chance AUC (0.49-0.55)
# on all three tasks while LGBMClassifier/LSTM were correctly in the
# 0.81-0.89 range on the same data/split. Traced to a real bug in YAIB's
# own MLWrapper.fit(): it reads class_weight from self.weight, which is
# never assigned anywhere in the codebase (always None) -- the actual
# computed balance weights land in a differently-named attribute,
# self.loss_weights, intended for the DL loss-weighting path instead. See
# fix_class_weight_bug.py's own docstring for the full trace.
#
# RERUN HISTORY (this script has been resubmitted several times for the
# LogisticRegression indices specifically -- keeping this history here
# rather than only in chat, since it explains choices below that would
# otherwise look arbitrary):
#   - Index 8 (Sepsis) first attempt: crashed on FileExistsError, a stale
#     run directory (same-minute timestamp collision) from an earlier
#     attempt. Fixed by deleting the stale yaib_logs directory before
#     resubmitting.
#   - Index 2 (Mortality24): completed with the class-weight fix applied
#     (confirmed via the "[fix_class_weight_bug] Set class_weight=
#     'balanced'" log line) -- but AUC only improved from ~0.49 to 0.567,
#     still far below LGBM/LSTM's 0.81-0.89 range. The class-weight bug
#     was real and worth fixing, but is evidently NOT the sole explanation
#     for LogisticRegression's weak performance here.
#   - Index 5 (AKI): hung for 21+ hours with EMPTY error logs (no output
#     at all, not even the initial "Found tasks:"/"Data directory:" lines
#     every other run -- including this same task/model on other days --
#     produced within seconds of starting). This looks like a stall
#     upstream of Python's own logging (node contention, a stale/locked
#     cache file under YAIB-cohorts' data/aki/mimic/cache, or similar),
#     not a slow-converging solver deep in training (which would still
#     produce log output). Cancelled manually; --time=03:00:00 added
#     below so a repeat of this specific failure mode gets killed and
#     flagged by SLURM automatically rather than silently running for a
#     day+ again.
#
# CURRENT PLAN: this array is currently scoped to index 5 (AKI) ONLY, as a
# bounded sanity-check rerun -- confirm it completes normally within the
# 3-hour cap once any stale cache/lock state is cleared. If it succeeds
# cleanly, the working theory is that the original hang was a one-off
# (node-specific or transient), and the next step is a SEPARATE run of
# indices 2, 5, and 8 together WITH --tune enabled (see the TUNE_FLAG
# variable below), since index 2's result shows the untuned config isn't
# good enough on its own -- the .gin config already defines a real
# hyperparameter search space (C, penalty, l1_ratio) that --no-tune
# currently skips entirely in favour of sklearn's bare defaults.
#
# PREREQUISITES:
#   1. yaib conda environment set up and verified (icu-benchmarks --help
#      works) -- confirmed done as of writing this script.
#   2. shared_split.csv (patient-level, verified zero cross-split overlap)
#      present on this cluster.
#   3. apply_fixed_split.py present on this cluster, alongside shared_split.csv.
#   4. fix_class_weight_bug.py present on this cluster, alongside the above.
#   5. Real cohort output (mortality24/aki/sepsis, --src mimic) present
#      under YAIB-cohorts' data directory on this cluster.
#   6. Before running THIS rerun specifically: clear any stale AKI cache/
#      run-directory state left over from the cancelled 21-hour job --
#      see the cache-clearing check added below, which warns (but does not
#      block) if leftover state is found, so you can inspect before it's
#      silently reused.
#
# EDIT THESE PATHS if your cluster layout differs from what's assumed here:
REPO_ROOT="$HOME/tsfm-icu-early-warning"
YAIB_DIR="$REPO_ROOT/external/YAIB"
COHORTS_DATA_DIR="$REPO_ROOT/external/YAIB-cohorts/data"
SPLIT_CONTROL_DIR="$REPO_ROOT/splits"          # where apply_fixed_split.py + shared_split.csv live
SPLIT_CSV="$SPLIT_CONTROL_DIR/shared_split.csv"
LOG_DIR="$REPO_ROOT/yaib_logs"

# Set to "--tune" to enable Optuna hyperparameter search (see CURRENT PLAN
# above -- deliberately NOT the default yet, since this rerun's purpose is
# first confirming AKI/LogisticRegression completes at all within the time
# cap, before spending the extra time --tune requires). Change to "--tune"
# once index 5 has completed cleanly and you're ready for the follow-up
# tuned rerun of indices 2, 5, 8 together.
TUNE_FLAG="--no-tune"

mkdir -p "$REPO_ROOT/baselines/results" "$LOG_DIR"

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
echo "Array Task ID: "$SLURM_ARRAY_TASK_ID
echo "======"

# Task/model grid, indexed 0-8. Task directory names match YAIB-cohorts'
# real output structure (mortality uses "mortality24", confirmed from
# source months ago); task-name (-tn) values for AKI and Sepsis are
# confirmed working from earlier successful runs; Mortality24 is inferred
# by the same naming convention but NOT yet independently confirmed --
# if array index 0/1/2 fails specifically on an unrecognized -tn value,
# this is the first thing to check.
TASK_DIRS=(mortality24 mortality24 mortality24 aki aki aki sepsis sepsis sepsis)
TASK_NAMES=(Mortality24 Mortality24 Mortality24 AKI AKI AKI Sepsis Sepsis Sepsis)
MODELS=(LGBMClassifier LSTM LogisticRegression LGBMClassifier LSTM LogisticRegression LGBMClassifier LSTM LogisticRegression)

TASK_DIR="${TASK_DIRS[$SLURM_ARRAY_TASK_ID]}"
TASK_NAME="${TASK_NAMES[$SLURM_ARRAY_TASK_ID]}"
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "Array task $SLURM_ARRAY_TASK_ID -> task_dir=$TASK_DIR task_name=$TASK_NAME model=$MODEL"

DATA_DIR="$COHORTS_DATA_DIR/$TASK_DIR/mimic"

if [ ! -d "$DATA_DIR" ]; then
    echo "FATAL: $DATA_DIR does not exist. Confirm real cohort output has"
    echo "been generated and is present on this cluster before running this job."
    exit 1
fi

if [ ! -f "$SPLIT_CSV" ]; then
    echo "FATAL: $SPLIT_CSV not found. Copy the patient-level shared_split.csv"
    echo "to this cluster before running this job."
    exit 1
fi

if [ ! -f "$SPLIT_CONTROL_DIR/apply_fixed_split.py" ]; then
    echo "FATAL: apply_fixed_split.py not found at $SPLIT_CONTROL_DIR."
    exit 1
fi

if [ ! -f "$SPLIT_CONTROL_DIR/fix_class_weight_bug.py" ]; then
    echo "FATAL: fix_class_weight_bug.py not found at $SPLIT_CONTROL_DIR."
    echo "Copy it there before running this job -- it fixes a real upstream"
    echo "YAIB bug (self.weight vs self.loss_weights attribute mismatch) that"
    echo "was silently preventing class_weight='balanced' from reaching"
    echo "LogisticRegression, producing near-chance AUC on the imbalanced"
    echo "mortality/sepsis/AKI tasks. See that file's own docstring for the"
    echo "full explanation."
    exit 1
fi

# Warn (don't auto-delete) if a run directory already exists for this exact
# (task, model) combination -- this is what caused the Sepsis FileExistsError
# earlier. Not blocking automatically since a same-day rerun with a fresh
# timestamp is normally fine; this is here so a genuinely stale/incomplete
# directory doesn't go unnoticed if a collision happens again.
EXISTING_RUN_DIRS=$(find "$LOG_DIR/mimic/$TASK_NAME/$MODEL" -maxdepth 1 -mindepth 1 -type d 2>/dev/null)
if [ -n "$EXISTING_RUN_DIRS" ]; then
    echo "NOTE: existing run director(ies) found for $TASK_NAME/$MODEL:"
    echo "$EXISTING_RUN_DIRS"
    echo "These are timestamped and a fresh run should not collide with them,"
    echo "but if this job fails with FileExistsError again, check whether one"
    echo "of these has the SAME timestamp as this job's start time and delete"
    echo "it manually before resubmitting."
fi

# Warn if a cache file exists for this task -- relevant specifically because
# the AKI/LogisticRegression hang (21+ hours, zero log output) is suspected
# to possibly involve a stale/locked cache artifact rather than a genuinely
# slow model fit. Not deleted automatically since --load_cache reusing a
# GOOD cache is a real speed win worth keeping when it's not the problem.
EXISTING_CACHE=$(find "$DATA_DIR/cache" -maxdepth 1 -type f 2>/dev/null)
if [ -n "$EXISTING_CACHE" ]; then
    echo "NOTE: existing cache file(s) found under $DATA_DIR/cache:"
    echo "$EXISTING_CACHE"
    echo "If this job hangs with no log output again, these are worth"
    echo "clearing manually and retrying before assuming a deeper problem:"
    echo "  rm -f $DATA_DIR/cache/*"
fi

module load anaconda/anaconda-2025-12-2

# MAJOR FIX: this script previously used `conda run -n yaib python -c "..."`
# throughout. `conda run` was confirmed, via direct interactive testing, to
# intermittently HANG INDEFINITELY on this cluster -- not the Python code
# inside it, not YAIB, not AKI's data. Confirmed by isolating the exact
# same python -c invocation two ways: via `conda run -n yaib python -c "..."`
# (hung for over an hour producing zero output, not even a trivial
# print() on the very first line) versus calling the environment's own
# python binary directly, bypassing `conda run` entirely (printed
# instantly, every time tested). This explains every "hang" seen across
# multiple earlier attempts (21+ hours once, 3-hour timeout once, 1+ hour
# once) -- they were very likely all `conda run` itself failing to launch
# the process, not anything about AKI, LogisticRegression, or this
# project's own code.
#
# Fix: resolve the yaib environment's bin/ directory directly (same
# pattern already used for the R/ricu conda environment, since this
# cluster's anaconda module doesn't expose conda.sh at the location
# `conda activate` expects) and call "$YAIB_ENV_BIN/python" directly,
# never `conda run` again in this script.
YAIB_ENV_BIN="$(conda env list | awk '/^yaib / {print $NF"/bin"}')"

if [ -z "$YAIB_ENV_BIN" ] || [ ! -x "$YAIB_ENV_BIN/python" ]; then
    echo "FATAL: could not resolve yaib environment's python binary."
    echo "conda env list output for diagnosis:"
    conda env list
    exit 1
fi
echo "Resolved yaib env python: $YAIB_ENV_BIN/python"

# CRITICAL: YAIB's own run.py resolves task/model config files via a BARE
# RELATIVE PATH -- get_config_files(Path("configs")) in run.py, which globs
# configs/tasks/*.gin and configs/prediction_models/*.gin relative to
# whatever the current working directory happens to be. Confirmed directly
# from source (icu_benchmarks/run.py line 65, run_utils.py's
# get_config_files). Without cd'ing into $YAIB_DIR first, this resolves
# against $SLURM_SUBMIT_DIR instead (wherever sbatch was run from), finds
# nothing, and -- because get_config_files silently swallows any exception
# and returns ([], []) rather than raising -- produces the misleading
# "Found tasks: []" / "Found models: []" log lines with no indication of
# what actually went wrong. This is the real root cause of the
# 'Invalid task or model... not found' error from the previous run.
cd "$YAIB_DIR"

export CUDA_VISIBLE_DEVICES=""
"$YAIB_ENV_BIN/python" -c "
import sys
sys.path.insert(0, '$SPLIT_CONTROL_DIR')
import apply_fixed_split
apply_fixed_split.patch(split_path='$SPLIT_CSV')

import fix_class_weight_bug
fix_class_weight_bug.patch()

from icu_benchmarks.run import main

# IMPORTANT: main()'s my_args parameter has a default value of
# tuple(sys.argv[1:]), which -- because Python default arguments are
# evaluated once at function DEFINITION time, not call time -- was already
# bound to whatever sys.argv looked like when icu_benchmarks.run was
# imported (effectively empty/irrelevant here, since we're running via
# a direct python -c invocation). Setting sys.argv AFTER that import, as an
# earlier version of this script did, has no effect on that already-bound
# default -- confirmed by the real failure this fixes: 'the following
# arguments are required: -d/--data-dir, -t/--task' despite sys.argv
# clearly containing both. Passing my_args directly to main() sidesteps
# the default entirely and is the correct fix.
my_args = [
    '-d', '$DATA_DIR',
    '-n', 'mimic',
    '-t', 'BinaryClassification',
    '-tn', '$TASK_NAME',
    '-m', '$MODEL',
    '--generate_cache', '--load_cache',
    '-s', '2222',
    '-l', '$LOG_DIR',
    '--cpu',
    '$TUNE_FLAG',
]
main(my_args=my_args)
"

echo "Done: $TASK_NAME / $MODEL (array task $SLURM_ARRAY_TASK_ID)."
echo "Check the log above for the 'Fixed split applied: train=X, val=Y, test=Z'"
echo "line to confirm the patch actually took effect -- if that line is"
echo "missing, the split patch did not apply and results should not be trusted."

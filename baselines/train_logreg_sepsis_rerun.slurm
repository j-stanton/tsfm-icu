#!/bin/bash
#SBATCH --job-name=logreg-sepsis
#SBATCH --output=/home-mscluster/%u/tsfm-icu-early-warning/baselines/results/logreg-sepsis-%N-%j.txt
#SBATCH --error=/home-mscluster/%u/tsfm-icu-early-warning/baselines/results/logreg-sepsis-err-%N-%j.txt
#SBATCH --partition=stampede
#SBATCH --time=24:00:00
#
# SUBMIT FROM ANYWHERE:  sbatch ~/tsfm-icu-early-warning/baselines/train_logreg_sepsis_rerun.slurm
#
# Reruns Sepsis / LogisticRegression ONCE (repetition 0, fold 0), with the
# class-weight fix applied, instead of YAIB's 5x5 nested CV (25 fits, too slow
# for this model: the 14 h attempt did not finish fold 0).
#
# DIFFERENCES FROM THE OTHER 24 BASELINE RUNS (state these in the report):
#   - one repetition x one fold instead of 5 x 5
#   - saga max_iter capped at $MAX_ITER (default 1000) instead of 100000, because
#     saga with elasticnet/l1 and balanced class weights does not converge in
#     reasonable time on ~millions of stay-hours. Override: MAX_ITER=5000 sbatch ...
#   - everything else identical: seed 2222, shared split, --no-tune (one random
#     hyper-parameter draw), class_weight='balanced'
#
# A fresh log directory per job avoids the earlier FileExistsError.

MAX_ITER="${MAX_ITER:-1000}"
REPO_ROOT="$HOME/tsfm-icu-early-warning"
YAIB_DIR="$REPO_ROOT/external/YAIB"
DATA_DIR="$REPO_ROOT/external/YAIB-cohorts/data/sepsis/mimic"
SPLIT_CONTROL_DIR="$REPO_ROOT/splits"
SPLIT_CSV="$SPLIT_CONTROL_DIR/shared_split.csv"
LOG_DIR="$REPO_ROOT/yaib_logs_sepsis_lr_${SLURM_JOB_ID:-manual}"

mkdir -p "$REPO_ROOT/baselines/results" "$LOG_DIR"
sleep 20
echo "Date: $(date)  Job: $SLURM_JOB_ID  Node: $HOSTNAME  MAX_ITER=$MAX_ITER  LOG_DIR=$LOG_DIR"

for f in "$DATA_DIR" "$SPLIT_CSV" "$SPLIT_CONTROL_DIR/apply_fixed_split.py" "$SPLIT_CONTROL_DIR/fix_class_weight_bug.py"; do
    [ -e "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

module load anaconda/anaconda-2025-12-2
YAIB_ENV_BIN="$(conda env list | awk '/^yaib / {print $NF"/bin"}')"
if [ -z "$YAIB_ENV_BIN" ] || [ ! -x "$YAIB_ENV_BIN/python" ]; then
    echo "FATAL: could not resolve yaib env python"; conda env list; exit 1
fi

cd "$YAIB_DIR"     # YAIB resolves configs/ relative to the working directory
export CUDA_VISIBLE_DEVICES=""

"$YAIB_ENV_BIN/python" -c "
import sys
sys.path.insert(0, '$SPLIT_CONTROL_DIR')
import apply_fixed_split
apply_fixed_split.patch(split_path='$SPLIT_CSV')
import fix_class_weight_bug
fix_class_weight_bug.patch()
from icu_benchmarks.run import main
main(my_args=[
    '-d', '$DATA_DIR', '-n', 'mimic', '-t', 'BinaryClassification',
    '-tn', 'Sepsis', '-m', 'LogisticRegression',
    '--generate_cache', '--load_cache', '-s', '2222', '-l', '$LOG_DIR', '--cpu',
    '--no-tune',
    '-hp',
    'execute_repeated_cv.cv_repetitions_to_train=1',
    'execute_repeated_cv.cv_folds_to_train=1',
    'model/hyperparameter.max_iter=$MAX_ITER',
])
"
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "FAILED (exit status $STATUS) -- do NOT use any numbers from this job."
    exit $STATUS
fi
echo "Done: Sepsis / LogisticRegression (1 repetition x 1 fold)."
echo "Check above for 'Starting nested CV with 1 repetitions of 1 folds' and 'Set class_weight=balanced'."
echo "Results: find $LOG_DIR -name 'aggregated_test_metrics.json' -o -name 'test_metrics.json'"

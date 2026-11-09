#!/bin/bash
#SBATCH --job-name=train-baselines
#SBATCH --output=/home-mscluster/%u/tsfm-icu-early-warning/baselines/results/result-%N-%A_%a.txt
#SBATCH --error=/home-mscluster/%u/tsfm-icu-early-warning/baselines/results/errors-%N-%A_%a.txt
#SBATCH --partition=stampede
#SBATCH --array=2,5,8
# Rerun logistic regression after the class-weight fix.

REPO_ROOT="$HOME/tsfm-icu-early-warning"
YAIB_DIR="$REPO_ROOT/external/YAIB"
COHORTS_DATA_DIR="$REPO_ROOT/external/YAIB-cohorts/data"
SPLIT_CONTROL_DIR="$REPO_ROOT/splits"          # where apply_fixed_split.py + shared_split.csv live
SPLIT_CSV="$SPLIT_CONTROL_DIR/shared_split.csv"
LOG_DIR="$REPO_ROOT/yaib_logs"

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

module load anaconda/anaconda-2025-12-2

cd "$YAIB_DIR"

conda run -n yaib python -c "
import sys
sys.path.insert(0, '$SPLIT_CONTROL_DIR')
import apply_fixed_split
apply_fixed_split.patch(split_path='$SPLIT_CSV')

import fix_class_weight_bug
fix_class_weight_bug.patch()

from icu_benchmarks.run import main

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
    '--no-tune',
]
main(my_args=my_args)
"

echo "Done: $TASK_NAME / $MODEL (array task $SLURM_ARRAY_TASK_ID)."
echo "Check the log above for the 'Fixed split applied: train=X, val=Y, test=Z'"
echo "line to confirm the patch actually took effect -- if that line is"
echo "missing, the split patch did not apply and results should not be trusted."

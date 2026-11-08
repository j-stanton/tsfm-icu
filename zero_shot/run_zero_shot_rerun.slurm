#!/bin/bash
#SBATCH --job-name=tsfm-zs-rerun
#SBATCH --partition=stampede
#SBATCH --output=/home-mscluster/%u/tsfm-icu-early-warning/zero_shot/results/rerun-output-%N-%j.txt
#SBATCH --error=/home-mscluster/%u/tsfm-icu-early-warning/zero_shot/results/rerun-error-%N-%j.txt
#
# SUBMIT FROM ~/tsfm-icu-early-warning/zero_shot/ :   sbatch run_zero_shot_rerun.slurm
#
# Reruns the zero-shot evaluation ONCE (zero-shot is deterministic, so extra
# seeds only duplicate rows) and saves per-patient predictions, then repeats
# with a persistence forecaster (last value repeated) as an ablation. Same
# cohorts, split, probe and rules as run_zero_shot.slurm.
#
# Outputs (in zero_shot/results/):
#   predictions/timesfm_<task>_<approach>_seed0.csv
#   predictions/persistence_<task>_<approach>_seed0.csv
#   rerun_timesfm.json, rerun_persistence.json

sleep 20
echo "Date: $(date)  Job: $SLURM_JOB_ID  Node: $HOSTNAME  Dir: $SLURM_SUBMIT_DIR"

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/predictions

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

    for t in mortality24 aki sepsis; do
        test -f '$COHORT_ROOT/'\$t'/mimic/dyn.parquet' || { echo \"MISSING: \$t/mimic/dyn.parquet\"; exit 1; }
    done
    test -f '$SPLIT_FILE' || { echo 'MISSING: shared_split.csv'; exit 1; }

    echo '--- 1/2 TimesFM, 1 seed, saving predictions ---'
    python3 run_zero_shot.py \
        --cohort-root '$COHORT_ROOT' --split-assignment '$SPLIT_FILE' \
        --tasks mortality aki sepsis --n-seeds 1 --forecaster timesfm \
        --predictions-dir '$SLURM_SUBMIT_DIR/results/predictions' \
        --output '$SLURM_SUBMIT_DIR/results/rerun_timesfm.json'

    echo '--- 2/2 Persistence ablation, 1 seed ---'
    python3 run_zero_shot.py \
        --cohort-root '$COHORT_ROOT' --split-assignment '$SPLIT_FILE' \
        --tasks mortality aki sepsis --n-seeds 1 --forecaster persistence \
        --predictions-dir '$SLURM_SUBMIT_DIR/results/predictions' \
        --output '$SLURM_SUBMIT_DIR/results/rerun_persistence.json'
"
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "FAILED with exit status $STATUS"
    exit $STATUS
fi
echo "Done OK. See results/rerun_timesfm.json, results/rerun_persistence.json and results/predictions/"

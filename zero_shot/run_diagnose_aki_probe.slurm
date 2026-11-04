#!/bin/bash
#SBATCH --job-name=diagnose-aki-probe
#SBATCH --partition=stampede
# No -w mscluster40 pin here deliberately: this is a quick diagnostic, not
# a job that needs a specific pre-confirmed node -- pinning would only add
# unnecessary queue wait if that one node happens to be busy. stampede's
# nodes are documented (mscluster "Community Guidelines" PDF, Section 5.1)
# to uniformly have 2x GTX1060 each, so any node on this partition should
# work; if it doesn't, that's itself useful information (would mean GPU
# access isn't as uniform across stampede as assumed).
#
# SUBMIT THIS FROM ~/tsfm-icu-early-warning/zero_shot/ (cd there, then
# `sbatch run_diagnose_aki_probe.slurm`).
#SBATCH --output=/home-mscluster/%u/tsfm-icu-early-warning/zero_shot/results/diagnose-aki-output-%N-%j.txt
#SBATCH --error=/home-mscluster/%u/tsfm-icu-early-warning/zero_shot/results/diagnose-aki-error-%N-%j.txt

# Runs diagnose_aki_probe.py as a batch job instead of interactively --
# interactive `srun --pty bash` queue wait was too long for a quick check.
# Same apptainer+conda pattern as the other real-data jobs in this
# project (run_real_dry_run.slurm, run_smoke_test.slurm).

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

# Same non-interactive apptainer+conda pattern as run_real_dry_run.slurm --
# see SETUP_ENVIRONMENT.md for why this specific sequence is needed.
#
# SIF_PATH: same caveat as the other .slurm jobs in this project --
# inferred, not yet directly confirmed against this exact module version.
SIF_PATH="/usr/local/containers/pytorch/pytorch-cuda12.9-ubuntu24.04-00.sif"

apptainer exec --nv "$SIF_PATH" bash -c "
    source /opt/conda/etc/profile.d/conda.sh
    conda activate tsfm-adapter

    set -e

    cd $SLURM_SUBMIT_DIR
    python3 diagnose_aki_probe.py
"

echo ""
echo "Done. Check results/diagnose-aki-output-*.txt for the per-patient"
echo "(true_label, probe_score) table and the inversion-check summary."

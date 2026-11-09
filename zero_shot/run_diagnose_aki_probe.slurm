#!/bin/bash
#SBATCH --job-name=diagnose-aki-probe
#SBATCH --partition=stampede
#SBATCH --output=/home-mscluster/%u/tsfm-icu-early-warning/zero_shot/results/diagnose-aki-output-%N-%j.txt
#SBATCH --error=/home-mscluster/%u/tsfm-icu-early-warning/zero_shot/results/diagnose-aki-error-%N-%j.txt
# Run diagnose_aki_probe.py.


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

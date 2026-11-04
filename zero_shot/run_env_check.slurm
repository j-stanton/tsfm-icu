#!/bin/bash
#SBATCH --job-name=tsfm-env-check
#SBATCH --partition=stampede
#SBATCH -w mscluster40
# ^ pinned to a node CONFIRMED working (nvidia-smi shows 2x GTX1060,
#   torch.cuda.is_available()==True via the pytorch module) -- remove this
#   line once the pytorch module is confirmed to work the same way on
#   other stampede nodes, so real Phase 2 jobs aren't artificially
#   bottlenecked onto one specific node.
#
# SUBMIT THIS FROM ~/tsfm-icu-early-warning/zero_shot/ (i.e. `cd` there first,
# then `sbatch run_env_check.slurm`) -- $SLURM_SUBMIT_DIR below and the
# check_cluster_env.py reference both assume that's where this script and
# its sibling pipeline files (yaib_data.py, zero_shot_pipeline.py, etc.) live,
# separate from ~/tsfm-icu-early-warning/baselines/'s earlier env-debugging
# scratch work.
#SBATCH --output=/home-mscluster/%u/tsfm-icu-early-warning/zero_shot/results/output-%N-%j.txt
#SBATCH --error=/home-mscluster/%u/tsfm-icu-early-warning/zero_shot/results/error-%N-%j.txt

# Uses the cluster's own pytorch module, NOT a hand-built conda env --
# see SETUP_ENVIRONMENT.md for the full story on why. Structure otherwise
# still follows the cluster's documented example
# (default-anaconda-with-local-env.slurm).
#
# GPU note (CONFIRMED working 2026-09-09, see SETUP_ENVIRONMENT.md):
# pytorch/pytorch-cuda12.9-ubuntu24.04-00's own `module spider` help text
# says it "Automatically uses 'apptainer exec --nv' when NVIDIA devices
# are detected" -- this is the module that actually gets GPU passthrough
# right, unlike the plain `anaconda` module (no --nv, torch.cuda.is_available()
# was always False through that path, regardless of packages installed).
# No --gres=gpu request needed or supported -- confirmed via
# `scontrol show node` / `sinfo -o "%N %G"` that stampede has no GRES
# configured at all (Gres=(null) cluster-wide).

# We pause execution briefly, so you get a chance to see you job in the queue.
# If you don't see your job in the queue at all, you know that execution has not even started,
# and there is likely something wrong with your #SBATCH lines above.

#sleep 20

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


# We now run our code.

cd $SLURM_SUBMIT_DIR
mkdir -p results

source /etc/profile.d/lmod.sh
module load pytorch/pytorch-cuda12.9-ubuntu24.04-00


python3 -c "import sys; print(sys.version)"


# IMPORTANT: unlike an interactive `srun --pty bash` session, a batch job
# has no interactive shell to drop into an `Apptainer>` prompt and run
# multi-step activation commands by hand. Instead, invoke apptainer
# directly with --nv (GPU passthrough, see SETUP_ENVIRONMENT.md) and pass
# the whole activation+run sequence as a single non-interactive command
# string, since that's what actually reaches the writable `tsfm-adapter`
# env (cloned from the module's read-only `ai` base) where timesfm is
# installed -- plain `python3 check_cluster_env.py` after `module load`
# alone only sees the base `ai` env, not `tsfm-adapter`.
#
# SIF_PATH: confirmed via the module's own conda-activate wrapper script
# and this project's hpc-examples/06-default-pytorch commented-out line
# (`apptainer exec --nv /usr/local/containers/pytorch/pytorch-2026-02-24.sif ...`)
# -- adjust the filename here if `module spider
# pytorch/pytorch-cuda12.9-ubuntu24.04-00` or `ls
# /usr/local/containers/pytorch/` shows a different .sif filename actually
# matching this module version; not yet directly confirmed against this
# exact module's .sif path as of this script's last update.
SIF_PATH="/usr/local/containers/pytorch/pytorch-cuda12.9-ubuntu24.04-00.sif"

apptainer exec --nv "$SIF_PATH" bash -c "
    source /opt/conda/etc/profile.d/conda.sh
    conda activate tsfm-adapter
    python3 $SLURM_SUBMIT_DIR/check_cluster_env.py
"

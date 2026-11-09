#!/bin/bash
#SBATCH --job-name=build-conda-env
#SBATCH --output=/home-mscluster/%u/tsfm-icu-early-warning/environment/results/result-%N-%j.txt
#SBATCH --error=/home-mscluster/%u/tsfm-icu-early-warning/environment/results/errors-%N-%j.txt
#SBATCH --partition=stampede
# Build the conda environments used on the cluster.


REPO_DIR="$HOME/tsfm-icu-early-warning"

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

module load anaconda/anaconda-2025-12-2

CONDA_BASE="$(conda info --base)"
echo "Reported conda base: $CONDA_BASE"

YAIB_ENV_BIN="$(conda env list | awk '/yaib_env/ {print $NF"/bin"}')"

if ! conda env list | grep -q "^yaib_env "; then
    echo "Creating yaib_env ..."
    conda create -n yaib_env -c conda-forge r-base=4.2.2 -y
    # Re-resolve after creation, in case the path only becomes valid now.
    YAIB_ENV_BIN="$(conda env list | awk '/yaib_env/ {print $NF"/bin"}')"
else
    echo "yaib_env already exists, skipping creation."
fi

if [ -z "$YAIB_ENV_BIN" ] || [ ! -d "$YAIB_ENV_BIN" ]; then
    echo "FATAL: could not determine yaib_env's actual bin directory."
    echo "Full conda env list output for diagnosis:"
    conda env list
    exit 1
fi
echo "Resolved yaib_env bin directory: $YAIB_ENV_BIN"

echo "Checking for R directly at $YAIB_ENV_BIN/R ..."
if [ ! -x "$YAIB_ENV_BIN/R" ]; then
    echo "FATAL: R not found at $YAIB_ENV_BIN/R even after conda create."
    echo "Listing $YAIB_ENV_BIN for diagnosis:"
    ls -la "$YAIB_ENV_BIN" 2>&1 | head -50
    exit 1
fi

export PATH="$YAIB_ENV_BIN:$PATH"

echo "R version check:"
R --version

conda install -n yaib_env -c conda-forge --strict-channel-priority \
    udunits2 r-arrow python=3.10 rpy2 -y

echo "Package versions after full install:"
conda list -n yaib_env | grep -E "^r-base|^udunits2|^r-arrow|^python |^rpy2"

if [ ! -d "$REPO_DIR/YAIB-cohorts" ]; then
    echo "Cloning YAIB-cohorts into $REPO_DIR ..."
    git clone https://github.com/rvandewater/YAIB-cohorts.git "$REPO_DIR/YAIB-cohorts"
fi

cd "$REPO_DIR/YAIB-cohorts/R"

echo "Running renv::restore(exclude = 'arrow') instead of setup_env.R directly ..."

if ! Rscript -e 'if (!require("renv")) install.packages("renv"); renv::activate(); renv::restore(exclude = "arrow", prompt = FALSE)'; then
    echo "FATAL: renv::restore() failed even with arrow excluded. Check the"
    echo "output above for which package failed this time -- do not trust"
    echo "any 'Done' message below this point, since this script now stops"
    echo "here instead of continuing past a real failure."
    exit 1
fi

echo "Sanity check: confirming ricu AND arrow both actually load, and"
echo "reporting versions."
if ! Rscript -e '
renv::activate()
library(ricu)
cat("ricu version:", as.character(packageVersion("ricu")), "\n")
cat("data.table version:", as.character(packageVersion("data.table")), "\n")
library(arrow)
cat("arrow version (conda-installed, NOT the renv.lock-pinned 22.0.0):",
    as.character(packageVersion("arrow")), "\n")
'; then
    echo "FATAL: ricu or arrow did not load successfully. The environment is"
    echo "NOT ready for cohort generation -- see the R error above for"
    echo "specifics. If arrow specifically failed here, the conda-installed"
    echo "version may not actually be usable from within this renv project"
    echo "library after all -- report back with the exact error."
    exit 1
fi

echo "Done. ricu AND arrow both loaded successfully -- confirm the printed"
echo "ricu/data.table versions above match monkeypatch0.6.0 and 1.14.8"
echo "respectively before treating the environment as fully ready. The"
echo "arrow version will correctly show 12.0.1 (conda's), not the"
echo "renv.lock-pinned 22.0.0 -- this is expected and intentional given"
echo "the exclude above, not a sign of something still being wrong."

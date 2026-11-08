#!/bin/bash
#SBATCH --job-name=build-conda-env
#SBATCH --output=/home-mscluster/%u/tsfm-icu-early-warning/environment/results/result-%N-%j.txt
#SBATCH --error=/home-mscluster/%u/tsfm-icu-early-warning/environment/results/errors-%N-%j.txt
#SBATCH --partition=stampede

# Builds the R 4.2.2 environment via conda-forge and then runs YAIB-cohorts' own documented setup_env.R against it.

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

# IMPORTANT, confirmed from an actual failure on this cluster: environments
# do NOT necessarily live under $CONDA_BASE/envs/. The very first successful
# `conda create` run printed:
#   environment location: /home-mscluster/jstanton/.conda/envs/yaib_env
# i.e. under the user's own home directory, NOT under /opt/anaconda/envs/
# (which doesn't exist at all -- likely because /opt/anaconda is a shared,
# read-only module install, so conda redirects new environments to the
# user's writable ~/.conda/envs/ instead, while `conda info --base` still
# reports the module's own install location for the base itself).
#
# Rather than hardcode either location and risk being wrong again, ask
# conda directly where THIS specific environment actually lives.
YAIB_ENV_BIN="$(conda env list | awk '/yaib_env/ {print $NF"/bin"}')"

# Step 1: create the env fresh if it doesn't already exist, so this job
# script is safely re-runnable without manual cleanup. Note: conda create/
# install/list all accept -n/--name directly and do NOT require the
# environment to be "activated" first -- this is why these worked
# correctly in earlier runs even though activation itself was broken.
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

# Put the env's bin directory at the FRONT of PATH -- this achieves the
# same practical effect as `conda activate` for the purposes of this
# script (R, Rscript, python, pip all resolve to the env's versions),
# without depending on conda's shell-integration files at all.
export PATH="$YAIB_ENV_BIN:$PATH"

echo "R version check:"
R --version

# Step 4: remaining system-level libraries ricu/YAIB-cohorts need, all via
# conda-forge rather than apt -- udunits2 (for the R `units` package),
# r-arrow (Arrow's own docs confirm this single package installs BOTH the
# Arrow C++ library and the matched R `arrow` package together -- an
# earlier draft of this script tried installing the bare C++ library
# arrow-cpp separately and hoping renv::restore() would find/match it
# later, which is both unnecessary and risky: arrow-cpp on conda-forge
# appears to be a legacy/frozen package name at a much older version than
# current Arrow releases, so it could easily mismatch whatever version
# renv.lock expects), and a Python + rpy2 side since the task scripts
# you'll actually run (mortality.py etc.) are Python calling into R.
#
# --strict-channel-priority avoids conda mixing packages from the
# "defaults" channel with conda-forge ones, which Arrow's own install docs
# specifically warn can cause solve issues/incompatible mixes.
conda install -n yaib_env -c conda-forge --strict-channel-priority \
    udunits2 r-arrow python=3.10 rpy2 -y

echo "Package versions after full install:"
conda list -n yaib_env | grep -E "^r-base|^udunits2|^r-arrow|^python |^rpy2"

# Step 5: clone the repo (if not already present) and run its own,
# documented setup script -- same renv.lock, same setup_env.R we already
# verified pins ricu/data.table/units correctly (see earlier discussion),
# just running inside this conda-provided R 4.2.2 instead of a container.
if [ ! -d "$REPO_DIR/YAIB-cohorts" ]; then
    echo "Cloning YAIB-cohorts into $REPO_DIR ..."
    git clone https://github.com/rvandewater/YAIB-cohorts.git "$REPO_DIR/YAIB-cohorts"
fi

cd "$REPO_DIR/YAIB-cohorts/R"

# NOT running setup_env.R as-is anymore. Its bare `renv::restore()` call
# (with no exclude argument) is exactly what failed on the previous run:
# it tried to install arrow@22.0.0 from CRAN source, which built against a
# mismatched/incompatible pre-fetched Arrow C++ binary (confirmed by the
# actual error: "undefined symbol: ...TypeVisitor5VisitERKNS_14StringViewTypeE",
# a real ABI mismatch between the arrow R package's expected Arrow C++ API
# version and what got linked). We already have a WORKING, version-matched
# arrow (conda's r-arrow 12.0.1, built specifically for R 4.2 and linked
# against its own matching libarrow -- confirmed loadable via conda-forge's
# own build, not something to second-guess here) -- we just need renv to
# leave it alone rather than overwrite it with the lockfile's pinned
# version. renv::restore(exclude = "arrow") is the documented way to do
# this (confirmed against renv's own reference docs).
#
# This does NOT touch the lockfile itself -- arrow remains pinned at
# 22.0.0 in renv.lock, unchanged. It only tells THIS restore call to skip
# arrow specifically, leaving conda's already-working 12.0.1 in place. If
# this later turns out to cause a real behavioural incompatibility in the
# actual cohort-generation scripts (unlikely for parquet read/write, which
# is the only arrow functionality YAIB-cohorts actually uses, but not
# something to assume blindly), that's the point to revisit this decision.
echo "Running renv::restore(exclude = 'arrow') instead of setup_env.R directly ..."

# Worth checking explicitly: renv performs a TRANSACTIONAL restore by
# default (per renv's own docs: "the project library is mutated only if
# all packages within the lockfile are successfully restored"). The
# previous run's arrow failure happened AFTER 18 other packages had
# already compiled successfully -- if that transactional behaviour rolled
# those back, this run needs to reinstall them too, not just arrow. This
# isn't assumed either way below; the sanity check after restore()
# actually tries to load ricu and data.table, which will fail loudly if
# they were rolled back, rather than silently trusting they're still there.
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

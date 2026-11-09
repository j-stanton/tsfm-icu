# environment

`build_conda_env.slurm` builds the R 4.2.2 conda environment (`yaib_env`) that YAIB-cohorts needs, then runs its `setup_env.R`. This compiles several packages from source, so it runs as a job and not on the login node.

On the cluster, conda environments live in `~/.conda/envs/`, so the script asks `conda env list` where the environment is instead of hard-coding the path.

The Python environments (`yaib` and `tsfm-adapter`) are described in `../baselines/README.md` and `../docs/SETUP_ENVIRONMENT.md`.

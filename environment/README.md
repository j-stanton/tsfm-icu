# environment

`build_conda_env.slurm` builds the **R 4.2.2** conda environment (`yaib_env`) that YAIB-cohorts needs for cohort extraction, then runs its `setup_env.R` (`renv::restore()` compiles several packages from source, so it runs as a job, not on the login node).

Conda environments on the cluster live under `~/.conda/envs/`, not under the module's install path. The script asks `conda env list` for the location rather than hard-coding it.

The Python environments (`yaib`, `tsfm-adapter`) are documented in `../docs/SETUP_ENVIRONMENT.md` and `../baselines/README.md`.
# baselines — supervised reference models

LightGBM, an LSTM and logistic regression, trained with YAIB on the three MIMIC-III cohorts using the shared split.

## Scripts

| Script | Purpose |
|---|---|
| `train_baselines.slurm` | Slurm array `0-8`: one job per (task, model). Index = 3 × task + model, with tasks (Mortality24, AKI, Sepsis) and models (LightGBM, LSTM, LogReg). Logistic regression is therefore indices 2, 5, 8. |
| `train_baselines_test.slurm` | Smaller test version |
| `train_logreg_rerun.slurm` | Reruns logistic regression only (`--array=2,5,8`) after the class-weight fix |
| `train_logreg_sepsis_rerun.slurm` | Sepsis logistic regression with a reduced budget |

Submit with, for example, `sbatch --array=0-8 baselines/train_baselines.slurm`.

## Setup

```bash
srun --partition=stampede --pty bash        # don't install on the login node
module load anaconda/anaconda-2025-12-2
conda create -n yaib python=3.11
conda run -n yaib pip install -e external/YAIB
exit
```

The `yaib` environment needs `polars-lts-cpu` rather than `polars`. Standard polars crashes with "Illegal instruction" on nodes without AVX2 (for example `mscluster23`).

## How the shared split and fixes are applied

`../splits/apply_fixed_split.py` and `../splits/fix_class_weight_bug.py` are monkeypatches. They must be applied inside the same Python process as the training call, so every script runs one `python -c "..."` that imports both patches and then calls `icu_benchmarks.run.main(...)`.

## Deviations to report

- **Class weights.** The first full run had logistic regression at near-chance (AUROC 0.49–0.55) because of a YAIB bug where class weights were never applied. Logistic regression was rerun after the fix.
- **Sepsis logistic regression.** YAIB's 5 × 5 nested CV (25 fits) did not finish in 14 h. The final run uses one repetition × one fold, `saga` with `max_iter` capped at 1000 instead of 100000, `--no-tune` (a single random hyper-parameter draw) and `class_weight='balanced'`. The seed (2222) and shared split are unchanged.

Results are written to `baselines/results/` and `yaib_logs*/` (git-ignored).
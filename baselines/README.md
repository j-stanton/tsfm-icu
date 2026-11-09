# baselines

LightGBM, an LSTM and logistic regression, trained with YAIB on the three MIMIC-III cohorts with the shared split.

## Scripts

| Script | What it does |
|---|---|
| `train_baselines.slurm` | Slurm array 0-8, one job per task and model. Index = 3 × task + model, with tasks (mortality, AKI, sepsis) and models (LightGBM, LSTM, logistic regression). Logistic regression is 2, 5 and 8. |
| `train_baselines_test.slurm` | Shorter test version |
| `train_logreg_rerun.slurm` | Reruns logistic regression (array 2, 5, 8) after the class-weight fix |

For example: `sbatch --array=0-8 baselines/train_baselines.slurm`

## Setup

```bash
srun --partition=stampede --pty bash      # don't install on the login node
module load anaconda/anaconda-2025-12-2
conda create -n yaib python=3.11
conda run -n yaib pip install -e external/YAIB
exit
```

Install `polars-lts-cpu` instead of `polars`. Normal polars crashes with "Illegal instruction" on nodes without AVX2, such as `mscluster23`.

## The patches

`../splits/apply_fixed_split.py` and `../splits/fix_class_weight_bug.py` patch YAIB while it runs. They have to be applied in the same Python process as the training call, so each script runs one `python -c "..."` that imports both and then calls `icu_benchmarks.run.main(...)`.

## Things to know when reading the results

- **Class weights.** In the first full run, logistic regression scored close to chance (AUROC 0.49 to 0.55) because YAIB never applied class weights. I reran it after the fix. The sepsis logistic regression result in the report did not get the fix, and is marked with a dagger there.
- **No tuning.** Hyperparameter tuning was switched off to save time, so each model was trained once with a single random draw of hyperparameters. The baselines are probably weaker than tuned models.
- **No spread.** YAIB repeats each run 5 times over 5 folds, but with the shared split every run uses the same patients. The numbers are the average of those 25 runs, and their spread does not measure sampling uncertainty.
- **Sepsis logistic regression.** YAIB's nested cross-validation (25 fits) did not finish within 14 hours.

Output goes to `baselines/results/` and `yaib_logs*/`. Raw logs are git-ignored.

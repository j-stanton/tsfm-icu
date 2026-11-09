# tsfm-icu

Compares TimesFM 2.5, used zero-shot and with a few labelled examples in its context (few-shot), against LightGBM, an LSTM and logistic regression. The tasks are ICU mortality, sepsis onset and acute kidney injury on MIMIC-III. TimesFM is never fine-tuned.

Cohorts and preprocessing come from [YAIB](https://github.com/rvandewater/YAIB) and [YAIB-cohorts](https://github.com/rvandewater/YAIB-cohorts).

## Folders

| Folder | Contents |
|---|---|
| `baselines/` | Slurm scripts that train the supervised baselines through YAIB |
| `splits/` | The shared patient split, and the patches that make YAIB use it |
| `zero_shot/` | Zero-shot TimesFM pipeline |
| `few_shot/` | Few-shot TimesFM pipeline |
| `environment/` | Slurm job that builds the R environment for YAIB-cohorts |
| `docs/` | Environment setup and zero-shot notes |
| `external/` | Submodules: `YAIB`, `YAIB-cohorts`, `timesfm` |

`RESULTS.md` has the result tables. The per-run JSON files are in `zero_shot/results/` and `few_shot/results/`.

## How it works

1. YAIB-cohorts extracts each task from MIMIC-III.
2. `splits/shared_split.csv` assigns every stay to train, validation or test. All models use it, so they are scored on the same test patients.
3. YAIB trains the baselines on the training split.
4. TimesFM forecasts 15 clinical variables for each patient. The forecasts are turned into a score by a logistic regression on summary features (the probe) or by a clinical rule (KDIGO creatinine for AKI, partial SOFA for sepsis).
5. In the few-shot runs, k = 1, 5, 10 or 20 training patients are placed in front of the query patient's series before forecasting.
6. Models are compared on AUROC, AUPRC and ECE.

## Running it

```bash
git clone --recurse-submodules https://github.com/j-stanton/tsfm-icu.git
cd tsfm-icu
```

No patient data is included. MIMIC-III needs credentialed PhysioNet access.

The scripts are written for a Slurm cluster and use paths like `$HOME/tsfm-icu-early-warning`, so edit those for your setup. Environment setup is in `baselines/README.md` (`yaib`) and `docs/SETUP_ENVIRONMENT.md` (`tsfm-adapter`).

```bash
sbatch environment/build_conda_env.slurm        # R environment for YAIB-cohorts
# extract the cohorts with YAIB-cohorts, then:
sbatch baselines/train_baselines.slurm          # array 0-8
cd zero_shot && sbatch run_zero_shot.slurm
cd ../few_shot && sbatch run_few_shot.slurm mortality random
```

Each folder has its own README.

## Data and licence

MIMIC-III v1.4 is used under a PhysioNet data use agreement. Do not commit or share any MIMIC-derived data.
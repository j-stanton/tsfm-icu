# tsfm-icu

Code for my Honours research report, *Evaluating Time-Series Foundation Models for Early Warning in Intensive Care Units*, at the University of the Witwatersrand (School of Computer Science and Applied Mathematics). Supervisor: Dr Hairong Bau.

The question: how close can TimesFM 2.5, used zero-shot or with a few labelled examples in its context, get to supervised models on ICU early-warning tasks? TimesFM is never fine-tuned. It is compared with LightGBM, an LSTM and logistic regression on three MIMIC-III tasks.

| Task | Prediction | TimesFM forecast horizon |
|---|---|---|
| Mortality (`mortality24`) | one per stay | 4 hours |
| Acute kidney injury (`aki`) | every hour, 3 hours sampled per stay | 6 hours |
| Sepsis onset (`sepsis`) | every hour, 3 hours sampled per stay | 6 hours |

Cohorts and preprocessing come from [YAIB](https://github.com/rvandewater/YAIB) and [YAIB-cohorts](https://github.com/rvandewater/YAIB-cohorts).

## What is where

| Folder | Contents |
|---|---|
| `baselines/` | Slurm scripts that train the supervised baselines through YAIB |
| `splits/` | The shared patient split, and the patches that make YAIB use it |
| `zero_shot/` | Zero-shot TimesFM pipeline |
| `few_shot/` | Few-shot (in-context) TimesFM pipeline |
| `environment/` | Slurm job that builds the R environment for YAIB-cohorts |
| `docs/` | Cluster setup notes and zero-shot design notes |
| `external/` | Submodules: `YAIB`, `YAIB-cohorts`, `timesfm` |

`RESULTS.md` has every table from the report. The per-run JSON files are in `zero_shot/results/` and `few_shot/results/`.

## How it works

1. YAIB-cohorts extracts each task from MIMIC-III into `dyn.parquet`, `outc.parquet` and `sta.parquet`.
2. `splits/shared_split.csv` puts every stay in train (26,662), validation (5,705) or test (5,762). Every model uses it, so they are all scored on the same test patients.
3. YAIB trains the three baselines on the training split.
4. TimesFM (`google/timesfm-2.5-200m-pytorch`) forecasts 15 of the 48 variables for each patient, one variable at a time: `hr sbp dbp map resp temp o2sat crea bili plt wbc lact bun k na`.
5. The forecasts are turned into a label score in one of two ways:
   - **Probe (approach A).** For each variable, take the forecast mean, minimum, maximum, a variance estimate from the quantile spread, and the slope. Join them in a fixed order and fit a logistic regression.
   - **Rule (approach B).** Apply a clinical rule to the forecast: a KDIGO creatinine rule for AKI, and a partial SOFA score for sepsis (platelets, bilirubin, creatinine and MAP only). Mortality has no rule.
6. Adaptation:
   - **Zero-shot.** The probe is fitted on the validation split, then Platt scaled.
   - **Few-shot.** k = 1, 5, 10 or 20 training patients are chosen (randomly, or by similarity to the query patient) and their series are placed in front of the query patient's series. The probe is fitted on those k patients. Labels are never given to TimesFM.
7. Metrics are AUROC, AUPRC and ECE.

## Running it

```bash
git clone --recurse-submodules https://github.com/j-stanton/tsfm-icu.git
cd tsfm-icu
```

No patient data is included. MIMIC-III needs credentialed PhysioNet access. `shared_split.csv` only lists MIMIC stay IDs and their split.

I ran everything on the Wits `mscluster` (partition `stampede`). The scripts use paths like `$HOME/tsfm-icu-early-warning`, so edit those if your layout differs. There are two conda environments:

| Environment | Used for | Setup |
|---|---|---|
| `yaib` | baselines | `baselines/README.md` |
| `tsfm-adapter` | TimesFM | `docs/SETUP_ENVIRONMENT.md` |

Order of jobs:

```bash
sbatch environment/build_conda_env.slurm        # R environment for YAIB-cohorts
# extract the cohorts with YAIB-cohorts, then:
sbatch baselines/train_baselines.slurm          # array 0-8
cd zero_shot && sbatch run_zero_shot.slurm
cd ../few_shot && sbatch run_few_shot.slurm mortality random
```

Each folder has its own README.

## Headline results (test AUROC)

| | Mortality | AKI | Sepsis |
|---|---|---|---|
| Best supervised model | 0.850 (LightGBM) | 0.839 (LSTM) | 0.890 (LightGBM) |
| TimesFM zero-shot, best approach | 0.673 (probe) | 0.643 (rule) | 0.603 (probe) |
| TimesFM 20-shot, random selection, best approach | 0.521 (probe) | 0.644 (rule) | 0.524 (probe) |

Demonstrations did not help. The few-shot rule scores barely change, because labels never reach TimesFM. The few-shot probe is fitted on very few labels and varies a lot between seeds.

Similarity-based selection is 15 to 20 times slower per patient than random selection and kept hitting the time limit. Only a capped mortality run finished (500 test patients, k = 1 for five seeds and k = 5 for three), and the probe was degenerate for at least some queries in every one of those runs. So I could not test random against similarity selection. See `few_shot/README.md`.

Supervised models are scored on every stay-hour and TimesFM on a sample of three hours per stay, so AUPRC is not comparable between them for AKI and sepsis.

## Fixes and workarounds

- **YAIB class weights.** `MLWrapper.fit()` reads an attribute that is never set, so scikit-learn models never got balanced class weights. `splits/fix_class_weight_bug.py` patches it.
- **Shared split.** YAIB computes its own split for each task. `splits/apply_fixed_split.py` replaces that with `shared_split.csv`. Both patches live in this repo, so `external/YAIB` stays untouched.
- **Probe features.** Features are built in a fixed variable order and padded with zeros, so every patient has a vector of the same length.
- **Probe scaling.** Features are standardised before the logistic regression, because `lbfgs` did not converge without it.

## Data and licence

MIMIC-III v1.4 is used under a PhysioNet data use agreement. Do not commit or share any MIMIC-derived data.

This repo has no licence yet, because it is coursework. YAIB, YAIB-cohorts and TimesFM keep their own licences.

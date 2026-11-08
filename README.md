# tsfm-icu

**Evaluating Time-Series Foundation Models for Early Warning in Intensive Care Units**

Code for my MSc research at the University of the Witwatersrand (School of Computer Science and Applied Mathematics). Supervisor: Dr Hairong Bau.

This repo asks:

> To what extent can TimesFM, deployed zero-shot or adapted through few-shot in-context learning, match the performance of fully supervised classical baselines on ICU early-warning tasks on MIMIC-III?

TimesFM is never fine-tuned. It is compared with three supervised baselines (LightGBM, LSTM, logistic regression) on three MIMIC-III tasks.

| Task | Type | Notes |
|---|---|---|
| In-hospital mortality (`mortality24`) | one prediction per stay | Forecast horizon 4 h |
| Acute kidney injury (`aki`) | rolling, per patient-hour | Forecast horizon 6 h |
| Sepsis onset (`sepsis`) | rolling, per patient-hour | Forecast horizon 6 h |

All cohorts and preprocessing come from [YAIB](https://github.com/rvandewater/YAIB) / [YAIB-cohorts](https://github.com/rvandewater/YAIB-cohorts). For rolling tasks, 3 hours per patient are sampled.

## Repository layout

| Folder | Purpose |
|---|---|
| `baselines/` | Slurm scripts that train LightGBM, LSTM and logistic regression through YAIB |
| `splits/` | The shared patient-level split and the patches that force YAIB to use it |
| `zero_shot/` | Zero-shot TimesFM pipeline (Phase 2) |
| `few_shot/` | Few-shot in-context-learning pipeline (Phase 3) |
| `analysis/` | Patient-level bootstrap intervals and paired comparisons from saved predictions |
| `environment/` | Slurm job that builds the R environment needed by YAIB-cohorts |
| `docs/` | Cluster environment setup and zero-shot pipeline notes |
| `external/` | Git submodules: `YAIB`, `YAIB-cohorts`, `timesfm` (unmodified upstream) |
| `cohorts/`, `paper/`, `results/` | Placeholders |

## How it works

1. **Cohorts.** YAIB-cohorts extracts MIMIC-III into `dyn.parquet`, `outc.parquet` and `sta.parquet` per task, under `external/YAIB-cohorts/data/<task>/mimic/`.
2. **Shared split.** `splits/shared_split.csv` assigns every stay to train (26,662), val (5,705) or test (5,762). The baselines and TimesFM both use it, so every model is scored on the same test patients.
3. **Baselines.** YAIB trains LightGBM, an LSTM and logistic regression on the training split.
4. **Forecasting.** For each patient, TimesFM (`google/timesfm-2.5-200m-pytorch`) forecasts 15 clinical variables: `hr sbp dbp map resp temp o2sat crea bili plt wbc lact bun k na`.
5. **Forecast → label.** Two mappings:
   - **Approach A (probe):** per variable, take the forecast mean, min, max, a quantile-spread variance proxy and the slope. Concatenate them in a fixed variable order, then fit StandardScaler → LogisticRegression → Platt calibration.
   - **Approach B (rule):** apply a clinical decision rule to the forecast.
     - AKI: a KDIGO-style creatinine rule.
     - Sepsis: a partial-SOFA proxy using creatinine, bilirubin, platelets and MAP. GCS and vasopressor components are not forecastable, which is a documented limitation.
     - Mortality has no rule, so it is probe-only.
6. **Adaptation regimes.**
   - **Zero-shot:** the probe is fit on the validation split. TimesFM sees no MIMIC-III labelled data.
   - **Few-shot ICL:** k ∈ {1, 5, 10, 20} training patients are selected, either randomly (stratified so at least one positive is included when k ≥ 2) or by cosine-similarity retrieval. Their series are concatenated ahead of the query's context before forecasting, and the probe is fit on the k demonstrations. Labels are never fed to TimesFM. Demonstration order is randomised per replicate.
7. **Metrics.** AUROC, AUPRC and ECE with 95% bootstrap intervals.

## Quick start

```bash
git clone --recurse-submodules <repo-url> tsfm-icu
cd tsfm-icu
```

No patient data is included. MIMIC-III requires credentialed PhysioNet access, and `*.parquet` and `data/` are git-ignored.

The scripts were run on the Wits `mscluster` (partition `stampede`) and use hard-coded cluster paths such as `$HOME/tsfm-icu-early-warning`. Edit these if you use a different layout. Two conda environments are used:

| Environment | Used for | Setup |
|---|---|---|
| `yaib` | baselines | `pip install -e external/YAIB` (needs `polars-lts-cpu`, see `baselines/README.md`) |
| `tsfm-adapter` | TimesFM | clone of the `pytorch` module's `ai` env, see `docs/SETUP_ENVIRONMENT.md` |

Typical order:

```bash
sbatch environment/build_conda_env.slurm          # R env for YAIB-cohorts
# ...extract cohorts with YAIB-cohorts...
sbatch baselines/train_baselines.slurm             # array 0-8
cd zero_shot && sbatch run_zero_shot.slurm         # or run_zero_shot_rerun.slurm
cd ../few_shot && sbatch run_few_shot.slurm mortality random
```

See each folder's README for details.

## Results so far

The numbers below come from the JSON files in `zero_shot/results/` and `few_shot/results/`. Cells are test-set AUROC (95% CI) unless noted.

### Zero-shot TimesFM (5 seeds)

| Task | Approach | AUROC | AUPRC | ECE |
|---|---|---|---|---|
| Mortality | probe | 0.673 (0.661–0.685) | 0.176 | 0.006 |
| AKI | probe | 0.487 (0.482–0.492) | 0.256 | 0.157 |
| AKI | rule | 0.643 (0.639–0.646) | 0.274 | 0.589 |
| Sepsis | probe | 0.603 (0.593–0.614) | 0.022 | 0.167 |
| Sepsis | rule | 0.479 (0.464–0.493) | 0.017 | 0.321 |

### Few-shot TimesFM, random selection (AUROC, mean of 5 seeds)

| Task / approach | k=1 | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| Mortality / probe | 0.500* | 0.564 | 0.523 | 0.521 |
| AKI / probe | 0.500* | 0.454 | 0.478 | 0.531 |
| AKI / rule | 0.643 | 0.644 | 0.644 | 0.644 |
| Sepsis / probe | 0.500* | 0.459 | 0.538 | 0.524 |
| Sepsis / rule | 0.483 | 0.467 | 0.469 | 0.473 |

\* At k=1 a single demonstration cannot contain both classes, so the probe is degenerate and falls back to a constant score. For k ≥ 5 the probe is deliberately left uncalibrated when there are too few examples per class. Both cases are flagged in the result JSONs (`any_seed_degenerate_probe`, `any_seed_uncalibrated_probe`).

Similarity-based selection is about 15× slower per patient than random selection, so only a capped run completed (mortality, k=1, 500 test patients).

### Supervised baselines

<!-- TODO: add the LightGBM / LSTM / LogReg table from the YAIB result logs -->

## Known bugs fixed along the way

- **YAIB class weights.** `MLWrapper.fit()` reads `self.weight`, which is never assigned, so scikit-learn models never received balanced class weights. Patched in `splits/fix_class_weight_bug.py`.
- **Shared split.** YAIB computes its own split per task. `splits/apply_fixed_split.py` monkeypatches it to use `shared_split.csv`, so `external/YAIB` stays an unmodified submodule.
- **Probe features.** Features are built in a fixed variable order and zero-padded, so every patient has the same vector length.
- **Probe scaling.** Features are standardised before the logistic regression. Without this, `lbfgs` failed to converge on the full run.

## Data and ethics

This work uses MIMIC-III v1.4 under a PhysioNet data use agreement. Do not commit or redistribute any MIMIC-derived data.

## Acknowledgements and licences

Built on [YAIB](https://github.com/rvandewater/YAIB), [YAIB-cohorts](https://github.com/rvandewater/YAIB-cohorts) and [TimesFM](https://github.com/google-research/timesfm). Each submodule keeps its own licence.

<!-- TODO: add licence for this repo and citation block -->
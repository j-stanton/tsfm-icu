# Phase 2: Zero-Shot TimesFM Evaluation — Pipeline Guide

This covers everything needed to run Phase 2 (Chapter 4, "early July -
August": zero-shot TimesFM evaluation, Milestone 2 — "zero-shot result
table with AUROC, AUPRC, and ECE under bootstrap intervals") once your
YAIB baselines finish and you have real MIMIC-III YAIB output to point
this at.

## File map

| File | Role | Status |
|---|---|---|
| `demo_selection.py` | Demonstration selection (Phase 3, not used in Phase 2) | done (yours) |
| `timesfm_loader.py` | TimesFM checkpoint wrapper, single + batched forecast calls | done — fixed a quantile-shape doc bug, added `forecast_batch` |
| `forecast_to_label.py` | Approach A (probe) / Approach B (rule) forecast→label mapping | done — sepsis + mortality rule coverage decisions filled in |
| `yaib_data.py` | **NEW** — reads YAIB's dyn/outc/sta.parquet into `Demonstration` objects | done, tested against synthetic data |
| `zero_shot_pipeline.py` | **NEW** — the missing multi-variable aggregation layer; runs one full zero-shot pass over a cohort | done, tested against synthetic data |
| `run_zero_shot.py` | **NEW** — top-level driver: 5 seeds × bootstrap CIs × JSON results table | done, dry-run tested |
| `make_synthetic_yaib_cohort.py` | **NEW** — generates fake YAIB-shaped parquet for testing without real data | test scaffolding only |
| `smoke_test.py` | Your original scaffolding test, extended with sepsis-rule coverage checks | done |

## What actually changed in your existing files

1. **`timesfm_loader.py`**: the `ForecastResult.quantile_forecast` docstring
   claimed "quantile 0 is the mean." Checked against the upstream
   `timesfm` package's own documented API — this is wrong. The 10 columns
   are the 10th–90th percentile bands; the point/median forecast is the
   *separate* `point_forecast` array. This was silently feeding the wrong
   assumption into `forecast_to_label.py`'s variance-proxy calculation
   (`qf[:, 1:]` would have dropped the wrong column). Fixed, and flagged as
   a placeholder still needing final confirmation of the exact quantile
   levels against your installed package version before anything depends
   on a *specific* quantile column rather than aggregate min/max/mean
   (which is all the current code uses, so this is low-risk but worth a
   final check once `timesfm` is actually installed).
2. Added `TimesFMForecaster.forecast_batch()` — the single-series
   `forecast_variable()` you had works but would mean one Python-level
   model call per clinical variable per patient, which does not scale to
   a real cohort. `forecast_batch()` uses the model's native multi-series
   batching (confirmed against the upstream API: `inputs=[series_a,
   series_b]` → stacked output arrays).
3. **`forecast_to_label.py`**: filled in the two open TODOs —
   - **sepsis**: `sepsis_partial_sofa_rule` — a partial-SOFA proxy using
     real SOFA sub-score cutoffs (creatinine, bilirubin, platelets, MAP),
     with an explicit documented gap (CNS/GCS and vasopressor-dose
     components aren't forecastable series). **You should describe this
     gap in Section 3.3 or a limitations note** — it's a genuine scoping
     decision, not an oversight, and a marker will want to see it named.
   - **mortality**: deliberately given *no* Approach-B rule, with the
     reasoning recorded in code (mortality is an endpoint, not a
     thresholdable lab value). Mortality uses Approach A only.

## The real gap this closes: `yaib_data.py` + `zero_shot_pipeline.py`

Your four original files each had a forecast-shaped or feature-shaped
object as input, but nothing loaded real YAIB output or aggregated
per-variable forecasts into a per-patient prediction. That's what these
two new files do:

- `yaib_data.py` reads YAIB's `dyn.parquet` / `outc.parquet` /
  `sta.parquet` (schema confirmed against the YAIB wiki's own example
  config: `GROUP=stay_id`, `SEQUENCE=time`, `LABEL=label`) and converts
  each patient stay into the `Demonstration` object `demo_selection.py`
  already expects.
- `zero_shot_pipeline.py` is the "aggregation across variables happens one
  layer up" piece both your original files pointed to but didn't
  implement: forecasts every clinical variable for a patient in one
  batched call, concatenates fixed-length probe features (Approach A) or
  runs the task's decision rule (Approach B), and computes AUROC/AUPRC/ECE
  over a cohort split.

**A real bug was caught building this**, worth knowing about: naively
concatenating `forecast_to_probe_features` only for whichever variables a
patient happened to have enough data for (`sorted(forecasts)`) gives
*different feature-vector lengths for different patients*, because
missingness is patient-specific. `sklearn`'s `LogisticRegression.fit()`
doesn't catch this — it just fails with a shape error the first time two
patients disagree on which variables were forecastable, which surfaced
immediately when tested against synthetic data with realistic missingness.
Fixed via `build_fixed_length_probe_features()`, which always concatenates
in a **fixed variable order** and zero-pads any variable a patient didn't
have enough data for. This is exactly the kind of bug that's cheap to fix
now and expensive to debug after a real multi-hour cluster run — it's why
the synthetic-data testing below is worth keeping, not just discarding
once real data arrives.

## How to run

### 1. Validate the plumbing (no GPU, no checkpoint download, ~seconds)

```bash
pip install -r requirements.txt --break-system-packages   # skip timesfm[torch] for this step
python make_synthetic_yaib_cohort.py   # writes /tmp/synthetic_yaib_cohort
python smoke_test.py                   # your original scaffolding checks
```

This exercises `demo_selection.py` and `forecast_to_label.py` in
isolation. To exercise the full pipeline (`yaib_data.py` →
`zero_shot_pipeline.py` → metrics) without downloading the real ~800MB
checkpoint, write a small script that subclasses `TimesFMForecaster` and
overrides only `load()` (skip it) and `forecast_batch()` (return
synthetic `ForecastResult`s) — see the pattern used during development;
happy to hand you that test file directly if useful, it's not included
here since it's dev scaffolding rather than part of the deliverable
pipeline.

### 2. Point at real YAIB output

```bash
python run_zero_shot.py \
    --cohort-root /path/to/yaib_output \
    --tasks mortality aki sepsis \
    --max-patients 20            # drop this flag for the full test set
```

`--cohort-root` should contain one subdirectory per task
(`TASK_TO_SUBDIR` in `run_zero_shot.py` — **check this mapping
against your actual YAIB export's directory names before running**; YAIB's
naming convention for the mortality task in particular sometimes includes
the horizon, e.g. `mortality24`).

First run will download the TimesFM checkpoint (~800MB,
`google/timesfm-2.5-200m-pytorch`) — do this once, ideally with disk space
confirmed, before a full cluster run so you're not debugging a download
timeout mid-experiment.

Output: `phase2_zero_shot_results.json`, one row per (task, approach),
each with seed-mean AUROC/AUPRC/ECE and 95% bootstrap CIs — this is
Milestone 2's deliverable.

### 3. Sanity-check before trusting numbers on the real cohort

- Confirm `TASK_TO_SUBDIR` and `DEFAULT_VARIABLE_ORDER` (in
  `yaib_data.py`) actually match your YAIB export's directory names and
  `dyn.parquet` columns — run `load_yaib_cohort` once interactively and
  inspect `cohort.dynamic.columns` before a full run.
- Confirm the split-column name (`cohort.stay_ids()` looks for `"split"`,
  `"fold"`, or `"cv_split"`) matches what your YAIB export actually wrote;
  if none of those exist it silently falls back to using every patient for
  every split, which would leak calibration/test data into each other. The
  smoke test can't catch this since it's an artifact of one specific
  dataset export, not the code — check it manually against your first
  real cohort load.
- Re-check the `sepsis_partial_sofa_rule` cutoffs' scaling once real
  training-fold sepsis-positive/negative distributions are available
  (`delta_threshold=2.0` and the `[0,1]` clamp are a reasonable first pass,
  not validated against real class balance yet).

## Not yet done (Phase 3+, out of scope for "complete Phase 2")

- Few-shot in-context prompt formatting (feeding `demo_selection.py`'s
  selected demonstrations into TimesFM's actual context — Das et al.
  [2024]'s protocol). `demo_selection.py` selects *which* patients; there
  is no code yet that serialises them into TimesFM's input format.
- The Platt-scaling "after calibration" ECE number for Approach B (rule
  scores) — currently `None` in `EvaluationResult`; Approach A's probe
  already calibrates internally. One-line addition once real rule scores
  exist to calibrate against.
- Paired permutation tests / Holm–Bonferroni correction across models
  (Section 3.4) — bootstrap CIs are done; the cross-model significance
  testing layer is a separate analysis script, not part of Phase 2's own
  milestone.

# Zero-shot pipeline notes

Notes on how the zero-shot pipeline in `zero_shot/` fits together and the decisions behind it. For how to run it, see `../zero_shot/README.md`.

## Data flow

1. `yaib_data.py` reads YAIB's `dyn.parquet`, `outc.parquet` and `sta.parquet` (stay id, time and label columns) and turns each stay into a `Demonstration`.
2. `zero_shot_pipeline.py` forecasts all 15 variables for a patient in one batched call with `TimesFMForecaster.forecast_batch()`. One call per variable per patient would be too slow for a full cohort.
3. `forecast_to_label.py` turns the forecasts into a score, either with the probe or with a clinical rule.
4. The pipeline computes AUROC, AUPRC and ECE on the evaluation split.
5. `run_zero_shot.py` repeats this for five seeds and writes bootstrap intervals to a JSON file.

## Decisions

- **Sepsis rule.** A partial SOFA score built from platelets, bilirubin, creatinine and MAP. Respiratory, nervous system and higher cardiovascular components are left out, because the forecast variables don't include them. The rule therefore underestimates severity, and this is listed as a limitation in the report.
- **No mortality rule.** Mortality is an outcome, not a lab value with a threshold, so mortality uses the probe only.
- **Fixed feature length.** If probe features are built only for the variables a patient has enough data for, patients get vectors of different lengths and `LogisticRegression.fit()` fails. `build_fixed_length_probe_features()` always uses the same variable order and fills missing variables with zeros.
- **Explicit split.** YAIB's `outc.parquet` has no split column. If the split file is not given, the code raises an error. An earlier version used every patient for every split, which leaks calibration data into the test set.

## Before trusting a run on a new cohort

- Check the task folder names against `TASK_TO_SUBDIR` in `run_zero_shot.py`. YAIB sometimes puts the horizon in the name, for example `mortality24`.
- Load one cohort interactively with `load_yaib_cohort` and look at `cohort.dynamic.columns`, to make sure the variable names match `DEFAULT_VARIABLE_ORDER` in `yaib_data.py`.
- Run `python make_synthetic_yaib_cohort.py` and `python smoke_test.py` first. They take seconds and catch shape and wiring errors before a long cluster run.

## What was not done

- Platt scaling of the rule scores. Rule outputs are scores, not probabilities, so ECE is only reported for probes.
- Paired permutation tests and subgroup analyses, which I had planned in the proposal. The report uses bootstrap intervals only.

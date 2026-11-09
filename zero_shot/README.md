# zero_shot

TimesFM with no demonstrations. The probe is fitted on the validation split and scored on the test split.

## Files

| File | What it does |
|---|---|
| `run_zero_shot.py` | Runs all tasks and approaches, computes bootstrap intervals, writes a JSON file |
| `zero_shot_pipeline.py` | Forecasts each variable, builds the probe features or rule score, computes the metrics |
| `timesfm_loader.py` | Wrapper around the TimesFM checkpoint, with batched forecasting |
| `forecast_to_label.py` | The probe and the clinical rules |
| `yaib_data.py` | Reads YAIB's parquet files |
| `demo_selection.py` | Demonstration selection. Used by `few_shot/`, but imported here too |
| `make_synthetic_yaib_cohort.py`, `smoke_test.py` | Quick checks on fake data (no GPU or checkpoint needed) |
| `check_cluster_env.py`, `diagnose_aki_probe.py` | Environment check and the AKI probe diagnostic |
| `*.slurm`, `*.sbatch` | Cluster jobs |

## Running

Quick check on fake data:

```bash
python make_synthetic_yaib_cohort.py
python smoke_test.py
```

Real run:

```bash
python run_zero_shot.py \
    --cohort-root ../external/YAIB-cohorts/data \
    --split-assignment ../splits/shared_split.csv \
    --tasks mortality aki sepsis \
    --output results/full_run_results.json
```

`--max-patients` and `--max-calibration-patients` give a smaller dry run.

On the cluster, submit from this folder:

| Script | Purpose |
|---|---|
| `run_env_check.slurm` | check the GPU and packages |
| `run_smoke_test.slurm` | smoke test on the cluster |
| `run_real_dry_run.slurm` | small run on the real data |
| `run_zero_shot.slurm` | the full run |

The first run downloads the checkpoint (about 800 MB).

## Notes

- `--split-assignment` is required for real runs. YAIB's `outc.parquet` has no split column, so without it the code stops instead of mixing the splits.
- Zero-shot inference is deterministic, so the five seeds give identical predictions. A zero-shot result is therefore a single run, and its confidence interval comes from bootstrapping the test instances.
- A small dry run gave an AKI AUROC of 0.098. That was a small-sample effect (see `diagnose_aki_probe.py`) and it did not appear in the full run.

More detail is in `../docs/ZERO_SHOT_NOTES.md`.

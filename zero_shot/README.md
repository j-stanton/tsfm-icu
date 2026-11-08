# zero_shot — Phase 2: zero-shot TimesFM

Evaluates TimesFM with no demonstrations. The probe is fit on the **validation** split and scored on **test**.

## Files

| File | Role |
|---|---|
| `run_zero_shot.py` | Driver: tasks × approaches × seeds, bootstrap CIs, JSON output |
| `zero_shot_pipeline.py` | Forecast every variable per patient, then build probe features or rule scores, then compute metrics |
| `timesfm_loader.py` | TimesFM checkpoint wrapper with batched forecasting |
| `forecast_to_label.py` | Approach A (probe) and Approach B (rule) |
| `yaib_data.py` | Reads YAIB `dyn/outc/sta.parquet` into `Demonstration` objects |
| `demo_selection.py` | Demonstration selection (used in `few_shot/`; kept here because the pipeline imports it) |
| `persistence_forecaster.py` | Ablation: repeats the last observed value instead of calling TimesFM |
| `make_synthetic_yaib_cohort.py`, `smoke_test.py` | Synthetic-data plumbing checks (no GPU or checkpoint needed) |
| `check_cluster_env.py`, `diagnose_aki_probe.py` | Environment check and AKI probe diagnostic |
| `*.slurm`, `*.sbatch` | Cluster jobs |

## Running

Smoke test (CPU, seconds):

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
    --output results/zero_shot_results.json \
    --predictions-dir results/predictions
```

Useful flags: `--max-patients` and `--max-calibration-patients` (dry runs), `--n-seeds`, `--forecaster {timesfm,persistence}`.

On the cluster, submit from this folder:

| Script | Purpose |
|---|---|
| `run_env_check.slurm` / `.sbatch` | check GPU and packages |
| `run_smoke_test.slurm` | smoke test on the cluster |
| `run_real_dry_run.slurm` | capped run on real data |
| `run_zero_shot.slurm` | full run |
| `run_zero_shot_rerun.slurm` | one seed, saves per-patient predictions, repeats with the persistence forecaster |

The first run downloads the checkpoint `google/timesfm-2.5-200m-pytorch` (about 800 MB).

## Notes

- `--split-assignment` is required for real runs. YAIB's `outc.parquet` has no split column, so omitting it raises an error rather than leaking splits.
- Zero-shot inference is deterministic, so extra seeds only duplicate rows. `run_zero_shot_rerun.slurm` therefore uses one seed.
- A capped dry run on AKI gave AUROC 0.098. This was a small-sample artefact (see `diagnose_aki_probe.py`), not a bug, and it disappears on the uncapped run.
- Output rows: seed-mean AUROC, AUPRC and ECE with 95% bootstrap CIs per (task, approach).

See `../docs/ZERO_SHOT_NOTES.md` for the design history.
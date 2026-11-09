# few_shot

TimesFM with k = 1, 5, 10 or 20 labelled demonstrations from the training split. No test patient is ever used as a demonstration.

## How the demonstrations are used

TimesFM only takes numbers in and gives a forecast out, so it cannot read a label. For each variable, the chosen demonstrations' series are joined in a random order and placed in front of the query patient's own series, and TimesFM forecasts from the end of that long series. The labels are only used to fit the probe on the demonstrations' own forecasts. See `timesfm_icl.py`.

## Files

| File | What it does |
|---|---|
| `run_few_shot.py` | Runs tasks, approaches, strategies, values of k and seeds |
| `few_shot_pipeline.py` | One few-shot evaluation |
| `timesfm_icl.py` | Builds the long input series and runs the forecast |
| `demo_selection.py` | `random` (at least one positive when k ≥ 2) and `similarity` (nearest neighbours by cosine similarity) |
| `forecast_to_label.py`, `timesfm_loader.py`, `yaib_data.py` | Copies of the `zero_shot/` modules, so this folder runs on its own |
| `smoke_test.py`, `make_synthetic_yaib_cohort.py` | Quick checks on fake data |
| `run_dry_run.slurm`, `run_dry_run_v2.slurm` | Small runs on the real data |
| `run_few_shot.slurm` | Full run for one task and one strategy |
| `run_few_shot_similarity_capped.slurm` | Capped similarity run |

The copied modules need to be kept in sync with `zero_shot/` by hand.

## Running

Small dry run:

```bash
python run_few_shot.py \
    --cohort-root ../external/YAIB-cohorts/data \
    --split-assignment ../splits/shared_split.csv \
    --tasks mortality --ks 1 5 --strategies random \
    --max-patients 10 --max-train-pool 50 --n-seeds 1
```

Full runs (submit from this folder):

```bash
sbatch run_few_shot.slurm mortality random
sbatch run_few_shot.slurm aki random
sbatch run_few_shot.slurm sepsis random
sbatch run_few_shot_similarity_capped.slurm
```

Other useful flags are `--approaches {probe,rule}` and `--resume`. The script saves after every cell (task, approach, strategy, k), and `--resume` skips the cells already in the output file.

## Cost and limits

- Random selection took about 2 seconds per test patient, so about an hour for the 5,762 mortality test patients.
- Similarity selection takes about 30 to 38 seconds per test patient, because it picks new neighbours and refits the probe for every query. A full run would take days. The first full attempt hit the 48 hour limit, so the run was capped to mortality, k = 1 and 5, probe only, and 500 test patients.
- In the capped run, k = 1 finished for all five seeds and k = 5 for three. The JSON file only holds k = 1. The k = 5 results (AUROC 0.470, 0.501 and 0.500) were read from the raw logs, which are not in this repo. They are listed in `../RESULTS.md`.
- At k = 1 the probe cannot be fitted, so every score is 0.5 and the AUROC is 0.5. Whenever the probe could not be fitted for a query, that query got the score 0.5, and the result is flagged in the JSON.
- The probe is only Platt scaled when each class has at least five demonstrations. In practice that happened in four of the five AKI runs at k = 20 and nowhere else, so the rest are flagged as uncalibrated.
- Mortality has no rule, so only the probe is run for it.

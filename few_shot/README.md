# few_shot — Phase 3: few-shot in-context learning

Evaluates TimesFM with k ∈ {1, 5, 10, 20} labelled demonstrations from the **training** split. No test patient is ever used as a demonstration.

## How demonstrations are used

TimesFM is a numeric univariate forecaster with no label input. For each clinical variable, the selected demonstrations' series are concatenated, in randomised order, **ahead of** the query patient's own context, and the whole sequence is forecast forward. The query's context is always last. The demonstrations' labels are used only to fit the Approach A probe on their own forecasts, never passed to TimesFM. See `timesfm_icl.py`.

## Files

| File | Role |
|---|---|
| `run_few_shot.py` | Driver: tasks × approaches × strategies × k × seeds |
| `few_shot_pipeline.py` | One few-shot evaluation pass |
| `timesfm_icl.py` | Builds the concatenated ICL context and forecasts |
| `demo_selection.py` | `random` (label-stratified) and `similarity` (cosine kNN on summary features) selection |
| `forecast_to_label.py`, `timesfm_loader.py`, `yaib_data.py` | Self-contained copies of the `zero_shot/` modules |
| `smoke_test.py`, `make_synthetic_yaib_cohort.py` | Synthetic-data checks |
| `run_dry_run.slurm`, `run_dry_run_v2.slurm` | Capped runs on real data |
| `run_few_shot.slurm` | Full run, one job per (task, strategy) |
| `run_few_shot_similarity_capped.slurm` | Capped similarity run |

This folder deliberately does not import from `zero_shot/`. The shared helpers are duplicated, so keep them in sync.

## Running

Dry run:

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

Useful flags: `--ks`, `--strategies`, `--approaches {probe,rule}`, `--max-patients`, `--max-train-pool`, `--resume`. The driver checkpoints after every (task, approach, strategy, k) cell, and `--resume` skips cells already in `--output`.

## Cost and known limitations

- Random selection took about 2 s per test patient (about 56 min for the 5,762 mortality test patients).
- Similarity selection takes about 30–38 s per test patient, because it reselects neighbours and refits the probe for every query. A full run would take roughly 10 days, so the similarity result is capped to mortality, k ∈ {1, 5}, 500 test patients. Only k=1 is saved.
- At k=1 the probe is degenerate (one class) and the score is constant, giving AUROC 0.5. For k ≥ 5 the probe is uncalibrated when there are too few examples per class. Both are flagged in the result JSON.
- Mortality has no Approach B rule.
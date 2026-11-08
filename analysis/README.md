# analysis

`bootstrap_from_predictions.py` computes exact **patient-level** bootstrap intervals from the per-patient prediction CSVs written by `zero_shot/run_zero_shot.py --predictions-dir ...`.

It differs from the bootstrap inside `run_zero_shot.py`, which resamples rows (instances). Here all of a patient's scored rows move together.

```bash
# one method
python analysis/bootstrap_from_predictions.py \
    zero_shot/results/predictions/timesfm_aki_rule_seed0.csv --score rule_score

# paired comparison (same instances, same order)
python analysis/bootstrap_from_predictions.py \
    zero_shot/results/predictions/timesfm_aki_rule_seed0.csv --score rule_score \
    --compare zero_shot/results/predictions/persistence_aki_rule_seed0.csv \
    --compare-score rule_score
```

Options: `--score`, `--compare`, `--compare-score`, `--n-boot` (default 1000), `--seed`.
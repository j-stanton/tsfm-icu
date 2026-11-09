# Results summary

These are the numbers reported in the research report. The raw cluster logs are not in this repository.
The per-run JSON files are in `zero_shot/results/` and `few_shot/results/`. The supervised baseline
numbers come from YAIB's own logs on the cluster (`yaib_logs/`), which are not included.

All metrics are on the held-out test patients. AUROC and AUPRC use the positive class as the event.
ECE is reported for probe outputs only.

## Main comparison (AUROC / AUPRC)

| Group | Method | Mortality | AKI | Sepsis |
|---|---|---|---|---|
| Supervised | LightGBM | 0.850 / 0.408 | 0.827 / 0.478 | 0.890 / 0.058 |
| Supervised | LSTM | 0.809 / 0.316 | 0.839 / 0.447 | 0.861 / 0.048 |
| Supervised | Logistic regression | 0.567 / 0.130 | 0.564 / 0.100 | 0.508 / 0.006 (no class weights) |
| Zero-shot | TimesFM, probe | 0.673 / 0.176 | 0.487 / 0.256 | 0.603 / 0.022 |
| Zero-shot | TimesFM, rule | n/a | 0.643 / 0.274 | 0.479 / 0.017 |
| 20-shot (random) | TimesFM, probe | 0.521 ± 0.066 / 0.113 ± 0.037 | 0.531 ± 0.075 / 0.265 ± 0.053 | 0.524 ± 0.041 / 0.019 ± 0.001 |
| 20-shot (random) | TimesFM, rule | n/a | 0.644 ± 0.001 / 0.273 ± 0.001 | 0.474 ± 0.007 / 0.017 ± 0.000 |

Few-shot entries are the mean ± SD over five seeds. Supervised entries are the mean over YAIB's 25 runs.
The TimesFM arm is scored on a sample of stay-hours, so AUPRC is not comparable with the baselines for AKI and sepsis.

## Zero-shot TimesFM, with 95% bootstrap intervals

| Task | Approach | AUROC [95% CI] | AUPRC [95% CI] | ECE | Test instances |
|---|---|---|---|---|---|
| Mortality | Probe | 0.673 [0.646, 0.699] | 0.176 [0.151, 0.205] | 0.006 | 5762 |
| AKI | Probe | 0.487 [0.475, 0.498] | 0.256 [0.243, 0.270] | 0.157 | 14334 |
| AKI | Rule | 0.643 [0.634, 0.651] | 0.274 [0.265, 0.284] | n/a | 14334 |
| Sepsis | Probe | 0.603 [0.579, 0.627] | 0.022 [0.018, 0.028] | 0.167 | 15867 |
| Sepsis | Rule | 0.479 [0.447, 0.511] | 0.017 [0.015, 0.020] | n/a | 15867 |

## Few-shot TimesFM, random selection (mean ± SD over five seeds)

AUROC:

| Task | Approach | k=1 | k=5 | k=10 | k=20 |
|---|---|---|---|---|---|
| Mortality | Probe | 0.500 | 0.564 ± 0.055 | 0.523 ± 0.057 | 0.521 ± 0.066 |
| AKI | Probe | 0.500 | 0.454 ± 0.071 | 0.479 ± 0.021 | 0.531 ± 0.075 |
| AKI | Rule | 0.643 ± 0.001 | 0.644 ± 0.001 | 0.644 ± 0.001 | 0.644 ± 0.001 |
| Sepsis | Probe | 0.500 | 0.459 ± 0.081 | 0.538 ± 0.046 | 0.524 ± 0.041 |
| Sepsis | Rule | 0.483 ± 0.045 | 0.467 ± 0.011 | 0.469 ± 0.004 | 0.474 ± 0.007 |

AUPRC:

| Task | Approach | k=1 | k=5 | k=10 | k=20 |
|---|---|---|---|---|---|
| Mortality | Probe | 0.081 | 0.114 ± 0.034 | 0.094 ± 0.017 | 0.113 ± 0.037 |
| AKI | Probe | 0.207 | 0.220 ± 0.035 | 0.238 ± 0.027 | 0.265 ± 0.053 |
| AKI | Rule | 0.274 ± 0.001 | 0.274 ± 0.001 | 0.273 ± 0.001 | 0.273 ± 0.001 |
| Sepsis | Probe | 0.018 | 0.018 ± 0.005 | 0.024 ± 0.010 | 0.019 ± 0.001 |
| Sepsis | Rule | 0.017 ± 0.002 | 0.017 ± 0.000 | 0.017 ± 0.000 | 0.017 ± 0.000 |

ECE (probe, mean over seeds):

| Task | k=1 | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| Mortality | 0.419 | 0.217 | 0.139 | 0.116 |
| AKI | 0.293 | 0.344 | 0.304 | 0.329 |
| Sepsis | 0.482 | 0.359 | 0.192 | 0.247 |

With k=1 the probe cannot be fitted and its score is a constant 0.5.
Platt scaling is applied only when each class has at least five demonstrations
(four AKI runs at k=20 and no other run).
Three sepsis runs with k>=5 had a probe that could not be fitted (AUROC exactly 0.500).

## Similarity-based selection (mortality only, 500 test patients per seed, probe only)

| k | Seeds finished | AUROC per seed |
|---|---|---|
| 1 | 5 of 5 | 0.500, 0.500, 0.500, 0.500, 0.500 |
| 5 | 3 of 5 | 0.470, 0.501, 0.500 |

The run hit the time limit before the remaining seeds finished. AKI and sepsis were not run with similarity-based selection.

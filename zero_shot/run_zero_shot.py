"""
Phase 2 driver script (proposal Chapter 4, Milestone 2: "zero-shot result
table with AUROC, AUPRC, and ECE under bootstrap intervals").

Runs zero-shot TimesFM evaluation across all three tasks (mortality, aki,
sepsis) and both mapping approaches (probe / rule, where a rule exists),
repeats with 5 seeds controlling the calibration-split sampling (Section
3.4: "All experimental configurations are run with five random seeds"),
and reports 95% bootstrap confidence intervals over 1000 patient-level
resamples of the test set (Section 3.4), matching the metrics YAIB/Section
3.4 specify: AUROC, AUPRC, ECE.

Usage:
    python run_zero_shot.py --cohort-root /path/to/yaib_output \
        --checkpoint google/timesfm-2.5-200m-pytorch

Each task is expected at <cohort-root>/<task_dir>/, matching YAIB's own
directory convention (see yaib_data.py's load_yaib_cohort docstring and
the wiki's `-d demo_data/mortality24/mimic_demo` example). Task -> subdir
names are configurable below since YAIB's own naming varies by dataset
export (e.g. "mortality24" vs "mortality").

Does NOT require a GPU to run structurally (TimesFM 2.5-200M runs on CPU,
just slowly — see the upstream RAM/timing notes in timesfm_loader.py's
checkpoint comments) but real full-cohort runs should be run on the
cluster per the proposal's Phase 2 window (early July - August).
"""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from timesfm_loader import TimesFMForecaster, TimesFMRunConfig, TIMESFM_CHECKPOINT_ID
from yaib_data import load_yaib_cohort, TASK_HORIZON_HOURS
from zero_shot_pipeline import run_zero_shot_evaluation, EvaluationResult
from forecast_to_label import TASK_DECISION_RULES
from persistence_forecaster import PersistenceForecaster

N_SEEDS = 5           # Section 3.4 (default; override with --n-seeds)
N_BOOTSTRAP = 1000    # Section 3.4: "1000 patient-level resamples"

# YAIB-cohorts' real on-disk layout, confirmed via `find ~/tsfm-icu-early-warning
# -name "dyn.parquet"` against the actual cluster output (2026-09-14):
#   <cohort-root>/mortality24/mimic/{dyn,outc,sta}.parquet
#   <cohort-root>/aki/mimic/{dyn,outc,sta}.parquet
#   <cohort-root>/sepsis/mimic/{dyn,outc,sta}.parquet
# i.e. each task has a "mimic" (real, full cohort) subdir AND a "mimic_demo"
# (YAIB's small bundled demo subset) subdir side by side -- this pipeline
# always wants "mimic", never "mimic_demo", for real results. Set
# --cohort-root to the YAIB-cohorts data directory itself (e.g.
# ~/tsfm-icu-early-warning/external/YAIB-cohorts/data), NOT a directory one
# level higher or lower than that.
TASK_TO_SUBDIR = {
    "mortality": "mortality24/mimic",
    "aki": "aki/mimic",
    "sepsis": "sepsis/mimic",
}


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = 0,
) -> tuple[float, float, float]:
    """
    95% bootstrap CI over patient-level resamples (Section 3.4). Returns
    (point_estimate, ci_low, ci_high). Resamples that end up single-class
    (possible with small cohorts / rare outcomes) are skipped rather than
    crashing metric_fn, and the skip count is not currently surfaced to the
    caller — worth logging if skip rate is ever high enough to bias the CI,
    which is more likely on the small demo cohort than on the real
    full-size MIMIC-III cohort this is intended for.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_score)

    boot_scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(set(yt)) < 2:
            continue
        boot_scores.append(metric_fn(yt, ys))

    if not boot_scores:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(boot_scores, [2.5, 97.5])
    return point, float(lo), float(hi)


def save_predictions(predictions_dir: Path | None, tag: str, task_name: str,
                     approach: str, seed: int, result: EvaluationResult) -> None:
    """Write one CSV per (task, approach, seed) with the per-patient scores, so
    intervals, ROC/calibration curves and paired tests can be computed exactly
    afterwards (analysis/ scripts) without re-running TimesFM."""
    if predictions_dir is None:
        return
    predictions_dir.mkdir(parents=True, exist_ok=True)
    path = predictions_dir / f"{tag}_{task_name}_{approach}_seed{seed}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "patient_id", "true_label", "probe_score", "rule_score"])
        for p in result.predictions:
            w.writerow([seed, p.patient_id, int(p.true_label),
                        "" if p.probe_score is None else repr(float(p.probe_score)),
                        "" if p.rule_score is None else repr(float(p.rule_score))])
    print(f"  saved {len(result.predictions)} predictions -> {path}")


def evaluate_task_across_seeds(
    cohort_root: Path,
    task_name: str,
    checkpoint_id: str,
    max_patients: int | None,
    split_assignment: str | Path | None = None,
    max_calibration_patients: int | None = None,
    n_seeds: int = N_SEEDS,
    predictions_dir: Path | None = None,
    forecaster_kind: str = "timesfm",
) -> list[dict]:
    from sklearn.metrics import roc_auc_score, average_precision_score

    cohort_dir = cohort_root / TASK_TO_SUBDIR[task_name]
    horizon = TASK_HORIZON_HOURS[task_name]

    approaches = ["probe"] + (["rule"] if task_name in TASK_DECISION_RULES else [])
    rows = []

    for approach in approaches:
        seed_results: list[EvaluationResult] = []

        # Load TimesFM ONCE per (task, approach), not once per seed. This is
        # a genuine fix, not just an optimisation: TimesFM is zero-shot and
        # weight-free (Section 3.3 — "TimesFM is not fine-tuned or otherwise
        # updated on MIMIC-III data at any stage"), so nothing about the
        # model itself changes across seeds; reloading it 5x accumulated
        # GPU memory across the loop with nothing ever freed and caused a
        # real CUDA OOM on stampede's 6GB GTX1060s (confirmed via a crash
        # inside load_checkpoint's .to(self.device) on the 2nd task's 2nd
        # seed, with 5.91GB already "in use" before that forecast even ran).
        # Only `np.random.seed(seed)` (which affects calibration-split
        # sampling order, not the model) needs to vary per seed.
        if forecaster_kind == "persistence":
            forecaster = PersistenceForecaster(TimesFMRunConfig(max_horizon=horizon))
        else:
            forecaster = TimesFMForecaster(TimesFMRunConfig(max_horizon=horizon))
        forecaster.load()  # real ~800MB checkpoint download on first call (TimesFM only)

        # Same reasoning as the forecaster fix above: nothing about the
        # loaded cohort or its split assignment changes across seeds either
        # (seeds only affect calibration-split *sampling order* inside
        # run_zero_shot_evaluation, not which patients exist or which split
        # they're in) — load once, reuse across all 5 seeds, instead of
        # re-reading and re-merging the same parquet + split files 5x.
        cohort = load_yaib_cohort(
            cohort_dir, task_name=task_name, split_assignment=split_assignment
        )

        for seed in range(n_seeds):
            np.random.seed(seed)  # controls calibration-split sampling
                                   # order and any stochastic tie-breaking
                                   # downstream (Section 3.4: seeds control
                                   # "random sampling of demonstrations" —
                                   # not applicable in zero-shot, but kept
                                   # consistent so this script's seed-loop
                                   # structure is reusable unchanged for
                                   # Phase 3's few-shot runs)

            result = run_zero_shot_evaluation(
                cohort, forecaster, task_name=task_name,
                approach=approach, max_patients=max_patients,
                max_calibration_patients=max_calibration_patients,
            )
            seed_results.append(result)
            save_predictions(predictions_dir, forecaster_kind, task_name, approach, seed, result)
            print(f"  [{task_name}/{approach}] seed={seed} "
                  f"AUROC={result.auroc:.3f} AUPRC={result.auprc:.3f} "
                  f"ECE={result.ece_before_calibration:.3f} "
                  f"n={result.n_patients}")

        # Free GPU memory before the next approach/task's forecaster is
        # constructed — same reasoning as above, just at the outer loop
        # boundary instead of the (now-removed) per-seed one.
        del forecaster
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        # pool per-patient predictions across seeds for the bootstrap CI,
        # matching Section 3.4: "recomputed per seed and pooled"
        y_true = np.concatenate([
            np.array([p.true_label for p in r.predictions]) for r in seed_results
        ])
        score_key = "probe_score" if approach == "probe" else "rule_score"
        y_score = np.concatenate([
            np.array([getattr(p, score_key) for p in r.predictions]) for r in seed_results
        ])

        auroc_pt, auroc_lo, auroc_hi = bootstrap_ci(y_true, y_score, roc_auc_score)
        auprc_pt, auprc_lo, auprc_hi = bootstrap_ci(y_true, y_score, average_precision_score)

        rows.append({
            "task": task_name,
            "approach": approach,
            "checkpoint": checkpoint_id if forecaster_kind == "timesfm" else "persistence",
            "forecaster": forecaster_kind,
            "n_seeds": n_seeds,
            "n_patients_per_seed": seed_results[0].n_patients,
            "auroc_mean": float(np.mean([r.auroc for r in seed_results])),
            "auroc_ci_low": auroc_lo,
            "auroc_ci_high": auroc_hi,
            "auprc_mean": float(np.mean([r.auprc for r in seed_results])),
            "auprc_ci_low": auprc_lo,
            "auprc_ci_high": auprc_hi,
            "ece_mean": float(np.mean([r.ece_before_calibration for r in seed_results])),
        })

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-root", type=Path, required=True,
                         help="Root directory containing per-task YAIB export subdirs")
    parser.add_argument("--checkpoint", type=str, default=TIMESFM_CHECKPOINT_ID)
    parser.add_argument("--tasks", nargs="+", default=["mortality", "aki", "sepsis"])
    parser.add_argument("--max-patients", type=int, default=None,
                         help="Cap eval-split size for a fast dry run; omit for full test set")
    parser.add_argument(
        "--max-calibration-patients", type=int, default=None,
        help=(
            "Cap calibration/val-split size independently from "
            "--max-patients (which only caps eval/test). Defaults to "
            "--max-patients's value if not given. IMPORTANT for dry runs: "
            "without this, --max-patients alone does NOT limit how many "
            "patients the calibration probe is fit on -- on the real "
            "mortality cohort this meant a '--max-patients 20' dry run "
            "still ran full TimesFM inference over all 5,705 val "
            "patients before reaching the small eval loop, making the "
            "run far slower than intended and prone to appearing to hang."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("zero_shot_results.json"))
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS,
                         help="Number of seeds. Zero-shot is deterministic, so 1 is enough "
                              "(and avoids pooling identical copies in the bootstrap).")
    parser.add_argument("--predictions-dir", type=Path, default=None,
                         help="If given, write per-patient predictions (one CSV per "
                              "task/approach/seed) here.")
    parser.add_argument("--forecaster", choices=["timesfm", "persistence"], default="timesfm",
                         help="'persistence' repeats the last observed value (ablation).")
    parser.add_argument(
        "--split-assignment", type=Path, default=None,
        help=(
            "Path to shared_split.csv (stay_id,split columns) -- the SAME "
            "file the classical baselines are trained against (Section "
            "3.2: 'the same patient-level split is used across all three "
            "tasks and four models'). REQUIRED for any real, non-synthetic "
            "run: YAIB-cohorts' own outc.parquet has no split column "
            "(confirmed: only ['stay_id', 'label']), so omitting this will "
            "make load_yaib_cohort() raise as soon as a specific split "
            "(train/val/test) is requested, rather than silently leaking "
            "calibration/test patients into each other."
        ),
    )
    args = parser.parse_args()

    all_rows = []
    for task in args.tasks:
        print(f"\n=== Task: {task} ===")
        rows = evaluate_task_across_seeds(
            args.cohort_root, task, args.checkpoint, args.max_patients,
            split_assignment=args.split_assignment,
            max_calibration_patients=args.max_calibration_patients,
            n_seeds=args.n_seeds,
            predictions_dir=args.predictions_dir,
            forecaster_kind=args.forecaster,
        )
        all_rows.extend(rows)

    args.output.write_text(json.dumps(all_rows, indent=2))
    print(f"\nWrote {len(all_rows)} result rows to {args.output}")

    print("\n=== Summary ===")
    for row in all_rows:
        print(f"{row['task']:10s} {row['approach']:6s}  "
              f"AUROC={row['auroc_mean']:.3f} "
              f"[{row['auroc_ci_low']:.3f},{row['auroc_ci_high']:.3f}]  "
              f"AUPRC={row['auprc_mean']:.3f} "
              f"[{row['auprc_ci_low']:.3f},{row['auprc_ci_high']:.3f}]  "
              f"ECE={row['ece_mean']:.3f}")


if __name__ == "__main__":
    main()

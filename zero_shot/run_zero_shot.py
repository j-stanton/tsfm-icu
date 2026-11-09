"""Run the zero-shot TimesFM evaluation for all tasks and seeds."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from timesfm_loader import TimesFMForecaster, TimesFMRunConfig, TIMESFM_CHECKPOINT_ID
from yaib_data import load_yaib_cohort, TASK_HORIZON_HOURS
from zero_shot_pipeline import run_zero_shot_evaluation, EvaluationResult
from forecast_to_label import TASK_DECISION_RULES

N_SEEDS = 5
N_BOOTSTRAP = 1000

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


def evaluate_task_across_seeds(
    cohort_root: Path,
    task_name: str,
    checkpoint_id: str,
    max_patients: int | None,
    split_assignment: str | Path | None = None,
    max_calibration_patients: int | None = None,
) -> list[dict]:
    from sklearn.metrics import roc_auc_score, average_precision_score

    cohort_dir = cohort_root / TASK_TO_SUBDIR[task_name]
    horizon = TASK_HORIZON_HOURS[task_name]

    approaches = ["probe"] + (["rule"] if task_name in TASK_DECISION_RULES else [])
    rows = []

    for approach in approaches:
        seed_results: list[EvaluationResult] = []

        forecaster = TimesFMForecaster(TimesFMRunConfig(max_horizon=horizon))
        forecaster.load()  # real ~800MB checkpoint download on first call

        cohort = load_yaib_cohort(
            cohort_dir, task_name=task_name, split_assignment=split_assignment
        )

        for seed in range(N_SEEDS):
            np.random.seed(seed)  # controls calibration-split sampling

            result = run_zero_shot_evaluation(
                cohort, forecaster, task_name=task_name,
                approach=approach, max_patients=max_patients,
                max_calibration_patients=max_calibration_patients,
            )
            seed_results.append(result)
            print(f"  [{task_name}/{approach}] seed={seed} "
                  f"AUROC={result.auroc:.3f} AUPRC={result.auprc:.3f} "
                  f"ECE={result.ece_before_calibration:.3f} "
                  f"n={result.n_patients}")

        del forecaster
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

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
            "checkpoint": checkpoint_id,
            "n_seeds": N_SEEDS,
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
    parser.add_argument("--output", type=Path, default=Path("phase2_zero_shot_results.json"))
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

"""Run the few-shot evaluation over tasks, values of k and seeds."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from timesfm_loader import TimesFMForecaster, TimesFMRunConfig, TIMESFM_CHECKPOINT_ID
from yaib_data import load_yaib_cohort, TASK_HORIZON_HOURS
from forecast_to_label import TASK_DECISION_RULES
from few_shot_pipeline import (
    run_few_shot_evaluation,
    build_demonstration_pool,
    FewShotEvaluationResult,
)

N_SEEDS = 5
N_BOOTSTRAP = 1000
DEFAULT_KS = (1, 5, 10, 20)
DEFAULT_STRATEGIES = ("random", "similarity")

# same cohort folders as zero_shot/run_zero_shot.py
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


def _write_checkpoint(output_path: Path, all_rows: list[dict]) -> None:
    output_path.write_text(json.dumps(all_rows, indent=2))


def _completed_cell_keys(existing_rows: list[dict]) -> set[tuple]:
    return {
        (row["task"], row["approach"], row["strategy"], row["k"])
        for row in existing_rows
    }


def evaluate_task_few_shot(
    cohort_root: Path,
    task_name: str,
    checkpoint_id: str,
    max_patients: int | None,
    max_train_pool: int | None,
    split_assignment: str | Path | None,
    ks: tuple[int, ...],
    strategies: tuple[str, ...],
    approaches_override: list[str] | None,
    n_seeds: int,
    output_path: Path | None = None,
    all_rows_so_far: list[dict] | None = None,
    skip_keys: set[tuple] | None = None,
) -> list[dict]:
    cohort_dir = cohort_root / TASK_TO_SUBDIR[task_name]
    horizon = TASK_HORIZON_HOURS[task_name]

    default_approaches = ["probe"] + (["rule"] if task_name in TASK_DECISION_RULES else [])
    approaches = approaches_override or default_approaches
    for a in approaches:
        if a == "rule" and task_name not in TASK_DECISION_RULES:
            raise ValueError(
                f"--approaches requested 'rule' for task '{task_name}', which "
                f"has no Approach-B decision rule (see forecast_to_label.py's "
                f"documented mortality coverage decision)."
            )

    forecaster = TimesFMForecaster(TimesFMRunConfig(max_horizon=horizon))
    forecaster.load()
    cohort = load_yaib_cohort(cohort_dir, task_name=task_name, split_assignment=split_assignment)

    t0 = time.time()
    demonstration_pool = build_demonstration_pool(cohort, split="train", max_pool=max_train_pool)
    print(f"  [{task_name}] demonstration pool: {len(demonstration_pool)} candidates "
          f"(built in {time.time() - t0:.1f}s)", flush=True)

    rows = []
    all_rows_so_far = all_rows_so_far if all_rows_so_far is not None else []
    skip_keys = skip_keys or set()
    for approach in approaches:
        for strategy in strategies:
            for k in ks:
                if (task_name, approach, strategy, k) in skip_keys:
                    print(f"  [{task_name}/{approach}/{strategy}/k={k}] "
                          f"SKIPPED -- already completed in a prior partial "
                          f"results file (--resume)", flush=True)
                    continue
                print(f"  [{task_name}/{approach}/{strategy}/k={k}] starting "
                      f"({n_seeds} seeds, {time.strftime('%H:%M:%S')})", flush=True)
                seed_results: list[FewShotEvaluationResult] = []
                for seed in range(n_seeds):
                    t0 = time.time()
                    result = run_few_shot_evaluation(
                        cohort, forecaster, task_name=task_name, approach=approach,
                        k=k, strategy=strategy, seed=seed,
                        demonstration_pool=demonstration_pool,
                        max_patients=max_patients,
                    )
                    seed_results.append(result)
                    degenerate_flag = " [DEGENERATE PROBE]" if result.probe_degenerate else ""
                    uncalibrated_flag = " [UNCALIBRATED]" if result.probe_uncalibrated else ""
                    print(f"  [{task_name}/{approach}/{strategy}/k={k}] seed={seed} "
                          f"AUROC={result.auroc:.3f} AUPRC={result.auprc:.3f} "
                          f"ECE={result.ece_before_calibration:.3f} "
                          f"n={result.n_patients} ({time.time() - t0:.1f}s)"
                          f"{degenerate_flag}{uncalibrated_flag}", flush=True)

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
                    "strategy": strategy,
                    "k": k,
                    "checkpoint": checkpoint_id,
                    "n_seeds": n_seeds,
                    "n_train_pool": len(demonstration_pool),
                    "n_patients_per_seed": seed_results[0].n_patients,
                    "any_seed_degenerate_probe": any(r.probe_degenerate for r in seed_results),
                    "any_seed_uncalibrated_probe": any(r.probe_uncalibrated for r in seed_results),
                    "auroc_mean": float(np.mean([r.auroc for r in seed_results])),
                    "auroc_ci_low": auroc_lo,
                    "auroc_ci_high": auroc_hi,
                    "auprc_mean": float(np.mean([r.auprc for r in seed_results])),
                    "auprc_ci_low": auprc_lo,
                    "auprc_ci_high": auprc_hi,
                    "ece_mean": float(np.mean([r.ece_before_calibration for r in seed_results])),
                })

                if output_path is not None:
                    _write_checkpoint(output_path, all_rows_so_far + rows)

    del forecaster
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, default=TIMESFM_CHECKPOINT_ID)
    parser.add_argument("--tasks", nargs="+", default=["mortality", "aki", "sepsis"])
    parser.add_argument("--ks", nargs="+", type=int, default=list(DEFAULT_KS),
                         help="k values to sweep (Section 3.3 default: 1 5 10 20)")
    parser.add_argument("--strategies", nargs="+", default=list(DEFAULT_STRATEGIES),
                         choices=["random", "similarity"])
    parser.add_argument("--approaches", nargs="+", default=None, choices=["probe", "rule"],
                         help="Override the per-task default approach list "
                              "(probe always; rule only for aki/sepsis)")
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS,
                         help="Override for dry runs; Section 3.4 specifies 5 for real results")
    parser.add_argument("--max-patients", type=int, default=None,
                         help="Cap eval/test-split size for a fast dry run; omit for full test set")
    parser.add_argument(
        "--max-train-pool", type=int, default=None,
        help=(
            "Cap the training-split demonstration pool size. Not needed for "
            "correctness (pool construction does no TimesFM inference), but "
            "similarity-strategy selection scans the WHOLE pool per query "
            "(cosine similarity, CPU-bound, no GPU) -- capping this keeps a "
            "dry run's similarity-strategy cells fast, mirroring the lesson "
            "phase2's uncapped calibration split taught the hard way for a "
            "different (GPU-bound) reason."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("phase3_few_shot_results.json"))
    parser.add_argument(
        "--split-assignment", type=Path, default=None,
        help="Path to shared_split.csv -- REQUIRED for any real run (see "
             "zero_shot/run_zero_shot.py's identical flag for the full "
             "explanation of why this can't default to anything).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "If --output already exists (e.g. from a job killed by a "
            "walltime limit partway through), load its rows, skip any "
            "(task,approach,strategy,k) cell already present in it, and "
            "keep appending to the same file instead of starting over. "
            "Added after job 68173 (mortality/similarity, 46h --time) "
            "lost a full k=1 + partial k=5 replicate to a timeout with "
            "nothing written to disk -- combined with the new per-cell "
            "checkpointing below (see _write_checkpoint), a resumed job "
            "now only ever re-does work from the cell that was actually "
            "in flight when it died, never anything already completed."
        ),
    )
    args = parser.parse_args()

    all_rows: list[dict] = []
    skip_keys: set[tuple] = set()
    if args.resume and args.output.exists():
        existing_text = args.output.read_text().strip()
        if existing_text:
            all_rows = json.loads(existing_text)
            skip_keys = _completed_cell_keys(all_rows)
            print(f"--resume: loaded {len(all_rows)} already-completed rows "
                  f"from {args.output}; skipping those cells.", flush=True)

    for task in args.tasks:
        print(f"\n=== Task: {task} ===")
        rows = evaluate_task_few_shot(
            args.cohort_root, task, args.checkpoint,
            max_patients=args.max_patients,
            max_train_pool=args.max_train_pool,
            split_assignment=args.split_assignment,
            ks=tuple(args.ks),
            strategies=tuple(args.strategies),
            approaches_override=args.approaches,
            n_seeds=args.n_seeds,
            output_path=args.output,
            all_rows_so_far=all_rows,
            skip_keys=skip_keys,
        )
        all_rows.extend(rows)

    _write_checkpoint(args.output, all_rows)
    print(f"\nWrote {len(all_rows)} result rows to {args.output}")

    print("\n=== Summary ===")
    for row in all_rows:
        flag = " [DEGENERATE]" if row["any_seed_degenerate_probe"] else ""
        flag += " [UNCALIBRATED]" if row["any_seed_uncalibrated_probe"] else ""
        print(f"{row['task']:10s} {row['approach']:6s} {row['strategy']:10s} k={row['k']:<3d} "
              f"AUROC={row['auroc_mean']:.3f} "
              f"[{row['auroc_ci_low']:.3f},{row['auroc_ci_high']:.3f}]  "
              f"AUPRC={row['auprc_mean']:.3f} "
              f"[{row['auprc_ci_low']:.3f},{row['auprc_ci_high']:.3f}]  "
              f"ECE={row['ece_mean']:.3f}{flag}")


if __name__ == "__main__":
    main()

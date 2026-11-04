"""
Phase 3 driver script (proposal Chapter 4, Milestone 3: "complete few-shot
results tables across k -> {1, 5, 10, 20}").

Runs few-shot ICL TimesFM evaluation across all three tasks, both mapping
approaches (probe / rule, where a rule exists), both demonstration-selection
strategies (random / similarity), k in {1, 5, 10, 20}, and 5 seeds per cell
-- Section 3.3/3.4's full "three-way (k x selection x task) factorial with
five seeds".

Usage:
    python run_few_shot.py --cohort-root /path/to/yaib_output \
        --split-assignment /path/to/shared_split.csv

For a fast dry run (recommended before any real cluster submission -- see
--max-patients / --max-train-pool below and the proposal's own Phase 3
milestone note: "Dry-runs verify that the prompt format does not silently
truncate"):
    python run_few_shot.py --cohort-root ... --split-assignment ... \
        --tasks mortality --ks 1 5 --strategies random \
        --max-patients 10 --max-train-pool 50 --n-seeds 1

Structurally identical CLI conventions to zero_shot/run_zero_shot.py
(same --cohort-root / --tasks / --split-assignment / --output shape) --
deliberately kept parallel so both drivers are easy to run/compare
side by side, even though this folder does not import zero_shot/'s code (see
few_shot_pipeline.py's module docstring for why).
"""

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

N_SEEDS = 5        # Section 3.4
N_BOOTSTRAP = 1000  # Section 3.4: "1000 patient-level resamples"
DEFAULT_KS = (1, 5, 10, 20)                # Section 3.3 / Abstract
DEFAULT_STRATEGIES = ("random", "similarity")

# Same real, confirmed on-disk layout as zero_shot/run_zero_shot.py --
# see that file's comment for the full provenance of these paths.
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
    """Identical to zero_shot/run_zero_shot.py's bootstrap_ci — see that
    file for the full rationale; duplicated here for the same
    self-containment reason as few_shot_pipeline.py's dispatch helpers."""
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
    """
    Flushes the full accumulated result set to `output_path` after EVERY
    (task, approach, strategy, k) cell, not just once at the very end of
    the whole run. Added after job 68173 (mortality, strategy=similarity,
    46h --time) ran a full k=1 (5 seeds) plus 3/5 seeds of k=5 -- real,
    expensive GPU compute -- and was killed by the walltime limit before
    the script ever reached its single end-of-run write_text() call,
    losing 100% of that output with nothing recoverable on disk. A cell
    (one full (approach,strategy,k) replicate across all n_seeds) is a
    natural checkpoint granularity: it's the same unit evaluate_task_few_
    shot already treats atomically (a seed loop that then computes one
    bootstrapped row), so writing after each one costs an extra cheap
    JSON dump per cell (rows is small -- a handful of floats/bools per
    row) in exchange for never losing more than one in-progress cell's
    worth of compute to a timeout again.
    """
    output_path.write_text(json.dumps(all_rows, indent=2))


def _completed_cell_keys(existing_rows: list[dict]) -> set[tuple]:
    """
    (task, approach, strategy, k) keys already present in a prior partial
    results file, so --resume can skip cells that already finished rather
    than re-running (and re-paying for) them.
    """
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

    # Load the model and cohort ONCE per task (not per approach/k/strategy/
    # seed) -- same reasoning as phase2's own fix: TimesFM is weight-free
    # and regime-agnostic (Section 3.3), so nothing about the loaded
    # checkpoint changes across any of these axes, only the CONTEXT fed
    # into forecast_batch does. Reloading it per cell would multiply an
    # already-large (k x strategy x approach x seed) factorial by another
    # avoidable factor and risks the same CUDA OOM phase2 hit.
    forecaster = TimesFMForecaster(TimesFMRunConfig(max_horizon=horizon))
    forecaster.load()
    cohort = load_yaib_cohort(cohort_dir, task_name=task_name, split_assignment=split_assignment)

    # Demonstration pool is likewise identical across every approach/k/
    # strategy/seed combination for this task (it's just "the training
    # split's candidates") -- build once, reuse everywhere below.
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
                    # Already present in a prior partial results file
                    # (see --resume) -- this exact (task,approach,
                    # strategy,k) cell already completed all n_seeds and
                    # was checkpointed, so redoing it would just burn
                    # another ~2000-19000s (per the random vs similarity
                    # timing data already on record) for a row we already
                    # have. seed-level resume (skipping individual
                    # completed seeds WITHIN a still-incomplete cell) is
                    # not implemented -- cell-level is the same
                    # granularity _write_checkpoint already uses, and the
                    # seed loop inside one cell is fast enough end-to-end
                    # that losing partial seed progress within a single
                    # resumed cell is an acceptable cost, unlike losing
                    # entire finished cells.
                    print(f"  [{task_name}/{approach}/{strategy}/k={k}] "
                          f"SKIPPED -- already completed in a prior partial "
                          f"results file (--resume)", flush=True)
                    continue
                # Printed BEFORE the first seed starts, not just after it
                # finishes -- a single replicate's full-test-set pass can
                # itself run long (especially strategy="similarity",
                # which refits the probe per eval patient), and without
                # this line a job's log shows nothing at all while that
                # first seed is in flight. Combined with
                # PYTHONUNBUFFERED=1 in the .slurm wrapper (stdout is
                # otherwise fully buffered once redirected to a file, so
                # NOTHING appears until a flush or clean exit -- this is
                # what made the first full-run attempt, job 61982, show
                # zero progress lines despite running the full 24h
                # walltime), this is what actually makes a long job's
                # progress visible in real time.
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

                # Checkpoint immediately -- see _write_checkpoint's
                # docstring for why this replaces the old "write once at
                # the very end of main()" pattern. all_rows_so_far
                # carries rows from tasks/cells already completed in
                # THIS process plus (on --resume) rows loaded from a
                # prior partial run, so this write is always the FULL
                # cumulative result set, not just this task's rows --
                # safe to overwrite output_path with on every call.
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

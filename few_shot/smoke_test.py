"""Quick check of the few-shot code, using a fake forecaster so it runs without TimesFM or a GPU."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from make_synthetic_yaib_cohort import make_synthetic_cohort
from yaib_data import (
    load_yaib_cohort, TASK_HORIZON_HOURS, YAIB_FILE_NAMES, DEFAULT_VARIABLE_ORDER,
)
from few_shot_pipeline import run_few_shot_evaluation, build_demonstration_pool
from timesfm_loader import ForecastResult, TimesFMRunConfig


class FakeForecaster:

    def __init__(self, run_config: TimesFMRunConfig | None = None):
        self.run_config = run_config or TimesFMRunConfig(max_context=64)
        self._loaded = False

    def load(self):
        self._loaded = True

    def forecast_batch(self, variable_names, contexts, horizon=None):
        horizon = horizon or self.run_config.max_horizon
        results = []
        for name, ctx in zip(variable_names, contexts):
            ctx = np.asarray(ctx, dtype=np.float32)
            mean = float(ctx.mean()) if ctx.size else 0.0
            point = np.full(horizon, mean, dtype=np.float32)
            quantiles = np.stack([point + 0.1 * i for i in range(10)], axis=1).astype(np.float32)
            results.append(ForecastResult(
                variable_name=name,
                point_forecast=point,
                quantile_forecast=quantiles,
                context_len_used=len(ctx),
                demonstrations_used=0,  # patched by timesfm_icl.py's caller
            ))
        return results


def make_synthetic_rolling_cohort(out_dir: Path, n_patients: int = 40, n_hours: int = 20, seed: int = 3333):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    splits = (["train"] * int(n_patients * 0.6) + ["val"] * int(n_patients * 0.2))
    splits += ["test"] * (n_patients - len(splits))
    rng.shuffle(splits)

    dyn_rows, outc_rows, sta_rows = [], [], []
    for i in range(n_patients):
        stay_id = 2000 + i
        split = splits[i]
        onset_hour = int(rng.integers(n_hours // 3, n_hours))
        base_shift_by_hour = {}
        for t in range(n_hours):
            label = int(t >= onset_hour)
            base_shift_by_hour[t] = label
            row = {"stay_id": stay_id, "time": t}
            row["hr"] = 80 + 15 * label + rng.normal(0, 6)
            row["crea"] = 0.9 + 1.2 * label + rng.normal(0, 0.1)
            row["bili"] = 0.6 + 1.5 * label + rng.normal(0, 0.1)
            row["plt"] = 220 - 70 * label + rng.normal(0, 10)
            row["map"] = 85 - 15 * label + rng.normal(0, 5)
            for var in DEFAULT_VARIABLE_ORDER:
                if var not in row:
                    row[var] = 90 + rng.normal(0, 5)  # filler for unused variables
            outc_rows.append({"stay_id": stay_id, "time": t, "label": label, "split": split})
            dyn_rows.append(row)
        sta_rows.append({"stay_id": stay_id, "age": float(rng.integers(40, 85)),
                          "sex": float(rng.integers(0, 2))})

    dyn_df = pd.DataFrame(dyn_rows)[["stay_id", "time"] + DEFAULT_VARIABLE_ORDER]
    outc_df = pd.DataFrame(outc_rows)
    sta_df = pd.DataFrame(sta_rows)
    dyn_df.to_parquet(out_dir / YAIB_FILE_NAMES["DYNAMIC"])
    outc_df.to_parquet(out_dir / YAIB_FILE_NAMES["OUTCOME"])
    sta_df.to_parquet(out_dir / YAIB_FILE_NAMES["STATIC"])
    return outc_df[["stay_id", "split"]].drop_duplicates()


def run():
    tmp_root = Path(tempfile.mkdtemp(prefix="phase3_smoke_"))
    try:
        cohort_root = tmp_root / "synthetic_yaib_root"
        split_dfs = {}
        for task_dir in ["mortality24"]:
            out_dir = cohort_root / task_dir / "mimic"
            make_synthetic_cohort(str(out_dir))
            outc = pd.read_parquet(out_dir / YAIB_FILE_NAMES["OUTCOME"])
            split_dfs[task_dir] = outc[["stay_id", "split"]].drop_duplicates()

        aki_out_dir = cohort_root / "aki" / "mimic"
        split_dfs["aki"] = make_synthetic_rolling_cohort(aki_out_dir)

        forecaster = FakeForecaster()
        forecaster.load()

        print("--- Fixed task (mortality): build_demonstration_pool ---")
        cohort = load_yaib_cohort(
            cohort_root / "mortality24" / "mimic", task_name="mortality",
            split_assignment=split_dfs["mortality24"],
        )
        pool = build_demonstration_pool(cohort, split="train")
        print(f"  train pool size: {len(pool)}")
        assert len(pool) > 0, "mortality train pool should not be empty"

        print("\n--- Fixed task (mortality): run_few_shot_evaluation, both strategies, k=1 and k=5 ---")
        for strategy in ["random", "similarity"]:
            for k in [1, 5]:
                result = run_few_shot_evaluation(
                    cohort, forecaster, task_name="mortality", approach="probe",
                    k=k, strategy=strategy, seed=0, demonstration_pool=pool,
                    max_patients=8,
                )
                print(f"  strategy={strategy:10s} k={k}: n={result.n_patients} "
                      f"AUROC={result.auroc} degenerate={result.probe_degenerate}")
                assert result.n_patients == 8
                for p in result.predictions:
                    assert p.k_requested == k
                    assert p.probe_score is not None
                    assert 0.0 <= p.probe_score <= 1.0

        print("\n--- k=0 must raise ValueError (that's the zero-shot regime) ---")
        try:
            run_few_shot_evaluation(
                cohort, forecaster, task_name="mortality", approach="probe",
                k=0, strategy="random", seed=0, demonstration_pool=pool, max_patients=4,
            )
            raise AssertionError("k=0 should have raised ValueError")
        except ValueError as e:
            print(f"  OK: {e}")

        print("\n--- Rolling task (aki): stay_hour dispatch + rule approach ---")
        aki_cohort = load_yaib_cohort(
            cohort_root / "aki" / "mimic", task_name="aki",
            split_assignment=split_dfs["aki"],
        )
        aki_pool = build_demonstration_pool(aki_cohort, split="train")
        print(f"  train pool size: {len(aki_pool)}")
        for approach in ["probe", "rule"]:
            result = run_few_shot_evaluation(
                aki_cohort, forecaster, task_name="aki", approach=approach,
                k=5, strategy="random", seed=0, demonstration_pool=aki_pool,
                max_patients=6,
            )
            print(f"  approach={approach:6s}: n={result.n_patients} AUROC={result.auroc} "
                  f"degenerate={result.probe_degenerate}")
            assert result.n_patients == 6

        print("\n--- Reproducibility: same seed -> identical predictions ---")
        r1 = run_few_shot_evaluation(
            cohort, forecaster, task_name="mortality", approach="probe",
            k=5, strategy="random", seed=42, demonstration_pool=pool, max_patients=5,
        )
        r2 = run_few_shot_evaluation(
            cohort, forecaster, task_name="mortality", approach="probe",
            k=5, strategy="random", seed=42, demonstration_pool=pool, max_patients=5,
        )
        scores1 = [p.probe_score for p in r1.predictions]
        scores2 = [p.probe_score for p in r2.predictions]
        assert scores1 == scores2, f"same seed should reproduce identically: {scores1} vs {scores2}"
        print(f"  OK: identical scores across two runs with seed=42")

        print("\n--- Different seeds -> at least ordering/sampling can differ (not asserting inequality, just no crash) ---")
        r3 = run_few_shot_evaluation(
            cohort, forecaster, task_name="mortality", approach="probe",
            k=5, strategy="random", seed=1, demonstration_pool=pool, max_patients=5,
        )
        print(f"  seed=1 scores: {[round(p.probe_score, 3) for p in r3.predictions]}")

        print("\nAll Phase 3 smoke tests passed.")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    run()

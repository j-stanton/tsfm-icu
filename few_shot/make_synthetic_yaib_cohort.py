"""
Builds a synthetic cohort matching YAIB's on-disk parquet schema (see
yaib_data.py's module docstring), purely for testing the pipeline plumbing
end-to-end before real MIMIC-III/YAIB output is available. NOT meant to
resemble real clinical distributions — same disclaimer as
smoke_test.py's make_synthetic_demonstration.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from yaib_data import DEFAULT_VARIABLE_ORDER, YAIB_FILE_NAMES


def make_synthetic_cohort(
    out_dir: str | Path,
    n_patients: int = 60,
    n_hours: int = 30,
    seed: int = 2222,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    dyn_rows, outc_rows, sta_rows = [], [], []

    # roughly 20% positive rate, split 60/20/20 train/val/test — enough
    # signal in each split for a probe to actually fit without degenerate
    # single-class folds
    splits = (["train"] * int(n_patients * 0.6)
              + ["val"] * int(n_patients * 0.2))
    splits += ["test"] * (n_patients - len(splits))
    rng.shuffle(splits)

    for i in range(n_patients):
        stay_id = 1000 + i
        label = int(rng.random() < 0.25)
        split = splits[i]

        base_shift = 1.0 if label == 1 else 0.0  # "sicker" patients trend worse
        for t in range(n_hours):
            row = {"stay_id": stay_id, "time": t}
            row["hr"] = 80 + 15 * base_shift + rng.normal(0, 6)
            row["sbp"] = 120 - 15 * base_shift + rng.normal(0, 8)
            row["dbp"] = 75 - 8 * base_shift + rng.normal(0, 5)
            row["map"] = 85 - 12 * base_shift + rng.normal(0, 5)
            row["resp"] = 16 + 4 * base_shift + rng.normal(0, 2)
            row["temp"] = 37.0 + 0.6 * base_shift + rng.normal(0, 0.3)
            row["o2sat"] = 97 - 3 * base_shift + rng.normal(0, 1.5)
            row["crea"] = 0.9 + 1.2 * base_shift * (t / n_hours) + rng.normal(0, 0.1)
            row["bili"] = 0.6 + 2.0 * base_shift * (t / n_hours) + rng.normal(0, 0.1)
            row["plt"] = 220 - 80 * base_shift * (t / n_hours) + rng.normal(0, 10)
            row["wbc"] = 8 + 5 * base_shift + rng.normal(0, 1.5)
            row["lact"] = 1.2 + 2.5 * base_shift * (t / n_hours) + rng.normal(0, 0.2)
            row["bun"] = 15 + 10 * base_shift + rng.normal(0, 3)
            row["k"] = 4.0 + 0.3 * base_shift + rng.normal(0, 0.2)
            row["na"] = 139 + rng.normal(0, 2)
            # sprinkle missingness, matching the "non-random missingness"
            # note in the proposal (Section 3.2, Che et al. 2018) loosely —
            # not a rigorous MNAR simulation, just enough to exercise the
            # NaN-filtering path in yaib_data.py / zero_shot_pipeline.py
            for col in list(row.keys()):
                if col not in ("stay_id", "time") and rng.random() < 0.05:
                    row[col] = np.nan
            dyn_rows.append(row)

        outc_rows.append({"stay_id": stay_id, "label": label, "split": split})
        sta_rows.append({
            "stay_id": stay_id,
            "age": float(rng.integers(40, 85)),
            "sex": float(rng.integers(0, 2)),
            "height": float(rng.normal(170, 10)),
            "weight": float(rng.normal(75, 15)),
        })

    dyn_df = pd.DataFrame(dyn_rows)[["stay_id", "time"] + DEFAULT_VARIABLE_ORDER]
    outc_df = pd.DataFrame(outc_rows)
    sta_df = pd.DataFrame(sta_rows)

    dyn_df.to_parquet(out_dir / YAIB_FILE_NAMES["DYNAMIC"])
    outc_df.to_parquet(out_dir / YAIB_FILE_NAMES["OUTCOME"])
    sta_df.to_parquet(out_dir / YAIB_FILE_NAMES["STATIC"])

    return out_dir


if __name__ == "__main__":
    path = make_synthetic_cohort("/tmp/synthetic_yaib_cohort")
    print(f"Synthetic YAIB-shaped cohort written to {path}")
    for f in YAIB_FILE_NAMES.values():
        df = pd.read_parquet(path / f)
        print(f"  {f}: {df.shape}")

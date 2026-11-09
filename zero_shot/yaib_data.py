"""Read the YAIB parquet output into the cohort and demonstration objects used by the pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from demo_selection import Demonstration

YAIB_FILE_NAMES = {
    "DYNAMIC": "dyn.parquet",
    "OUTCOME": "outc.parquet",
    "STATIC": "sta.parquet",
}
YAIB_GROUP_COL = "stay_id"
YAIB_TIME_COL = "time"
YAIB_LABEL_COL = "label"

TASK_KIND = {
    "mortality": "fixed",
    "aki": "rolling",
    "sepsis": "rolling",
}

ROLLING_HOURS_PER_PATIENT = 3

DEFAULT_VARIABLE_ORDER = [
    "hr", "sbp", "dbp", "map", "resp", "temp", "o2sat",  # vitals
    "crea", "bili", "plt", "wbc", "lact", "bun", "k", "na",  # labs
]

TASK_HORIZON_HOURS = {
    "mortality": 4,
    "sepsis": 6,
    "aki": 6,
}


@dataclass
class YAIBCohort:
    """One task's preprocessed cohort, loaded from a YAIB output directory."""
    dynamic: pd.DataFrame
    outcomes: pd.DataFrame  # shape depends on task_kind -- see above
    static: pd.DataFrame    # indexed by stay_id
    variable_order: list[str]
    task_name: str
    task_kind: str = "fixed"  # "fixed" or "rolling" -- see TASK_KIND

    def stay_ids(self, split: str | None = None) -> list:
        """FIXED tasks only (task_kind="fixed", e.g. mortality)."""
        if self.task_kind == "rolling":
            raise ValueError(
                f"stay_ids() called on a task_kind='rolling' cohort "
                f"(task={self.task_name!r}). Rolling tasks (aki, sepsis) "
                f"have an hourly label, not one label per patient — use "
                f"stay_hour_ids() instead, which returns (stay_id, time) "
                f"pairs and preserves the per-hour label structure Section "
                f"3.2/3.4's 'rolling-window tasks' framing requires."
            )
        if split is None:
            return list(self.outcomes.index.unique())
        if "split" not in self.outcomes.columns:
            raise ValueError(
                f"stay_ids(split={split!r}) requested, but this cohort's "
                f"outcomes table has no 'split' column. YAIB-cohorts' own "
                f"outc.parquet does not include one (confirmed: it only "
                f"has ['stay_id', 'label']) — pass `split_assignment=` to "
                f"load_yaib_cohort() pointing at your shared_split.csv "
                f"(e.g. ~/tsfm-icu-early-warning/splits/shared_split.csv) "
                f"so the same patient-level split used for the classical "
                f"baselines is used here too. Returning every patient for "
                f"every split silently, as an earlier version of this "
                f"method did, causes calibration/test leakage."
            )
        mask = self.outcomes["split"] == split
        return list(self.outcomes.index[mask].unique())

    def stay_hour_ids(
        self,
        split: str | None = None,
        hours_per_patient: int | None = None,
        seed: int = 0,
    ) -> list[tuple]:
        """ROLLING tasks only (task_kind="rolling", e.g. aki, sepsis)."""
        if self.task_kind != "rolling":
            raise ValueError(
                f"stay_hour_ids() called on a task_kind={self.task_kind!r} "
                f"cohort (task={self.task_name!r}). Use stay_ids() for "
                f"fixed tasks (mortality)."
            )
        if split is not None and "split" not in self.outcomes.columns:
            raise ValueError(
                f"stay_hour_ids(split={split!r}) requested, but this "
                f"cohort's outcomes table has no 'split' column — pass "
                f"`split_assignment=` to load_yaib_cohort(). Same reasoning "
                f"as stay_ids()'s equivalent check."
            )

        df = self.outcomes
        if split is not None:
            df = df[df["split"] == split]

        n_per = hours_per_patient if hours_per_patient is not None else ROLLING_HOURS_PER_PATIENT
        rng = np.random.default_rng(seed)

        result: list[tuple] = []
        for stay_id, group in df.groupby(level=0):
            times = group.index.get_level_values(1)
            n_available = len(times)
            if n_available <= n_per:
                chosen_idx = np.arange(n_available)
            else:
                chosen_idx = np.linspace(0, n_available - 1, n_per).round().astype(int)
                chosen_idx = np.unique(chosen_idx)  # dedupe if rounding collides
            for idx in chosen_idx:
                result.append((stay_id, times[idx]))

        return result


def load_yaib_cohort(
    cohort_dir: str | Path,
    task_name: str,
    variable_order: Sequence[str] = DEFAULT_VARIABLE_ORDER,
    split_assignment: str | Path | pd.DataFrame | None = None,
) -> YAIBCohort:
    cohort_dir = Path(cohort_dir)
    dyn_path = cohort_dir / YAIB_FILE_NAMES["DYNAMIC"]
    outc_path = cohort_dir / YAIB_FILE_NAMES["OUTCOME"]
    sta_path = cohort_dir / YAIB_FILE_NAMES["STATIC"]

    for p in (dyn_path, outc_path, sta_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Expected YAIB output file not found: {p}. Confirm "
                f"`cohort_dir` points at a task-specific YAIB export "
                f"directory (containing dyn/outc/sta.parquet), not the "
                f"top-level YAIB output root."
            )

    dynamic = pd.read_parquet(dyn_path)
    outcomes = pd.read_parquet(outc_path)
    static = pd.read_parquet(sta_path)

    missing_vars = [v for v in variable_order if v not in dynamic.columns]
    if missing_vars:
        raise ValueError(
            f"variable_order includes columns not present in {dyn_path}: "
            f"{missing_vars}. Available dynamic columns: "
            f"{sorted(dynamic.columns)}"
        )

    if YAIB_LABEL_COL not in outcomes.columns:
        raise ValueError(
            f"Expected a '{YAIB_LABEL_COL}' column in {outc_path}, found: "
            f"{list(outcomes.columns)}"
        )

    outc_is_rolling = YAIB_TIME_COL in outcomes.columns
    expected_kind = TASK_KIND.get(task_name)
    detected_kind = "rolling" if outc_is_rolling else "fixed"
    if expected_kind is not None and expected_kind != detected_kind:
        raise ValueError(
            f"TASK_KIND says task {task_name!r} is {expected_kind!r}, but "
            f"{outc_path} actually has columns {list(outcomes.columns)} "
            f"(detected as {detected_kind!r} based on presence of a "
            f"'{YAIB_TIME_COL}' column). Either TASK_KIND is stale or "
            f"this cohort_dir points at unexpected data — resolve before "
            f"trusting downstream results, since fixed vs rolling changes "
            f"which stay_ids()/stay_hour_ids() method is valid to call."
        )
    task_kind = detected_kind

    outcomes = outcomes.drop(columns=["split"], errors="ignore")

    if split_assignment is not None:
        if isinstance(split_assignment, pd.DataFrame):
            split_df = split_assignment
        else:
            split_path = Path(split_assignment)
            if not split_path.exists():
                raise FileNotFoundError(
                    f"split_assignment file not found: {split_path}. This "
                    f"should point at shared_split.csv (e.g. "
                    f"~/tsfm-icu-early-warning/splits/shared_split.csv) — "
                    f"see YAIBCohort's docstring for why this file is "
                    f"required for any real (non-synthetic) run."
                )
            split_df = pd.read_csv(split_path)

        if YAIB_GROUP_COL not in split_df.columns or "split" not in split_df.columns:
            raise ValueError(
                f"split_assignment must have '{YAIB_GROUP_COL}' and 'split' "
                f"columns; found {list(split_df.columns)}. Confirm this is "
                f"actually shared_split.csv and not some other file."
            )

        split_df = split_df.copy()
        split_df[YAIB_GROUP_COL] = split_df[YAIB_GROUP_COL].astype(
            outcomes[YAIB_GROUP_COL].dtype
        )

        n_before = len(outcomes)
        outcomes = outcomes.merge(
            split_df[[YAIB_GROUP_COL, "split"]], on=YAIB_GROUP_COL, how="inner"
        )
        n_after = len(outcomes)
        if n_after < n_before:
            unit = "patient-hours" if task_kind == "rolling" else "stays"
            print(
                f"[yaib_data] task={task_name!r}: {n_before - n_after} of "
                f"{n_before} {unit} in {outc_path} have no matching entry "
                f"in the split file and were dropped (expected when this "
                f"task's cohort exclusion criteria differ from the "
                f"mortality cohort shared_split.csv was derived from)."
            )

    dynamic = dynamic.set_index([YAIB_GROUP_COL, YAIB_TIME_COL]).sort_index()
    if task_kind == "rolling":
        outcomes = outcomes.set_index([YAIB_GROUP_COL, YAIB_TIME_COL]).sort_index()
    else:
        outcomes = outcomes.set_index(YAIB_GROUP_COL).sort_index()
    static = static.set_index(YAIB_GROUP_COL).sort_index()

    return YAIBCohort(
        dynamic=dynamic,
        outcomes=outcomes,
        static=static,
        variable_order=list(variable_order),
        task_name=task_name,
        task_kind=task_kind,
    )


def stay_to_demonstration(
    cohort: YAIBCohort,
    stay_id,
) -> Demonstration:
    if cohort.task_kind == "rolling":
        raise ValueError(
            f"stay_to_demonstration() called on a task_kind='rolling' "
            f"cohort (task={cohort.task_name!r}). Use "
            f"stay_hour_to_demonstration(cohort, stay_id, time) instead — "
            f"using the full stay as context for a rolling task would leak "
            f"future data past the hour being predicted at."
        )
    stay_dynamic = cohort.dynamic.xs(stay_id, level=YAIB_GROUP_COL)
    dynamic_series = {
        var: stay_dynamic[var].to_numpy(dtype=np.float32)
        for var in cohort.variable_order
    }

    label = int(cohort.outcomes.loc[stay_id, YAIB_LABEL_COL])

    static_row = cohort.static.loc[stay_id]
    static_features = {
        col: float(static_row[col])
        for col in static_row.index
        if pd.api.types.is_numeric_dtype(type(static_row[col])) or isinstance(static_row[col], (int, float))
    }

    return Demonstration(
        patient_id=str(stay_id),
        label=label,
        dynamic_series=dynamic_series,
        static_features=static_features,
    )


def stay_hour_to_demonstration(
    cohort: YAIBCohort,
    stay_id,
    time,
) -> Demonstration:
    if cohort.task_kind != "rolling":
        raise ValueError(
            f"stay_hour_to_demonstration() called on a "
            f"task_kind={cohort.task_kind!r} cohort (task="
            f"{cohort.task_name!r}). Use stay_to_demonstration() for fixed "
            f"tasks (mortality)."
        )

    stay_dynamic = cohort.dynamic.xs(stay_id, level=YAIB_GROUP_COL)
    truncated = stay_dynamic[stay_dynamic.index <= time]

    dynamic_series = {
        var: truncated[var].to_numpy(dtype=np.float32)
        for var in cohort.variable_order
    }

    label = int(cohort.outcomes.loc[(stay_id, time), YAIB_LABEL_COL])

    static_row = cohort.static.loc[stay_id]
    static_features = {
        col: float(static_row[col])
        for col in static_row.index
        if pd.api.types.is_numeric_dtype(type(static_row[col])) or isinstance(static_row[col], (int, float))
    }

    return Demonstration(
        patient_id=f"{stay_id}@{time}",
        label=label,
        dynamic_series=dynamic_series,
        static_features=static_features,
    )


def split_context_and_horizon(
    series: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    if series.size <= horizon:
        return series[:1], series
    return series[: series.size - horizon], series[series.size - horizon :]

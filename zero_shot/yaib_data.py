"""
Reads YAIB's preprocessed parquet output into the Demonstration-shaped
objects the rest of the pipeline (demo_selection.py, timesfm_loader.py,
forecast_to_label.py) already expects.

YAIB's on-disk format (confirmed against the project wiki's example
configuration for MIMIC/sepsis, Section 3.2's "YAIB framework" reference):

    dyn.parquet   -- one row per (stay_id, time), one column per dynamic
                     variable, e.g. "alb", "bili", "crea", "hr", "map",
                     "plt", "sbp", "dbp", "temp", "wbc", ...
    outc.parquet  -- schema DIFFERS by task, confirmed against real cluster
                     data (2026-09-14):
                       - mortality: one row per stay_id, columns
                         ['stay_id', 'label'] -- a single fixed-window
                         outcome per patient.
                       - aki / sepsis: one row per (stay_id, time), columns
                         ['stay_id', 'time', 'label'] -- an HOURLY,
                         time-varying label (confirmed: AKI's outc.parquet
                         has 2,216,091 rows across 38,129-ish patients,
                         e.g. one sample patient alone had 142 hourly
                         rows). This matches Section 3.2/3.4's own
                         "rolling-window tasks (sepsis, AKI)" framing --
                         it is NOT a simplification or a data-quality
                         artifact, it's the actual task structure YAIB
                         generates, and the pipeline must predict at
                         multiple hours per patient to match what the
                         proposal commits to, not collapse to one
                         prediction per patient the way mortality does.
    sta.parquet   -- one row per stay_id, static covariates: "age", "sex",
                     "height", "weight"

GROUP variable = "stay_id", SEQUENCE variable = "time" (matches the
wiki's `vars = {"GROUP": "stay_id", "SEQUENCE": "time", "LABEL": "label", ...}`).

This module does NOT resample, impute, or normalise — Section 3.2 is
explicit that all of that already happened inside YAIB's own pipeline
before the parquet files are written; duplicating it here would violate
the "one preprocessing source of truth" principle already stated in
timesfm_loader.py's forecast_variable() docstring.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from demo_selection import Demonstration

# --- YAIB's own file/column naming convention ---------------------------
# Mirrors the wiki's "preprocess.file_names" / "vars" gin-config block
# verbatim so a config file lifted from a real YAIB experiment can be
# dropped in with minimal translation.
YAIB_FILE_NAMES = {
    "DYNAMIC": "dyn.parquet",
    "OUTCOME": "outc.parquet",
    "STATIC": "sta.parquet",
}
YAIB_GROUP_COL = "stay_id"
YAIB_TIME_COL = "time"
YAIB_LABEL_COL = "label"

# Task -> "fixed" (one label per patient, e.g. mortality) or "rolling"
# (one label per (patient, hour), e.g. aki/sepsis) -- confirmed against
# real outc.parquet schemas on 2026-09-14 (see module docstring). This is
# NOT a guess or a convention borrowed from YAIB's docs -- it was
# discovered by a real crash (TypeError: int() argument ... not 'Series'
# from stay_to_demonstration's .loc[stay_id, 'label'] silently returning
# multiple rows for AKI) and confirmed directly by reading both real
# outc.parquet files' actual columns.
TASK_KIND = {
    "mortality": "fixed",
    "aki": "rolling",
    "sepsis": "rolling",
}

# For rolling tasks: how many hours to sample per patient for evaluation,
# rather than using every eligible hour (which would mean ~2.2M AKI rows
# across ~38K patients — computationally impractical for TimesFM
# inference, and statistically redundant: adjacent hours share almost
# the same context and very often the same label, so exhaustive rolling
# evaluation adds compute far faster than it adds independent evidence).
# This is a deliberate, stated trade-off for the methodology write-up
# (Section 3.4's patient-level bootstrap CI reasoning assumes roughly
# independent per-patient contributions; sampling a small fixed count
# per patient keeps that assumption closer to true than either a single
# hour per patient — which would lose the rolling-window structure
# entirely — or exhaustive per-hour evaluation, which would let
# long-stay patients dominate the sample and violate patient-level
# independence).
ROLLING_HOURS_PER_PATIENT = 3

# Clinical variables this study's TSFM pipeline actually forecasts —
# a subset of YAIB's full harmonised variable list (Section 3.2: "vital
# signs, routine laboratory measurements, ... basic patient covariates"),
# restricted to what's needed by (a) the AKI/sepsis Approach-B rules in
# forecast_to_label.py and (b) a reasonably broad summary-feature basis for
# Approach A and similarity-based demonstration retrieval. Extend this list
# freely; nothing downstream hardcodes its length except the smoke test.
DEFAULT_VARIABLE_ORDER = [
    "hr", "sbp", "dbp", "map", "resp", "temp", "o2sat",  # vitals
    "crea", "bili", "plt", "wbc", "lact", "bun", "k", "na",  # labs
]

# Task -> TimesFM forecast horizon in hours (i.e. how many hours FORWARD
# TimesFM predicts, held out as the tail of each series — NOT how many
# hours of data the task observes/uses as context).
#
# CORRECTED 2026-09-14 after a real bug: mortality was originally set to
# 24 here, based on misreading "mortality24"/"once per stay after 24h" as
# meaning a 24-HOUR FORECAST HORIZON. It does not — confirmed directly
# against real cluster data (`df.groupby('stay_id').size()` on the real
# mortality24/mimic/dyn.parquet: EVERY one of 38,129 patients has EXACTLY
# 25 rows, zero variance). "mortality24" means the task observes a FIXED
# ~24-25 HOUR WINDOW and predicts one outcome at the end of it — the "24"
# describes the (fixed-length) CONTEXT window, not a forecast horizon.
#
# With the old horizon=24 value, split_context_and_horizon(series,
# horizon=24) on a 25-point series computed context = series[:1] (25 - 24
# = 1 point of context, all 24 remaining points reserved as "horizon") --
# a single point fails forecast_patient_variables' "context.size < 2"
# check, so EVERY variable for EVERY mortality patient was silently
# skipped, which surfaced as a "0 usable patients" RuntimeError that
# looked like a data problem but was actually this mis-set horizon value.
#
# Fixed to a small genuine forecast horizon instead (a few hours forward),
# leaving most of the ~24-25 point series as usable context — this is a
# judgement call about what "TimesFM forecasts mortality risk" should
# mean structurally (forecast a short window forward from near-the-end-of-
# stay vitals, then map that forecast to a mortality probability via
# Approach A's probe), not a value pulled from YAIB's own task
# definition, since YAIB's classical baselines don't forecast anything —
# they classify directly from the full window. Worth revisiting/discussing
# in the methodology writeup as a TSFM-adaptation-specific design choice.
#
# AKI/sepsis are unaffected by this bug (different, rolling-window task
# structure per Section 3.2/3.4: "hourly, within 6h") -- their row-count
# distributions have NOT yet been checked against this same failure mode;
# confirm before trusting their results too, using the same
# `df.groupby('stay_id').size()` check used to diagnose mortality here.
TASK_HORIZON_HOURS = {
    "mortality": 4,
    "sepsis": 6,
    "aki": 6,
}


@dataclass
class YAIBCohort:
    """
    One task's preprocessed cohort, loaded from a YAIB output directory.

    IMPORTANT — outcomes shape differs by task_kind (see TASK_KIND):
      - task_kind="fixed" (mortality): `outcomes` indexed by stay_id alone,
        one row per patient.
      - task_kind="rolling" (aki, sepsis): `outcomes` indexed by
        (stay_id, time), one row per patient-hour — confirmed against real
        cluster data, NOT a simplification (see module docstring).
    Callers must check `task_kind` before assuming which shape they have;
    `stay_ids()` and `stay_hour_ids()` below are the two access patterns
    for fixed vs rolling tasks respectively.

    IMPORTANT — split source: YAIB-cohorts' own outc.parquet does NOT
    contain a split column (confirmed against real cluster output,
    2026-09-14). The train/val/test assignment instead comes from a
    separately-generated shared_split.csv (this project's own
    make_shared_split.py / apply_fixed_split.py pipeline — see that
    project's other scripts), the SAME file the classical baselines
    (LightGBM/LSTM/logistic regression) are trained against. This is not
    optional: Section 3.2 is explicit that "the same patient-level split
    is used across all three tasks and four models, so that comparisons
    across configurations are made on identical patient cohorts" —
    loading a different or re-derived split here would make any
    TimesFM-vs-baseline AUROC comparison invalid, not just slightly off.

    `split_assignment`, if provided to load_yaib_cohort(), is merged onto
    `outcomes` as a `split` column at load time (merged on stay_id either
    way, so every hour of a rolling-task patient gets the same split
    label — matching the patient-level split intent) — see `stay_ids()`
    below.
    """
    dynamic: pd.DataFrame   # indexed by (stay_id, time), one col per variable
    outcomes: pd.DataFrame  # shape depends on task_kind -- see above
    static: pd.DataFrame    # indexed by stay_id
    variable_order: list[str]
    task_name: str
    task_kind: str = "fixed"  # "fixed" or "rolling" -- see TASK_KIND

    def stay_ids(self, split: str | None = None) -> list:
        """
        FIXED tasks only (task_kind="fixed", e.g. mortality). For rolling
        tasks, use `stay_hour_ids()` instead — calling this on a rolling
        cohort still technically works (it returns unique stay_ids, since
        `.unique()` on a MultiIndex level collapses duplicates) but throws
        away the per-hour label structure that's the entire point of a
        rolling task, so it's guarded against below rather than left as a
        silent footgun.

        `split` in {"train", "val", "test"} (or whatever values
        shared_split.csv actually uses — this project's split file uses
        exactly these three, confirmed via `head` against the real file).

        Requires a `split` column to already be present on `outcomes` —
        load_yaib_cohort() is responsible for merging one in from a
        shared_split.csv-shaped file. If no split column is present and a
        specific `split` was requested, this FAILS LOUDLY rather than
        silently returning every patient for every split (an earlier
        version of this method had that silent fallback; it was a real
        latent bug — outc.parquet in practice never has a split column,
        so calibration_split="val" and eval_split="test" would have
        silently both resolved to the FULL cohort, causing calibration/
        test leakage with no error or warning at all).
        """
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
        """
        ROLLING tasks only (task_kind="rolling", e.g. aki, sepsis). Returns
        a list of (stay_id, time) tuples — one prediction target per
        sampled patient-hour, NOT one per patient.

        `hours_per_patient`: sample at most this many hours per patient
        (defaults to ROLLING_HOURS_PER_PATIENT). Sampling, not exhaustive
        enumeration, is deliberate — see ROLLING_HOURS_PER_PATIENT's own
        comment for the full reasoning (compute cost + patient-level
        independence for the bootstrap CI). Sampled hours are spread
        across each patient's available timeline (evenly-spaced indices
        into their sorted hour list) rather than clustered at the start or
        end, so within-patient coverage reflects the whole stay, not just
        one part of it.

        `split` behaves the same as `stay_ids()`, but is matched against
        each (stay_id, time) row's own 'split' column value (which, since
        the split merge in load_yaib_cohort() is done on stay_id alone,
        will be identical across all hours of one patient — the split is
        still fundamentally patient-level, just repeated across that
        patient's rows).
        """
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
        # groupby on the stay_id level of the MultiIndex -- level=0 by
        # convention (dynamic/outcomes are always set_index'd with
        # stay_id first, time second, see load_yaib_cohort()).
        for stay_id, group in df.groupby(level=0):
            times = group.index.get_level_values(1)
            n_available = len(times)
            if n_available <= n_per:
                chosen_idx = np.arange(n_available)
            else:
                # evenly-spaced indices across the sorted timeline, not
                # random -- gives deterministic, reproducible within-
                # patient coverage of early/mid/late stay rather than a
                # seed-dependent scatter (the seed still controls which
                # PATIENTS' hours get used when hours_per_patient itself
                # varies run to run, via any upstream patient-level
                # sampling — this only controls WHICH hours within one
                # patient are chosen, which should be stable).
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
    """
    Loads dyn/outc/sta.parquet from a YAIB task-cohort directory, e.g.
    `<yaib_output_root>/mortality24/mimic/` per YAIB-cohorts' real
    directory convention (confirmed against actual cluster output,
    2026-09-14: `.../mortality24/mimic/`, `.../aki/mimic/`,
    `.../sepsis/mimic/` — NOT `mimic_demo`, which is YAIB's small bundled
    demo subset sitting alongside the real cohort in the same parent dir).

    `split_assignment`: path to (or already-loaded DataFrame of)
    shared_split.csv — two columns, `stay_id` and `split`, with values
    like `train`/`val`/`test` (confirmed against this project's real
    split file). This is merged onto `outcomes` as a `split` column.
    STRONGLY recommended to always pass this for any real (non-synthetic-
    test) run — see YAIBCohort's docstring for why silently omitting it is
    a correctness bug, not just a missing convenience. Left optional
    rather than required only so synthetic/smoke-test cohorts (which embed
    their own ad hoc split column directly in a test-only outc.parquet)
    don't need a real split file to exercise the rest of the pipeline.

    IMPORTANT — patient-set mismatch across tasks: shared_split.csv is
    derived from the mortality cohort's patient set (per this project's
    make_shared_split.py). AKI and sepsis have their own exclusion
    criteria and will NOT contain exactly the same patients. This function
    does an inner-join-shaped merge (rows in outc.parquet with no matching
    stay_id in the split file are dropped, not silently kept with a null
    split) — matching apply_fixed_split.py's documented intersect policy
    for the classical-baseline training pipeline, so TimesFM and the
    baselines see the same resolved patient set per task, not just the
    same split *file*.

    Only columns in `variable_order` are kept from the dynamic table, so a
    variable_order mismatch with what's actually in dyn.parquet fails loudly
    here rather than silently downstream in feature extraction.
    """
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

    # Detect whether this is a fixed (one row per patient) or rolling (one
    # row per patient-hour) outcomes table, from the columns actually
    # present — confirmed real schemas, not guessed (2026-09-14): fixed
    # tasks have ['stay_id', 'label']; rolling tasks have
    # ['stay_id', 'time', 'label']. TASK_KIND is a cross-check /
    # documentation aid, not the source of truth — the actual column
    # presence decides, so this doesn't silently mis-handle a task if
    # TASK_KIND and the real file ever disagree (fails loudly instead,
    # below).
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

    # Unconditionally strip any pre-existing 'split' column from the raw
    # outcomes table before deciding what to do next. Real YAIB-cohorts
    # output never has one (confirmed: neither fixed nor rolling
    # outc.parquet schemas include one), but this guards against any other
    # source that might — without this, `stay_ids()`'s "no split column ->
    # raise" safety check further down could be silently bypassed by a
    # leftover column from whatever produced outc.parquet, rather than
    # reflecting only what THIS function's own split_assignment merge
    # actually resolved.
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

        # Dtype guard: outc.parquet's stay_id has been observed as int32
        # while shared_split.csv's (read from CSV, no explicit dtype) came
        # back int64. pandas' merge() handles this correctly on its own in
        # practice (confirmed: a real merge with this exact int32/int64
        # mismatch produced correct, non-duplicated results) — but relying
        # on that implicit coercion silently is fragile, so it's made
        # explicit and verified here instead of trusted blindly.
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
            # Expected for aki/sepsis (different exclusion criteria than
            # mortality, which shared_split.csv is derived from) — logged
            # rather than silent, per apply_fixed_split.py's own documented
            # precedent of surfacing dropped-patient counts instead of
            # burying them. For rolling tasks n_before/n_after count ROWS
            # (patient-hours), not patients, so the printed numbers are
            # much larger than the patient-level drop count would be —
            # noted explicitly so this isn't misread as "273,658 patients
            # dropped" when it's actually patient-hours.
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
    """
    FIXED tasks only (task_kind="fixed", e.g. mortality) — converts one
    patient stay into the Demonstration dataclass that demo_selection.py's
    selection functions and build_summary_feature_vector already operate
    on. For rolling tasks (aki, sepsis), use stay_hour_to_demonstration()
    instead, which truncates context at a specific hour rather than using
    the patient's whole stay (using the whole stay would leak future data
    into a prediction that's supposed to be made using only data up to
    that hour).
    """
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
    """
    ROLLING tasks only (task_kind="rolling", e.g. aki, sepsis) — converts
    one (patient, hour) pair into a Demonstration whose dynamic_series are
    TRUNCATED to end at `time` (inclusive), and whose label is that
    specific hour's label (not the patient's last or "final" label — for
    a rolling task there is no single label per patient, per this
    module's docstring).

    Truncation is the critical correctness point here: a rolling-window
    prediction at hour t must only see data up to and including hour t,
    never later — otherwise the "forecast" is really backfilling against
    data TimesFM was implicitly given, which would inflate performance in
    a way that doesn't reflect genuine early-warning capability (the
    entire premise of this study per the proposal's title and Section
    1.1's "early prediction of circulatory failure" framing).

    `patient_id` on the returned Demonstration encodes both stay_id and
    time (e.g. "200003@6") rather than just stay_id, since a single
    patient now contributes multiple independent Demonstration objects —
    reusing bare stay_id as patient_id would collide across a patient's
    sampled hours in anything that assumes patient_id is unique per
    Demonstration (e.g. dict keys, set membership checks in
    demo_selection.py's stratified sampling).
    """
    if cohort.task_kind != "rolling":
        raise ValueError(
            f"stay_hour_to_demonstration() called on a "
            f"task_kind={cohort.task_kind!r} cohort (task="
            f"{cohort.task_name!r}). Use stay_to_demonstration() for fixed "
            f"tasks (mortality)."
        )

    stay_dynamic = cohort.dynamic.xs(stay_id, level=YAIB_GROUP_COL)
    # Truncate to rows at or before `time` -- this is the leakage guard.
    # dyn.parquet's time column dtype was observed as a pandas Timedelta
    # (e.g. "0 days 06:00:00") in the real AKI data, matching outc.parquet's
    # time column, so a direct <= comparison works without extra parsing.
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
    """
    Splits one variable's full hourly series into the context TimesFM sees
    and the held-out tail used only for the Approach-B rules that need a
    "current" reading (e.g. baseline_creatinine in aki_kdigo_creatinine_rule)
    — NOT used to compute metrics, since this study forecasts forward from
    context into a genuinely unobserved window, it does not backtest against
    already-observed values. If `series` is shorter than `horizon + 1`, the
    context is whatever remains (possibly a single point); TimesFM supports
    short/variable-length contexts (Das et al. [2023]), so this is not an
    error case, just a low-information one, and is left to the caller to
    filter out via `context_len_used` on the resulting ForecastResult if a
    minimum context length policy is wanted for the real experiments.
    """
    if series.size <= horizon:
        return series[:1], series
    return series[: series.size - horizon], series[series.size - horizon :]

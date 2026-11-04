"""
Forces YAIB's data loading to use the pre-computed shared split
(shared_split.csv, from make_shared_split.py) instead of letting
make_single_split_polars() compute a fresh, task-specific split.

WHY A MONKEYPATCH RATHER THAN EDITING split_process_data.py DIRECTLY:
YAIB lives in external/YAIB/ as a git submodule tracking the upstream repo.
Editing it in place would mean either (a) maintaining a diverged fork with
merge conflicts on every upstream update, or (b) losing the fix if the
submodule is ever re-cloned/reset. A monkeypatch applied from OUR code,
imported before icu-benchmarks runs, keeps the fix in our repo where it's
version-controlled independently of the submodule.

This trades a bit of fragility (it depends on YAIB's internal function
signature not changing) for keeping external/YAIB/ an unmodified,
upstream-trackable checkout — worth revisiting if YAIB's API churns, but
reasonable for now.

USAGE:
This needs to run BEFORE icu_benchmarks.run.main() executes, so that the
patched make_single_split_polars is what gin's config system actually binds
to. Two ways to invoke:

  1. As a `python -c` preamble (works with the plain script, not the
     installed `icu-benchmarks` console entry point, since that doesn't
     give us a hook to inject code first):

     uv run python -c "
     import apply_fixed_split; apply_fixed_split.patch()
     from icu_benchmarks.run import main
     main()
     " -- <normal icu-benchmarks CLI args>

  2. Import and call patch() at the top of a small custom run script if
     you end up writing one (recommended once this stabilizes, so the
     patch isn't easy to forget to apply).

POLICY ON MISMATCHED STAY SETS (documented in make_shared_split.py's output
too): AKI and sepsis cohorts have their own exclusion criteria (different
attrition than mortality). This module's default policy is "intersect" —
for each task, only stays present in BOTH the shared split AND that task's
own cohort are used, keeping each task's train/val/test proportions close
to 70/15/15 but with fewer total stays than the full task cohort. Stays
dropped by this intersection are logged so attrition is traceable,
consistent with the methodology's attrition-tracking elsewhere. This is a
real methodological choice, not just an implementation detail — worth
explicitly noting in the final write-up alongside each task's headline
attrition table.

NOTE ON TERMINOLOGY: the shared split itself is assigned at the PATIENT
(subject) level via ICUSTAYS.csv (see make_shared_split.py) — every stay of
a given patient is guaranteed to land in the same split, verified with zero
cross-split overlap on the real mortality cohort. This module still talks
in terms of "stays" throughout, since that's the unit each task's cohort is
keyed on and the unit this patch actually joins on — the patient-level
guarantee is already baked into shared_split.csv by the time this code
runs, not something this module needs to re-enforce itself.
"""

import logging
from pathlib import Path

import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

_SHARED_SPLIT_CACHE: pd.DataFrame | None = None


def _load_shared_split(split_path: str = "../splits/shared_split.csv") -> pd.DataFrame:
    global _SHARED_SPLIT_CACHE
    if _SHARED_SPLIT_CACHE is None:
        path = Path(split_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Shared split file not found at {path}. Run make_shared_split.py first."
            )
        _SHARED_SPLIT_CACHE = pd.read_csv(path)
        logger.info(f"Loaded shared split from {path} ({len(_SHARED_SPLIT_CACHE)} stays)")
    return _SHARED_SPLIT_CACHE


def _fixed_split_polars(data, vars, cv_repetitions, repetition_index, cv_folds,
                         fold_index, train_size=None, seed=42, debug=False,
                         runmode=None, split_path="../splits/shared_split.csv"):
    """
    Drop-in replacement for make_single_split_polars with the same return
    shape (dict of {train, val, test} -> dict of {DYNAMIC, STATIC, OUTCOME}
    Polars DataFrames), but sourcing the split assignment from the shared
    split file rather than computing one via StratifiedKFold.

    cv_repetitions / repetition_index / cv_folds / fold_index are accepted
    for signature compatibility but NOT used for re-splitting — the shared
    split is a single fixed assignment, not a k-fold rotation. This means
    repeated-CV / multi-fold requests (Section 3.4: "five random seeds...
    bootstrap confidence intervals... paired permutation tests") get their
    variability from elsewhere (seed-controlled model init, demonstration
    sampling, etc.) rather than from re-splitting patients each repetition.
    This is a deliberate trade-off to satisfy the "same split across tasks
    and models" requirement; flagging it explicitly here since it changes
    what "5 repetitions" means relative to YAIB's default behaviour.
    """
    from icu_benchmarks.data.constants import VarType, DataSegment, DataSplit

    _id = vars[VarType.group]
    shared = _load_shared_split(split_path)
    id_col_shared = shared.columns[0]  # written as <stay_id_col>, split in make_shared_split.py

    # Task's own available stays (post that task's own exclusion criteria)
    task_ids = set(data[DataSegment.outcome][_id].unique().to_list())
    shared_ids = set(shared[id_col_shared].tolist())

    usable = shared[shared[id_col_shared].isin(task_ids)]
    dropped = shared_ids - task_ids
    if dropped:
        logger.warning(
            f"{len(dropped)} stays from the shared split are not present in "
            f"this task's cohort (excluded by this task's own criteria) and "
            f"will be dropped from this task's split. "
            f"{len(usable)} of {len(shared)} shared-split stays retained."
        )

    id_to_split = dict(zip(usable[id_col_shared], usable["split"]))

    data_split = {}
    for split_name in ["train", "val", "test"]:
        ids_for_split = [pid for pid, s in id_to_split.items() if s == split_name]
        ids_frame = pl.DataFrame({_id: ids_for_split}).with_columns(
            pl.col(_id).cast(pl.datatypes.Int64)
        )
        data_split[split_name] = {
            data_type: ids_frame.join(
                data[data_type].with_columns(pl.col(_id).cast(pl.datatypes.Int64)),
                on=_id,
                how="left",
            ).sort(by=_id)
            for data_type in data.keys()
        }

    # Fixed from an earlier version: the previous log line
    # (`len(id_to_split and [...])`) relied on `dict and list` evaluating
    # to the list when the dict is truthy — technically correct in the
    # common case, but confusing to read and latently wrong if id_to_split
    # were ever empty (dict and X -> dict when the dict is falsy, so
    # len() would report the dict's own key count instead of 0). Computing
    # each count directly and independently, matching how val/test were
    # already being computed, removes both problems.
    n_train = sum(1 for v in id_to_split.values() if v == "train")
    n_val = sum(1 for v in id_to_split.values() if v == "val")
    n_test = sum(1 for v in id_to_split.values() if v == "test")
    logger.info(f"Fixed split applied: train={n_train}, val={n_val}, test={n_test}")

    return data_split


def patch(split_path: str = "../splits/shared_split.csv"):
    """
    Applies the monkeypatch. Call this ONCE, before icu_benchmarks.run.main()
    is invoked, so that gin's config binding picks up the patched function
    rather than the original.
    """
    import icu_benchmarks.data.split_process_data as split_module

    def _patched(*args, **kwargs):
        kwargs.setdefault("split_path", split_path)
        return _fixed_split_polars(*args, **kwargs)

    split_module.make_single_split_polars = _patched
    logger.info(f"Patched make_single_split_polars to use fixed split at {split_path}")
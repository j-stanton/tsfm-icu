"""Make YAIB use the shared split file instead of computing a new split per task."""

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
    from icu_benchmarks.data.constants import VarType, DataSegment, DataSplit

    _id = vars[VarType.group]
    shared = _load_shared_split(split_path)
    id_col_shared = shared.columns[0]

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

    n_train = sum(1 for v in id_to_split.values() if v == "train")
    n_val = sum(1 for v in id_to_split.values() if v == "val")
    n_test = sum(1 for v in id_to_split.values() if v == "test")
    logger.info(f"Fixed split applied: train={n_train}, val={n_val}, test={n_test}")

    return data_split


def patch(split_path: str = "../splits/shared_split.csv"):
    """Applies the monkeypatch."""
    import icu_benchmarks.data.split_process_data as split_module

    def _patched(*args, **kwargs):
        kwargs.setdefault("split_path", split_path)
        return _fixed_split_polars(*args, **kwargs)

    split_module.make_single_split_polars = _patched
    logger.info(f"Patched make_single_split_polars to use fixed split at {split_path}")

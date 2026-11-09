"""Build the TimesFM input from the selected demonstrations and the query patient."""

from dataclasses import replace
from typing import Sequence

import numpy as np

from demo_selection import Demonstration
from timesfm_loader import TimesFMForecaster, TimesFMRunConfig, ForecastResult
from yaib_data import split_context_and_horizon


def _clean_series(raw: np.ndarray) -> np.ndarray:
    """NaN-dropped view of one variable's raw hourly series."""
    raw = np.asarray(raw, dtype=np.float32)
    return raw[~np.isnan(raw)]


def order_demonstrations(
    demonstrations: Sequence[Demonstration],
    rng: np.random.Generator,
) -> list[Demonstration]:
    ordered = list(demonstrations)
    rng.shuffle(ordered)
    return ordered


def build_icl_context(
    query_context: np.ndarray,
    demo_series: Sequence[np.ndarray],
    max_context: int,
) -> tuple[np.ndarray, int]:
    query_context = np.asarray(query_context, dtype=np.float32)
    if query_context.size >= max_context:
        # no room left for demonstrations
        return query_context[-max_context:], 0

    demo_budget = max_context - query_context.size
    if demo_series:
        demo_concat = np.concatenate([np.asarray(s, dtype=np.float32) for s in demo_series])
    else:
        demo_concat = np.zeros(0, dtype=np.float32)

    if demo_concat.size > demo_budget:
        demo_concat = demo_concat[-demo_budget:]  # keep the demonstrations closest to the query

    full_context = np.concatenate([demo_concat, query_context])
    return full_context, int(demo_concat.size)


def forecast_patient_variables_icl(
    forecaster: TimesFMForecaster,
    query: Demonstration,
    demonstrations: Sequence[Demonstration],
    variable_order: Sequence[str],
    horizon: int,
    rng: np.random.Generator,
    run_config: TimesFMRunConfig | None = None,
) -> dict[str, ForecastResult]:
    run_config = run_config or forecaster.run_config
    ordered_demos = order_demonstrations(demonstrations, rng)

    var_names, contexts, demo_counts_used = [], [], []
    for var in variable_order:
        query_series = _clean_series(query.dynamic_series.get(var, np.array([])))
        query_context, _horizon_tail = split_context_and_horizon(query_series, horizon)
        if query_context.size < 2:
            continue

        demo_series_for_var = []
        n_contributing = 0
        for demo in ordered_demos:
            s = _clean_series(demo.dynamic_series.get(var, np.array([])))
            if s.size == 0:
                continue
            demo_series_for_var.append(s)
            n_contributing += 1

        full_context, _n_demo_points = build_icl_context(
            query_context, demo_series_for_var, max_context=run_config.max_context
        )
        var_names.append(var)
        contexts.append(full_context)
        demo_counts_used.append(n_contributing)

    if not var_names:
        return {}

    results = forecaster.forecast_batch(var_names, contexts, horizon=horizon)

    patched = {}
    for result, n_used in zip(results, demo_counts_used):
        patched[result.variable_name] = replace(result, demonstrations_used=n_used)
    return patched

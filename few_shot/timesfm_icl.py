"""
Phase 3: the in-context-learning integration layer referenced (but not yet
implemented) in demo_selection.py's own module docstring:

    "It does not touch TimesFM itself — see timesfm_icl.py for how selected
    demonstrations get formatted into the model's input context."

This is that module. It turns a list of already-SELECTED Demonstration
objects (demo_selection.py's job) into the actual concatenated numeric
context TimesFM forecasts from, per Das et al. [2024]'s protocol as the
proposal states it (Section 3.3):

    "k labelled demonstrations from the training cohort are prepended to
    the input at inference time ... Demonstration ordering within each
    prompt is randomised across replicate runs and the result averaged,
    controlling for the known sensitivity of ICL to demonstration order
    [Lu et al. 2022]."

Mechanism (this is the one substantive design decision this module makes,
since neither the proposal nor demo_selection.py specify it at the
implementation level): TimesFM has no text-prompt / label-token input --
it is a purely numeric univariate forecaster (timesfm_loader.py's own
docstring: "TimesFM is a univariate forecaster ... each input clinical
variable is processed through its own forward pass"). There is therefore
no native channel to inject a demonstration's *label* into the model
input directly. Das et al.'s "labelled time-series examples ... appended
to the model input" is implemented here as literal numeric concatenation:
for a given clinical variable, each selected demonstration's own observed
series for that variable is concatenated ahead of the query patient's own
context series for that variable, in randomised order, and the whole
concatenation is forecast forward as a single context. The demonstration's
label itself is used ONLY downstream, by few_shot_pipeline.py, to fit
Approach A's probe on each demonstration's own (zero-shot) forecast --
never fed into TimesFM as a numeric value. This keeps the "label" concept
where it can actually be used (a discriminative classifier fit outside
TimesFM) while still letting the demonstrations' raw trajectory SHAPES
condition TimesFM's own forecast of the query series, which is the only
lever TimesFM's architecture actually exposes.

The query's own context is always placed LAST in the concatenation (i.e.
immediately preceding the forecast horizon) since TimesFM is decoder-only/
causal — only what precedes a position can influence it, so the model's
next-token predictions naturally extend from the tail of the sequence,
which must be the query's own recent history, not a demonstration's.
"""

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
    """
    Randomises demonstration order for one ICL prompt (one query, one
    replicate/seed) — the "demonstration ordering ... randomised across
    replicate runs" control from Section 3.3. Returns a NEW list; does not
    mutate the input (callers may reuse the same selected-demo list across
    multiple orderings, e.g. the random-selection regime's fixed demo set
    reused across every query in a replicate — see few_shot_pipeline.py).

    Ordering is applied ONCE per call, as a single ordering over the k
    demonstrations as whole patients, then reused identically for every
    clinical variable of that query — a "demonstration" is one patient's
    worth of context, not a per-variable draw, so its position in the
    prompt should be consistent across variables rather than re-shuffled
    per variable (which would make "demonstration order" an incoherent
    concept once you have more than one variable).
    """
    ordered = list(demonstrations)
    rng.shuffle(ordered)
    return ordered


def build_icl_context(
    query_context: np.ndarray,
    demo_series: Sequence[np.ndarray],
    max_context: int,
) -> tuple[np.ndarray, int]:
    """
    Concatenates demo_series (already NaN-dropped, in the caller's chosen
    order) ahead of query_context (already NaN-dropped and horizon-split
    via split_context_and_horizon), then enforces `max_context`.

    Truncation policy (front-truncation of the DEMO block, never of the
    query's own context): the query patient's own recent history is what
    the forecast is actually being made for and is always the closest
    material to the point being predicted, so it is never dropped except
    in the pathological case where query_context alone already exceeds
    max_context (handled the same way zero-shot forecasting would have to
    handle it — truncate the query's own context to its most recent
    max_context points, no demonstration budget remains). Otherwise, the
    demo block is truncated from its FRONT (its oldest points / earliest-
    ordered demonstrations first) so that whatever demo material survives
    is the material immediately preceding the query context — the
    positions closest to the query in the sequence, which matter most for
    a causal-attention model's most recent context window.

    Returns (concatenated_context, n_demo_points_used) — the latter is
    recorded on the resulting ForecastResult for diagnostics (e.g.
    confirming a dry run's "prompt format does not silently truncate" per
    the proposal's own Phase 3 milestone description).
    """
    query_context = np.asarray(query_context, dtype=np.float32)
    if query_context.size >= max_context:
        # Pathological case: nothing left for demonstrations at all.
        return query_context[-max_context:], 0

    demo_budget = max_context - query_context.size
    if demo_series:
        demo_concat = np.concatenate([np.asarray(s, dtype=np.float32) for s in demo_series])
    else:
        demo_concat = np.zeros(0, dtype=np.float32)

    if demo_concat.size > demo_budget:
        demo_concat = demo_concat[-demo_budget:]  # keep the END of the demo
                                                    # block (closest to the
                                                    # query), drop the START

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
    """
    Few-shot equivalent of zero_shot_pipeline.forecast_patient_variables: builds
    a per-variable ICL-conditioned context (demonstrations prepended, per
    build_icl_context above) for every variable in variable_order, and
    forecasts all of them in a single batched model call.

    `demonstrations` should already be the SELECTED k demonstrations for
    this query (demo_selection.py's job) — this function only orders and
    formats them, it does not select them.

    Mirrors forecast_patient_variables' own skip rule: a variable is
    skipped entirely (not forecast) if the QUERY's own context has fewer
    than 2 non-NaN points — same reasoning (nothing to fit a trend to, and
    degenerate 1-point contexts forecast poorly in practice), independent
    of how much demonstration material is available for that variable.

    Per-variable, demonstrations lacking any non-NaN observations for that
    particular variable contribute nothing to that variable's ICL context
    (silently — there is no "gap" placeholder in a purely numeric
    concatenation) rather than being dropped from the demonstration set
    entirely; a demo missing e.g. bilirubin can still contribute its heart
    rate series to that variable's context. `demonstrations_used` on each
    returned ForecastResult reports how many of the k demonstrations
    actually contributed at least one point to THAT variable's context —
    this can be less than k, and callers relying on "exactly k demos were
    used" for a specific variable should check this field rather than
    assuming it.
    """
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

    # forecast_batch hardcodes demonstrations_used=0 (it has no concept of
    # ICL — that's exactly the layering this module exists to add "one
    # layer up", per timesfm_loader.py's own module docstring). Patch it
    # here rather than touching timesfm_loader.py, keeping that module
    # zero-shot-only and adaptation-regime-agnostic as designed.
    patched = {}
    for result, n_used in zip(results, demo_counts_used):
        patched[result.variable_name] = replace(result, demonstrations_used=n_used)
    return patched

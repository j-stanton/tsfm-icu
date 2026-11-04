"""
Phase 3: few-shot in-context-learning evaluation pipeline (proposal Section
3.3/3.4, Chapter 4 Milestone 3).

Few-shot equivalent of zero_shot/zero_shot_pipeline.py's run_zero_shot_evaluation.
Deliberately a SEPARATE function rather than a k=0 special case bolted onto
the zero-shot one: the two regimes differ in where the probe is fit (a
held-out validation split for zero-shot vs. the k demonstrations themselves
for few-shot, per Section 3.3: "for zero-shot the probe is trained on a
validation calibration set, for few-shot it is trained on the k
demonstrations"), in where the demonstration pool comes from (the TRAINING
split only, per demo_selection.py's own docstring: "no test patient is ever
used as a demonstration for another test patient"), and now also in how
TimesFM itself is called (forecast_patient_variables_icl, not
forecast_patient_variables). Sharing one function with a k==0 branch would
mean threading all of that through conditionals; two functions with the
same result-table shape (EvaluationResult, imported from zero_shot_pipeline.py)
is more readable and matches how the proposal itself frames zero-shot and
few-shot as two named "adaptation regimes", not one regime with a
parameter.

k=0 is NOT accepted here (raises ValueError) — that IS the zero-shot
regime and belongs to zero_shot/run_zero_shot_evaluation, not this module.

This module does NOT import anything from ../zero_shot/ — per an explicit
project-structure decision (few_shot/ is a self-contained sibling folder,
same pattern used to set up zero_shot/ itself), the small dispatch helpers
zero_shot_pipeline.py defines for fixed-vs-rolling task handling
(_get_candidate_ids, _demonstration_for_id) are duplicated here rather
than cross-imported. They are copied verbatim/unchanged from
zero_shot/zero_shot_pipeline.py; if that dispatch logic changes there, it should
be changed here too.
"""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from demo_selection import (
    Demonstration,
    select_random_demonstrations,
    select_similarity_demonstrations,
)
from forecast_to_label import (
    forecast_to_probe_features,
    fit_few_shot_probe,
    ProbeClassifier,
    TASK_DECISION_RULES,
)
from timesfm_icl import forecast_patient_variables_icl
from timesfm_loader import TimesFMForecaster, ForecastResult
from yaib_data import (
    YAIBCohort,
    stay_to_demonstration,
    stay_hour_to_demonstration,
    split_context_and_horizon,
    TASK_HORIZON_HOURS,
)


# --- duplicated from zero_shot/zero_shot_pipeline.py (see module docstring) -----

YAIB_TO_RULE_VARNAME = {
    "crea": "creatinine",
    "bili": "bilirubin",
    "plt": "platelets",
    "map": "map",
}


def _get_candidate_ids(cohort: YAIBCohort, split: str | None, cap: int | None):
    if cohort.task_kind == "rolling":
        ids = cohort.stay_hour_ids(split)
    else:
        ids = cohort.stay_ids(split)
    if cap is not None:
        ids = ids[:cap]
    return ids


def _demonstration_for_id(cohort: YAIBCohort, candidate_id) -> Demonstration:
    if cohort.task_kind == "rolling":
        stay_id, time = candidate_id
        return stay_hour_to_demonstration(cohort, stay_id, time)
    return stay_to_demonstration(cohort, candidate_id)


# --- result types (structurally identical to zero_shot_pipeline.py's, kept as
# separate definitions for the same self-containment reason as above) -----

@dataclass
class FewShotPatientPrediction:
    """
    One patient's few-shot prediction. Adds k/strategy/n_demos_available
    on top of zero_shot_pipeline.PatientPrediction's fields — useful for
    debugging a specific (task, k, strategy) cell of the results factorial
    without re-running it, and for catching the case where fewer than k
    demonstrations were actually available (a small training pool, or a
    rolling task's train split producing few candidate patient-hours).
    """
    patient_id: str
    true_label: int
    k_requested: int
    k_available: int
    strategy: str
    probe_score: float | None = None
    probe_calibrated: bool | None = None  # None when approach != "probe";
                                           # True/False otherwise — whether
                                           # probe_score is Platt-scaled or
                                           # ProbeClassifier's uncalibrated
                                           # fallback (see forecast_to_label.
                                           # py's ProbeClassifier.is_calibrated)
    rule_score: float | None = None
    forecasts: dict[str, ForecastResult] = field(default_factory=dict)


@dataclass
class FewShotEvaluationResult:
    task_name: str
    approach: str
    k: int
    strategy: str
    seed: int
    auroc: float
    auprc: float
    ece_before_calibration: float
    n_patients: int
    n_train_pool: int
    probe_degenerate: bool  # True if Approach A's probe could not be fit
                             # for this replicate (see _fit_probe_on_demos)
                             # and every prediction fell back to the
                             # neutral 0.5 score — flagged explicitly so
                             # this doesn't silently masquerade as a real
                             # (bad) result in the final tables.
    probe_uncalibrated: bool  # True if approach="probe" and AT LEAST ONE
                               # prediction in this replicate used
                               # ProbeClassifier's uncalibrated fallback
                               # (see ProbeClassifier.is_calibrated) rather
                               # than Platt-scaled probabilities — expected
                               # to be True for essentially every real
                               # (task, k<=20) probe replicate, per
                               # ProbeClassifier's documented
                               # MIN_CLASS_COUNT_FOR_CALIBRATION fallback.
                               # False for approach="rule" (not applicable)
                               # and for the (rare) probe replicate where
                               # every draw happened to have >=5 per class.
    predictions: list[FewShotPatientPrediction]


def build_demonstration_pool(
    cohort: YAIBCohort,
    split: str = "train",
    max_pool: int | None = None,
) -> list[Demonstration]:
    """
    Materialises the demonstration pool as Demonstration objects, from
    `split` (must be "train" — see module docstring / demo_selection.py's
    own "no test patient is ever used as a demonstration" requirement;
    Section 3.3: "k labelled demonstrations from the training cohort").

    Cheap relative to a TimesFM forecast call — this only reads/reshapes
    already-loaded YAIB dataframes (stay_to_demonstration /
    stay_hour_to_demonstration do no model inference), so materialising
    the WHOLE pool is fine even for large training splits. `max_pool` is
    still exposed and honoured, primarily so similarity-based selection's
    per-query O(pool size) cosine-similarity scan (demo_selection.
    select_similarity_demonstrations) has an explicit, documented way to
    be capped during a dry run — the same class of "looks hung, isn't"
    problem Phase 2's uncapped calibration loop hit is possible here for
    a large pool x large eval-set combination, even though no GPU work is
    involved.
    """
    if split != "train":
        raise ValueError(
            f"Demonstration pool must be built from split='train' "
            f"(got '{split}') — using val/test patients as few-shot "
            f"demonstrations would violate the no-test-leakage "
            f"requirement documented in demo_selection.py."
        )
    ids = _get_candidate_ids(cohort, split, max_pool)
    return [_demonstration_for_id(cohort, cid) for cid in ids]


def _demo_probe_features(
    forecaster: TimesFMForecaster,
    demo: Demonstration,
    variable_order: Sequence[str],
    horizon: int,
) -> np.ndarray | None:
    """
    Forecasts ONE demonstration's own series zero-shot (no ICL context —
    this is deliberately a plain forecast, see _fit_probe_on_demos'
    docstring for why) and returns its fixed-length probe-feature vector,
    or None if every variable had fewer than 2 usable context points
    (mirrors zero_shot_pipeline.forecast_patient_variables' own skip rule).

    Imported lazily from phase2-equivalent logic duplicated here rather
    than via cross-import — build_fixed_length_probe_features itself is
    reimplemented inline below (N_PROBE_FEATURES_PER_VARIABLE=5, matching
    forecast_to_probe_features' output shape) since it's an 8-line
    function, not worth a separate module split for one caller.
    """
    var_names, contexts = [], []
    for var in variable_order:
        series = demo.dynamic_series.get(var, np.array([]))
        series = series[~np.isnan(series)]
        context, _horizon_tail = split_context_and_horizon(series, horizon)
        if context.size < 2:
            continue
        var_names.append(var)
        contexts.append(context)

    if not var_names:
        return None

    results = forecaster.forecast_batch(var_names, contexts, horizon=horizon)
    forecasts = {r.variable_name: r for r in results}

    blocks = []
    for var in variable_order:
        if var in forecasts:
            blocks.append(forecast_to_probe_features(forecasts[var]))
        else:
            blocks.append(np.zeros(5, dtype=np.float32))
    return np.concatenate(blocks)


def _fit_probe_on_demos(
    forecaster: TimesFMForecaster,
    demonstrations: Sequence[Demonstration],
    variable_order: Sequence[str],
    horizon: int,
) -> ProbeClassifier | None:
    """
    Approach A, few-shot: "the probe is trained on the k demonstrations"
    (Section 3.3). Each demonstration is forecast ZERO-SHOT on its own
    series (NOT ICL-conditioned on the other k-1 demonstrations) to
    produce its probe-feature vector; the probe is then fit on these k
    (features, label) pairs. This deliberately does not recursively apply
    ICL when building the demonstrations' own features — the ICL context
    (build_icl_context / forecast_patient_variables_icl) is reserved for
    conditioning the QUERY patient's forecast, which is the actual
    adaptation mechanism Das et al. [2024] describe; the demonstrations
    themselves are the "known" examples the probe learns from, and
    forecasting them plainly keeps their probe-feature representation
    consistent with how the validation calibration set is forecast in the
    zero-shot regime (also plain, unconditioned forecasts) — same
    representational basis in both regimes, which matters since the same
    ProbeClassifier / forecast_to_probe_features code path scores both.

    Returns None (a "cannot fit" signal, NOT an exception) when the
    selected demonstrations contain fewer than 2 distinct labels — this
    is a documented, expected edge case at small k (fit_few_shot_probe's
    own docstring already flags k=1 as unfittable: a sigmoid calibrator
    cannot be fit to one point, and LogisticRegression itself requires 2+
    classes). Random-stratified selection guarantees >=1 positive at
    k>=2 but does NOT guarantee >=1 negative, so this can in principle
    still happen at k=2 with an unlucky draw against the real ~5-10%
    positive-rate cohorts here; callers must handle None by falling back
    to a neutral score (see run_few_shot_evaluation) rather than treating
    it as a crash.
    """
    demo_X, demo_y = [], []
    for demo in demonstrations:
        feats = _demo_probe_features(forecaster, demo, variable_order, horizon)
        if feats is None:
            continue
        demo_X.append(feats)
        demo_y.append(demo.label)

    if len(demo_X) < 2 or len(set(demo_y)) < 2:
        return None

    try:
        return fit_few_shot_probe(np.stack(demo_X), np.array(demo_y))
    except ValueError:
        # Genuinely unfittable case (NOT the ordinary small-k-per-class
        # case, which is now handled INSIDE ProbeClassifier.fit() itself
        # — see MIN_CLASS_COUNT_FOR_CALIBRATION in forecast_to_label.py,
        # which skips Platt-scaling rather than raising whenever the
        # minority class has <5 members, and still returns a real,
        # discriminative uncalibrated score). This except clause is now a
        # defensive fallback for whatever ValueError sklearn might still
        # raise outside that specific, already-handled path (e.g. a
        # degenerate feature matrix) — signal "cannot fit" the same way
        # the len(set(demo_y))<2 check above does, rather than crashing
        # the whole replicate over one query's bad luck.
        return None


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Identical to zero_shot_pipeline.compute_expected_calibration_error — see
    that module for the full rationale; duplicated here for the same
    self-containment reason as the dispatch helpers above."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bin_edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def _rule_score_for_forecasts(task_name: str, demo: Demonstration, forecasts: dict) -> float:
    """
    Approach B applied to an ICL-conditioned forecast. Identical logic to
    zero_shot_pipeline.run_zero_shot_evaluation's inline rule branch (including
    the empty-slice-NaN AKI baseline-creatinine bugfix) — demonstrations
    only affect the SCORE indirectly, by having conditioned the forecast
    itself; the rule function applied to that forecast is exactly the
    same one used in the zero-shot regime, per Section 3.3 ("the forecast
    is mapped to a binary score by applying a clinical rule directly to
    the predicted trajectory" — this is regime-agnostic by construction).
    """
    rule = TASK_DECISION_RULES[task_name]
    rule_forecasts = {
        YAIB_TO_RULE_VARNAME[v]: forecasts[v]
        for v in forecasts
        if v in YAIB_TO_RULE_VARNAME
    }
    if task_name == "aki":
        crea_series = demo.dynamic_series.get("crea", np.array([]))
        crea_nonnan = crea_series[~np.isnan(crea_series)]
        baseline = float(crea_nonnan[0]) if crea_nonnan.size > 0 else 1.0
        return rule(
            rule_forecasts.get("creatinine", next(iter(forecasts.values()))),
            baseline,
        )
    elif task_name == "sepsis":
        return rule(rule_forecasts, baseline_partial_sofa=0.0)
    raise ValueError(f"No Approach-B rule wired up for task '{task_name}'")


def run_few_shot_evaluation(
    cohort: YAIBCohort,
    forecaster: TimesFMForecaster,
    task_name: str,
    approach: str,
    k: int,
    strategy: str,
    seed: int,
    demonstration_pool: list[Demonstration] | None = None,
    eval_split: str = "test",
    max_patients: int | None = None,
    max_train_pool: int | None = None,
) -> FewShotEvaluationResult:
    """
    Runs ONE (task, approach, k, strategy, seed) cell of Section 3.3's
    "three-way (k x selection x task) factorial with five seeds" — the
    seed loop and cross-cell aggregation belong one layer up (see
    run_few_shot.py), matching how run_zero_shot_evaluation is
    itself a single-seed function in Phase 2.

    strategy="random": ONE set of k demonstrations is drawn ONCE for this
        entire (task, k, seed) replicate and reused for EVERY query
        patient in eval_split — this is a deliberate reading of Section
        3.4's "seeds ... controlling ... the random sampling of
        demonstrations in the random-selection regime": the sampling
        event is a property of the REPLICATE (one random draw per seed),
        not of each individual query. Approach A's probe (if any) is
        correspondingly fit ONCE per replicate too, since it depends only
        on this fixed demo set.

    strategy="similarity": demonstrations are selected FRESH per query
        (each query's own k nearest neighbours in summary-feature space —
        select_similarity_demonstrations takes a `query` argument for
        exactly this reason), since there is no single well-defined "the
        demonstrations for this replicate" independent of which patient
        is being scored. Approach A's probe is correspondingly refit per
        query too. This is more expensive (k extra forecast_batch calls
        PER eval query, not just k per replicate) — max_patients should
        be set conservatively for a similarity-strategy dry run.

    Demonstration ORDER (within the prompt, not which demos are chosen)
    is always randomised per query regardless of strategy — see
    timesfm_icl.order_demonstrations — since order sensitivity is a
    property of the transformer's attention over the concatenated
    sequence, independent of how the demo set itself was chosen.
    """
    if k <= 0:
        raise ValueError(
            f"run_few_shot_evaluation requires k >= 1 (got k={k}); k=0 IS "
            f"the zero-shot regime — use "
            f"zero_shot/zero_shot_pipeline.run_zero_shot_evaluation for that."
        )
    if strategy not in ("random", "similarity"):
        raise ValueError(f"strategy must be 'random' or 'similarity', got '{strategy}'")
    if approach == "rule" and task_name not in TASK_DECISION_RULES:
        raise ValueError(
            f"No Approach-B decision rule registered for task '{task_name}' "
            f"(available: {list(TASK_DECISION_RULES.keys())}). This mirrors "
            f"the zero-shot regime's identical restriction — mortality has "
            f"no Approach-B rule by design; use approach='probe'."
        )

    horizon = TASK_HORIZON_HOURS.get(task_name)
    if horizon is None:
        raise ValueError(
            f"Unknown task_name '{task_name}'; expected one of "
            f"{list(TASK_HORIZON_HOURS.keys())}"
        )

    if demonstration_pool is None:
        demonstration_pool = build_demonstration_pool(
            cohort, split="train", max_pool=max_train_pool
        )
    if len(demonstration_pool) == 0:
        raise RuntimeError(
            f"Demonstration pool for task '{task_name}' is empty (train "
            f"split produced 0 candidates) — cannot run few-shot ICL. "
            f"Check cohort.stay_ids()/stay_hour_ids() against the real "
            f"split assignment file."
        )

    rng = np.random.default_rng(seed)
    eval_ids = _get_candidate_ids(cohort, eval_split, max_patients)

    # --- random-strategy: ONE demo set + ONE probe for the whole replicate
    fixed_demos: list[Demonstration] | None = None
    fixed_probe: ProbeClassifier | None = None
    if strategy == "random":
        fixed_demos = select_random_demonstrations(
            demonstration_pool, k=k, rng=rng, stratify_on_label=True
        )
        if approach == "probe":
            fixed_probe = _fit_probe_on_demos(forecaster, fixed_demos, cohort.variable_order, horizon)

    probe_degenerate = strategy == "random" and approach == "probe" and fixed_probe is None
    probe_uncalibrated = (
        strategy == "random" and approach == "probe"
        and fixed_probe is not None and not fixed_probe.is_calibrated
    )

    predictions: list[FewShotPatientPrediction] = []
    # Progress heartbeat every 100 eval candidates -- same reasoning as
    # zero_shot/zero_shot_pipeline.py's calibration-loop logging (Phase 2 added
    # this after an uncapped loop over thousands of patients looked
    # indistinguishable from a hang during real debugging). It matters
    # MORE here: a single seed's pass over the full test split (~5,762
    # patients for mortality, more for aki/sepsis's rolling tasks) can
    # itself run long, especially strategy="similarity" (per-query probe
    # refit) -- without this, a job's log can show the
    # "[task/approach/strategy/k] starting" line and then nothing for the
    # entire length of that one seed, which is what made the first real
    # full-run attempt (job 61982, killed at the 24h walltime) show zero
    # progress despite the run almost certainly making real headway.
    for i, candidate_id in enumerate(eval_ids):
        if i > 0 and i % 100 == 0:
            print(f"      [{task_name}/{approach}/{strategy}/k={k}] "
                  f"{i}/{len(eval_ids)} eval candidates processed", flush=True)
        query_demo = _demonstration_for_id(cohort, candidate_id)

        if strategy == "similarity":
            demos = select_similarity_demonstrations(
                query_demo, demonstration_pool, k=k, variable_order=cohort.variable_order
            )
            probe = None
            if approach == "probe":
                probe = _fit_probe_on_demos(forecaster, demos, cohort.variable_order, horizon)
                if probe is None:
                    probe_degenerate = True  # sticky across the loop: at least
                                              # one query hit the degenerate
                                              # case, worth flagging on the
                                              # whole replicate's result even
                                              # if other queries fit fine
                elif not probe.is_calibrated:
                    probe_uncalibrated = True  # sticky, same reasoning as
                                                # probe_degenerate above
        else:
            demos = fixed_demos
            probe = fixed_probe

        forecasts = forecast_patient_variables_icl(
            forecaster, query_demo, demos, cohort.variable_order, horizon, rng
        )

        pred = FewShotPatientPrediction(
            patient_id=query_demo.patient_id,
            true_label=query_demo.label,
            k_requested=k,
            k_available=len(demos),
            strategy=strategy,
            forecasts=forecasts,
        )

        if not forecasts:
            # Same neutral-score convention as zero_shot_pipeline.py's zero-shot
            # loop: no usable series for this patient, recorded rather than
            # silently dropped.
            pred.probe_score = 0.5
            pred.rule_score = 0.5
            predictions.append(pred)
            continue

        if approach == "probe":
            if probe is not None:
                blocks = []
                for var in cohort.variable_order:
                    if var in forecasts:
                        blocks.append(forecast_to_probe_features(forecasts[var]))
                    else:
                        blocks.append(np.zeros(5, dtype=np.float32))
                feats = np.concatenate(blocks)
                pred.probe_score = float(probe.predict_proba(feats.reshape(1, -1))[0])
                pred.probe_calibrated = probe.is_calibrated
            else:
                # Degenerate demo label set (see _fit_probe_on_demos'
                # docstring) — no discriminative classifier can be fit from
                # a single class. Neutral fallback, matching the documented
                # "fall back to ... an uncalibrated raw score for small k"
                # guidance already recorded in forecast_to_label.py.
                pred.probe_score = 0.5
                pred.probe_calibrated = None

        if approach == "rule":
            pred.rule_score = _rule_score_for_forecasts(task_name, query_demo, forecasts)

        predictions.append(pred)

    y_true = np.array([p.true_label for p in predictions])
    score_key = "probe_score" if approach == "probe" else "rule_score"
    y_score = np.array([getattr(p, score_key) for p in predictions])

    auroc = float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan")
    auprc = float(average_precision_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan")
    ece = compute_expected_calibration_error(y_true, y_score)

    return FewShotEvaluationResult(
        task_name=task_name,
        approach=approach,
        k=k,
        strategy=strategy,
        seed=seed,
        auroc=auroc,
        auprc=auprc,
        ece_before_calibration=ece,
        n_patients=len(predictions),
        n_train_pool=len(demonstration_pool),
        probe_degenerate=probe_degenerate,
        probe_uncalibrated=probe_uncalibrated,
        predictions=predictions,
    )

"""Few-shot TimesFM evaluation with k demonstrations, using random or similarity-based selection."""

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


# copied from zero_shot/zero_shot_pipeline.py

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


# result types

@dataclass
class FewShotPatientPrediction:
    """One patient's few-shot prediction."""
    patient_id: str
    true_label: int
    k_requested: int
    k_available: int
    strategy: str
    probe_score: float | None = None
    probe_calibrated: bool | None = None  # None for the rule
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
    probe_degenerate: bool  # probe could not be fitted
    probe_uncalibrated: bool  # at least one probe skipped Platt scaling
    predictions: list[FewShotPatientPrediction]


def build_demonstration_pool(
    cohort: YAIBCohort,
    split: str = "train",
    max_pool: int | None = None,
) -> list[Demonstration]:
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
        return None


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
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
    """Approach B applied to an ICL-conditioned forecast."""
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

    # random selection: one demonstration set and one probe per replicate
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
                    probe_degenerate = True
                elif not probe.is_calibrated:
                    probe_uncalibrated = True
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

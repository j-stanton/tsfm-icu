"""Zero-shot TimesFM evaluation: forecast, map to a score, compute AUROC, AUPRC and ECE."""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score

from demo_selection import Demonstration
from forecast_to_label import (
    forecast_to_probe_features,
    fit_zero_shot_probe,
    ProbeClassifier,
    TASK_DECISION_RULES,
)
from timesfm_loader import TimesFMForecaster, TimesFMRunConfig, ForecastResult
from yaib_data import (
    YAIBCohort,
    stay_to_demonstration,
    stay_hour_to_demonstration,
    split_context_and_horizon,
    TASK_HORIZON_HOURS,
)


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


@dataclass
class PatientPrediction:
    patient_id: str
    true_label: int
    probe_score: float | None = None   # Approach A output, if computed
    rule_score: float | None = None
    forecasts: dict[str, ForecastResult] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    task_name: str
    approach: str  # "probe" or "rule"
    auroc: float
    auprc: float
    ece_before_calibration: float
    ece_after_calibration: float | None
    n_patients: int
    predictions: list[PatientPrediction]


def forecast_patient_variables(
    forecaster: TimesFMForecaster,
    demo: Demonstration,
    variable_order: Sequence[str],
    horizon: int,
) -> dict[str, ForecastResult]:
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
        return {}

    results = forecaster.forecast_batch(var_names, contexts, horizon=horizon)
    return {r.variable_name: r for r in results}


N_PROBE_FEATURES_PER_VARIABLE = 5


def build_fixed_length_probe_features(
    forecasts: dict[str, ForecastResult],
    variable_order: Sequence[str],
) -> np.ndarray:
    blocks = []
    for var in variable_order:
        if var in forecasts:
            blocks.append(forecast_to_probe_features(forecasts[var]))
        else:
            blocks.append(np.zeros(N_PROBE_FEATURES_PER_VARIABLE, dtype=np.float32))
    return np.concatenate(blocks)


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


def run_zero_shot_evaluation(
    cohort: YAIBCohort,
    forecaster: TimesFMForecaster,
    task_name: str,
    approach: str = "probe",
    calibration_split: str = "val",
    eval_split: str = "test",
    max_patients: int | None = None,
    max_calibration_patients: int | None = None,
) -> EvaluationResult:
    if approach == "rule" and task_name not in TASK_DECISION_RULES:
        raise ValueError(
            f"No Approach-B decision rule registered for task '{task_name}' "
            f"(available: {list(TASK_DECISION_RULES.keys())}). This is a "
            f"deliberate scope decision for mortality — see "
            f"forecast_to_label.py — not a missing implementation; use "
            f"approach='probe' for mortality."
        )

    horizon = TASK_HORIZON_HOURS.get(task_name)
    if horizon is None:
        raise ValueError(
            f"Unknown task_name '{task_name}'; expected one of "
            f"{list(TASK_HORIZON_HOURS.keys())}"
        )

    cal_cap = max_calibration_patients if max_calibration_patients is not None else max_patients
    cal_ids = _get_candidate_ids(cohort, calibration_split, cal_cap)
    eval_ids = _get_candidate_ids(cohort, eval_split, max_patients)

    predictions: list[PatientPrediction] = []

    # --- Approach A: fit the probe on the calibration split's forecasts ---
    probe: ProbeClassifier | None = None
    if approach == "probe":
        cal_X, cal_y = [], []
        for i, candidate_id in enumerate(cal_ids):
            if i > 0 and i % 100 == 0:
                print(f"    [calibration] {i}/{len(cal_ids)} candidates processed "
                      f"({len(cal_X)} usable so far)")
            demo = _demonstration_for_id(cohort, candidate_id)
            forecasts = forecast_patient_variables(
                forecaster, demo, cohort.variable_order, horizon
            )
            if not forecasts:
                continue
            feats = build_fixed_length_probe_features(forecasts, cohort.variable_order)
            cal_X.append(feats)
            cal_y.append(demo.label)
        if len(cal_X) < 2 or len(set(cal_y)) < 2:
            raise RuntimeError(
                f"Calibration split for task '{task_name}' produced "
                f"{len(cal_X)} usable patients (out of {len(cal_ids)} "
                f"candidate {'patient-hours' if cohort.task_kind == 'rolling' else 'stay_ids'} "
                f"in split '{calibration_split}') with "
                f"{len(set(cal_y))} distinct label(s) — cannot fit a "
                f"zero-shot probe. If len(cal_ids) is large but len(cal_X) "
                f"is 0, every candidate patient's forecast_patient_variables "
                f"call returned an empty dict (all clinical variables had "
                f"fewer than 2 non-NaN observed points) — check "
                f"cohort.variable_order against what's actually populated "
                f"for these patients, not just present as a column. If "
                f"len(cal_ids) itself is small/zero, check "
                f"`calibration_split` name and cohort.stay_ids() output, "
                f"and whether max_calibration_patients truncated too "
                f"aggressively."
            )
        probe = fit_zero_shot_probe(np.stack(cal_X), np.array(cal_y))

    # score the evaluation split
    for candidate_id in eval_ids:
        demo = _demonstration_for_id(cohort, candidate_id)
        forecasts = forecast_patient_variables(
            forecaster, demo, cohort.variable_order, horizon
        )
        pred = PatientPrediction(
            patient_id=demo.patient_id, true_label=demo.label, forecasts=forecasts
        )

        if not forecasts:
            pred.probe_score = 0.5
            pred.rule_score = 0.5
            predictions.append(pred)
            continue

        if approach == "probe" and probe is not None:
            feats = build_fixed_length_probe_features(forecasts, cohort.variable_order)
            pred.probe_score = float(probe.predict_proba(feats.reshape(1, -1))[0])

        if approach == "rule":
            rule = TASK_DECISION_RULES[task_name]
            rule_forecasts = {
                YAIB_TO_RULE_VARNAME[v]: forecasts[v]
                for v in forecasts
                if v in YAIB_TO_RULE_VARNAME
            }
            if task_name == "aki":
                crea_series = demo.dynamic_series.get("crea", np.array([]))
                crea_nonnan = crea_series[~np.isnan(crea_series)]
                if crea_nonnan.size > 0:
                    baseline = float(crea_nonnan[0])
                else:
                    baseline = 1.0
                pred.rule_score = rule(
                    rule_forecasts.get("creatinine", next(iter(forecasts.values()))),
                    baseline,
                )
            elif task_name == "sepsis":
                pred.rule_score = rule(rule_forecasts, baseline_partial_sofa=0.0)

        predictions.append(pred)

    y_true = np.array([p.true_label for p in predictions])
    score_key = "probe_score" if approach == "probe" else "rule_score"
    y_score = np.array([getattr(p, score_key) for p in predictions])

    auroc = float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan")
    auprc = float(average_precision_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan")
    ece_before = compute_expected_calibration_error(y_true, y_score)

    return EvaluationResult(
        task_name=task_name,
        approach=approach,
        auroc=auroc,
        auprc=auprc,
        ece_before_calibration=ece_before,
        ece_after_calibration=None,
        n_patients=len(predictions),
        predictions=predictions,
    )

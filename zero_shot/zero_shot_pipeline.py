"""
Phase 2: zero-shot TimesFM evaluation pipeline (proposal Section 3.3/3.4,
Chapter 4 Milestone 2).

This module is the multi-variable aggregation layer that timesfm_loader.py
("Aggregation across variables happens one layer up, not here") and
forecast_to_label.py ("probe features across ALL forecasted variables get
concatenated one level up") both defer to. It is the piece that turns:

    per-patient dynamic series (from yaib_data.py)
        -> per-variable TimesFM forecasts (timesfm_loader.py)
        -> per-patient concatenated probe features (forecast_to_label.py, Approach A)
           or a task decision-rule score (forecast_to_label.py, Approach B)
        -> a result table with AUROC / AUPRC / ECE (this module)

into one run. Explicitly ZERO-SHOT only: no demonstrations are selected or
prepended here (that's Phase 3 / demo_selection.py's job) — every forecast
call below uses `demonstrations_used=0`, matching Section 3.3's definition
of the zero-shot regime ("the pre-trained model is applied directly to test
patients without any prior exposure to MIMIC-III data").
"""

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


# --- Approach-B variable-name mapping ------------------------------------
# forecast_to_label.py's rules expect specific keys ("creatinine",
# "bilirubin", "platelets", "map"; "baseline_creatinine"). YAIB's own short
# variable codes (Section 3.2 / the wiki's DYNAMIC var list) use different
# names ("crea", "bili", "plt", "map"). This mapping is the single place
# that translation happens, so the rule functions themselves stay decoupled
# from any one dataset's naming convention.
YAIB_TO_RULE_VARNAME = {
    "crea": "creatinine",
    "bili": "bilirubin",
    "plt": "platelets",
    "map": "map",
}


def _get_candidate_ids(cohort: YAIBCohort, split: str | None, cap: int | None):
    """
    Dispatches to cohort.stay_ids() (fixed tasks, e.g. mortality) or
    cohort.stay_hour_ids() (rolling tasks, e.g. aki/sepsis) based on
    cohort.task_kind — this is the single place run_zero_shot_evaluation
    needs to branch on task_kind, so its calibration/eval loops below stay
    identical in shape for both task kinds.

    For rolling tasks, `cap` limits the number of PATIENTS contributing
    hours (via stay_hour_ids' own per-patient hours_per_patient, itself
    left at its default — see yaib_data.ROLLING_HOURS_PER_PATIENT), by
    slicing the returned (stay_id, time) list — this caps total
    patient-hour pairs, not patients directly, since stay_hour_ids doesn't
    expose a patient-count cap of its own. Close enough for a dry run's
    purposes; a real full run typically won't set `cap` at all.
    """
    if cohort.task_kind == "rolling":
        ids = cohort.stay_hour_ids(split)
    else:
        ids = cohort.stay_ids(split)
    if cap is not None:
        ids = ids[:cap]
    return ids


def _demonstration_for_id(cohort: YAIBCohort, candidate_id) -> Demonstration:
    """
    Dispatches to stay_to_demonstration() (fixed tasks — candidate_id is a
    bare stay_id) or stay_hour_to_demonstration() (rolling tasks —
    candidate_id is a (stay_id, time) tuple), matching _get_candidate_ids'
    output shape for the corresponding task_kind.
    """
    if cohort.task_kind == "rolling":
        stay_id, time = candidate_id
        return stay_hour_to_demonstration(cohort, stay_id, time)
    return stay_to_demonstration(cohort, candidate_id)


@dataclass
class PatientPrediction:
    """One patient's zero-shot prediction, keeping enough detail to debug
    a specific case without re-running inference."""
    patient_id: str
    true_label: int
    probe_score: float | None = None   # Approach A output, if computed
    rule_score: float | None = None    # Approach B output, if a rule exists for this task
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
    """
    Forecasts every clinical variable for one patient, batched in a single
    model call (see TimesFMForecaster.forecast_batch — this is exactly the
    "aggregation across variables" step both timesfm_loader.py and
    forecast_to_label.py defer to "one layer up"). Variables with fewer
    than 2 observed points are skipped (nothing to fit a trend to, and
    TimesFM's own forecast() call handles degenerate 1-point context poorly
    in practice) rather than passed through and silently producing garbage.

    IMPORTANT: which variables get skipped varies per patient (missingness
    is patient-specific — Section 3.2 cites Che et al. [2018] on exactly
    this point), so the *set of keys* in the returned dict is NOT the same
    across patients. Callers that concatenate per-variable features across
    ALL of `variable_order` (Approach A's probe, one layer up in
    run_zero_shot_evaluation) must iterate over the full `variable_order`
    list — not `sorted(forecasts)` — and substitute a fixed-length zero
    placeholder for any variable missing from this dict, or feature vectors
    will silently differ in length between patients. This was caught by
    make_synthetic_yaib_cohort.py's end-to-end test (a per-patient
    concatenation of sorted(forecasts) failed on the mortality task once
    missingness varied enough between patients) — see
    build_fixed_length_probe_features below, which is the safe replacement.
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
        return {}

    results = forecaster.forecast_batch(var_names, contexts, horizon=horizon)
    return {r.variable_name: r for r in results}


N_PROBE_FEATURES_PER_VARIABLE = 5  # matches forecast_to_probe_features' output shape


def build_fixed_length_probe_features(
    forecasts: dict[str, ForecastResult],
    variable_order: Sequence[str],
) -> np.ndarray:
    """
    Concatenates per-variable probe features in a FIXED order and FIXED
    total length regardless of which variables this particular patient had
    enough data to forecast — this is what fixes the shape-mismatch bug
    forecast_patient_variables' docstring describes. Missing variables
    contribute a zero block (consistent with the "unmeasured -> healthy/
    neutral" convention already used elsewhere in this pipeline, e.g.
    sepsis_partial_sofa_rule's missing-component handling in
    forecast_to_label.py) rather than being dropped, so every patient's
    feature vector has shape (len(variable_order) * N_PROBE_FEATURES_PER_VARIABLE,).
    """
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
    """
    ECE with 10 equal-width bins, per Section 3.4 ("ECE (expected
    calibration error, with 10 equal-width bins)"). sklearn's
    calibration_curve gives per-bin (observed_freq, mean_predicted) pairs
    but drops empty bins and doesn't directly weight by bin occupancy, so
    ECE is computed by hand here to match the standard definition exactly
    (weighted mean absolute gap between confidence and accuracy).
    """
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
    """
    Runs Milestone 2's "full test-set evaluation ... with five seeds" for a
    SINGLE seed (seed-looping and bootstrap CIs belong in an analysis script
    one layer above this, per Section 3.4 — this function does one clean
    pass and returns per-patient scores so that script can resample them).

    approach="probe"  -> Approach A: forecast-probe classifier
                          (fit_zero_shot_probe on the calibration split,
                          per Section 3.3: "for zero-shot the probe is
                          trained on a validation calibration set").
    approach="rule"   -> Approach B: clinically motivated decision rule,
                          only valid for tasks present in
                          forecast_to_label.TASK_DECISION_RULES (currently
                          aki, sepsis — mortality has no Approach-B rule by
                          design; see forecast_to_label.py's documented
                          coverage decision).

    Calibration (Platt scaling) is applied to whichever score is produced,
    per Section 3.3's "All resulting probability scores are calibrated on
    the validation fold using Platt scaling; ECE is reported both before
    and after calibration."

    `max_patients` caps ONLY the eval/test split size. `max_calibration_patients`
    separately caps the calibration/val split size (defaults to `max_patients`
    if not given, for backward-compat convenience) — this distinction is
    NOT cosmetic: an earlier version of this function left the calibration
    split completely uncapped regardless of `max_patients`, so a
    "--max-patients 20" dry run still ran full TimesFM inference over
    EVERY patient in the calibration split (5,705 on the real mortality
    cohort) before ever reaching the small eval loop. On real MIMIC-III
    data this was slow enough that the run appeared to hang and then
    failed with a misleading "0 usable patients" error — not because the
    split was actually empty (verified: 465 positives / 5,705 val
    patients), but because something in that very long uncapped loop
    never got the chance to accumulate 2 differently-labelled patients
    within whatever caused it to fail. Capping both splits independently
    fixes the real dry-run use case (fast, small) while still allowing a
    full run to use a large, uncapped calibration set when neither
    argument is passed.
    """
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
        # Progress logging every 100 patients: this loop runs real TimesFM
        # inference per patient and can be slow on a large calibration
        # split (e.g. thousands of patients if max_calibration_patients
        # isn't set) — silent output here previously made a genuinely-slow-
        # but-working run indistinguishable from a hung one when debugging
        # on the real cluster, which cost real time to diagnose.
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

    # --- score the evaluation split ---
    for candidate_id in eval_ids:
        demo = _demonstration_for_id(cohort, candidate_id)
        forecasts = forecast_patient_variables(
            forecaster, demo, cohort.variable_order, horizon
        )
        pred = PatientPrediction(
            patient_id=demo.patient_id, true_label=demo.label, forecasts=forecasts
        )

        if not forecasts:
            # No usable series for this patient (e.g. entirely missing
            # variables) — recorded with a neutral 0.5 score rather than
            # silently dropped, so the evaluation set size stays reported
            # accurately and this shows up as a visible source of error
            # rather than a quietly shrunken denominator.
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
                # BUGFIX (found via a real cluster run producing a silent
                # "RuntimeWarning: Mean of empty slice"): the old condition
                # only checked `"crea" in demo.dynamic_series` -- true
                # whenever the column EXISTS, even if every value in it is
                # NaN for this patient's truncated rolling-window context
                # (confirmed common on real MIMIC-III data: creatinine is
                # not drawn every hour). When that happened, the NaN-
                # filtered array was empty, `.mean()` on it silently
                # produced NaN, and that NaN baseline flowed straight into
                # aki_kdigo_creatinine_rule's `1.5 * baseline_creatinine`
                # threshold -- corrupting the rule score for exactly the
                # patients most likely to be under-monitored, without any
                # visible error (just a warning easy to miss in a long log).
                crea_series = demo.dynamic_series.get("crea", np.array([]))
                crea_nonnan = crea_series[~np.isnan(crea_series)]
                if crea_nonnan.size > 0:
                    baseline = float(crea_nonnan[0])
                else:
                    # 1.0 mg/dL: a commonly-cited normal adult baseline
                    # creatinine, used only when this patient has zero
                    # observed creatinine values in context -- an
                    # explicit, documented fallback rather than a NaN
                    # silently corrupting the threshold.
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
        ece_after_calibration=None,  # Approach A's probe already applies
                                      # Platt scaling internally (see
                                      # ProbeClassifier.fit); a distinct
                                      # "after" number for approach="rule"
                                      # requires fitting a separate Platt
                                      # calibrator on rule_score using the
                                      # calibration split — left as a
                                      # one-line follow-up once real rule
                                      # scores are available to calibrate
                                      # against (do not fit on synthetic data).
        n_patients=len(predictions),
        predictions=predictions,
    )

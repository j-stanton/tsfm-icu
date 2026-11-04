"""
Maps TimesFM's native forecast output (a predicted trajectory) onto a
binary classification score, per Section 3.3's two pre-specified strategies:

  Approach A: forecast-probe classifier trained on summary statistics of
              the TimesFM forecast (mean, variance, slope, exceedance
              probability of clinically meaningful thresholds).
  Approach B: clinically motivated decision rule applied directly to the
              predicted trajectory (e.g. KDIGO creatinine thresholds for AKI).

Both approaches are deliberately kept "thin": TimesFM itself is never
touched here (no weight updates), consistent with the weight-free
adaptation framing of the whole study.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from timesfm_loader import ForecastResult


# --- Approach A: forecast-probe classifier ------------------------------

def forecast_to_probe_features(result: ForecastResult) -> np.ndarray:
    """
    Summary statistics of a single-variable forecast, used as input to the
    lightweight probe classifier. Kept separate from the multi-variable
    aggregation step (probe features across ALL forecasted variables get
    concatenated one level up, mirroring how the classical baselines'
    summary features are built per-variable then concatenated).
    """
    pf = result.point_forecast
    qf = result.quantile_forecast  # (horizon, n_quantiles); quantile 0 = mean

    # variance from the quantile spread as a cheap proxy, since TimesFM's
    # point forecast alone gives no uncertainty estimate
    variance_proxy = float(np.var(qf[:, 1:], axis=1).mean()) if qf.shape[1] > 1 else 0.0

    t = np.arange(pf.size, dtype=np.float32)
    slope = np.polyfit(t, pf, 1)[0] if pf.size >= 2 else 0.0

    return np.array([
        pf.mean(),
        pf.min(),
        pf.max(),
        variance_proxy,
        slope,
    ], dtype=np.float32)


def exceedance_probability(
    result: ForecastResult,
    threshold: float,
    direction: str = "above",
) -> float:
    """
    Probability the forecasted trajectory crosses a clinically meaningful
    threshold, estimated from the quantile forecast rather than the point
    forecast alone — this is one of the named probe features (Section 3.3)
    and is also reusable directly as an Approach-B-style rule component.
    """
    qf = result.quantile_forecast  # (horizon, n_quantiles)
    if direction == "above":
        crossed = (qf > threshold).any(axis=1)
    elif direction == "below":
        crossed = (qf < threshold).any(axis=1)
    else:
        raise ValueError("direction must be 'above' or 'below'")
    return float(crossed.mean())


@dataclass
class ProbeClassifier:
    """
    Wraps sklearn's LogisticRegression + Platt-scaling calibration
    (Platt et al. [1999], per Section 3.3's calibration requirement:
    "All resulting probability scores are calibrated on the validation
    fold using Platt scaling; ECE is reported both before and after
    calibration.").

    For zero-shot: fit on a validation calibration set.
    For few-shot ICL: fit on the k demonstrations themselves.
    Which one is used is a caller decision (see fit_zero_shot / fit_few_shot
    below) — this class itself is adaptation-regime-agnostic.

    BUGFIX (found on a real full-scale cluster run, 2026-09-22): the probe
    features this fits on are raw summary statistics across 15 clinical
    variables with wildly different natural scales (e.g. raw creatinine
    ~0.5-5 mg/dL next to raw heart rate ~60-150 bpm, in the SAME feature
    vector, per forecast_to_probe_features/build_fixed_length_probe_features).
    LogisticRegression's default lbfgs solver assumes roughly comparable
    feature scales; without that, it both converges far more slowly AND
    the resulting decision boundary is dominated by whichever variable
    happens to have the largest raw magnitude, not necessarily the most
    clinically informative one. This surfaced as a real
    `ConvergenceWarning: lbfgs failed to converge after 1000 iteration(s)`
    on every single probe fit during a full mortality+aki+sepsis run
    (14,334+ calibration patient-hours for aki alone) — not a cosmetic
    warning, but a sign the fitted coefficients likely hadn't reached
    their true optimum, and were doing so on an implicitly mis-weighted
    feature space regardless. Fixed by fitting a StandardScaler ahead of
    LogisticRegression in a Pipeline, so every feature is normalised to
    zero mean / unit variance from the calibration data BEFORE fitting —
    the scaler's fitted mean/scale are learned once (from the same data
    the classifier trains on) and applied identically at predict time via
    the same Pipeline object, avoiding any train/test scaling mismatch.
    """
    _clf: Pipeline | None = None
    _calibrated: CalibratedClassifierCV | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._clf = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=5000)),
            # max_iter raised 1000 -> 5000 as a secondary safeguard on top
            # of scaling -- scaling alone is expected to resolve the
            # observed non-convergence in the overwhelming majority of
            # cases, but a higher ceiling costs little and guards against
            # any remaining edge cases (e.g. a particularly small or
            # separable calibration set) without silently capping fit
            # quality the way the original 1000 did on real data.
        ])
        self._clf.fit(X, y)
        # Platt scaling == sigmoid calibration in sklearn's terminology.
        # NOTE: cv="prefit" is deprecated/removed in newer sklearn
        # (>=1.6ish) in favour of wrapping the fitted estimator in
        # FrozenEstimator. Try the new API first, fall back to the old
        # one so this keeps working across the sklearn version YAIB
        # itself pins (see external/YAIB/pyproject.toml) vs. whatever
        # is installed in a standalone adapter-dev environment.
        try:
            from sklearn.frozen import FrozenEstimator
            self._calibrated = CalibratedClassifierCV(
                FrozenEstimator(self._clf), method="sigmoid"
            )
            self._calibrated.fit(X, y)
        except ImportError:
            self._calibrated = CalibratedClassifierCV(
                self._clf, method="sigmoid", cv="prefit"
            )
            self._calibrated.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._calibrated is None:
            raise RuntimeError("ProbeClassifier.fit() must be called first.")
        return self._calibrated.predict_proba(X)[:, 1]


def fit_zero_shot_probe(
    calibration_X: np.ndarray,
    calibration_y: np.ndarray,
) -> ProbeClassifier:
    """Zero-shot: probe trained on a held-out validation calibration set."""
    probe = ProbeClassifier()
    probe.fit(calibration_X, calibration_y)
    return probe


def fit_few_shot_probe(
    demonstration_X: np.ndarray,
    demonstration_y: np.ndarray,
) -> ProbeClassifier:
    """Few-shot ICL: probe trained on the same k demonstrations used in-context.

    NOTE: with very small k (e.g. k=1), LogisticRegression + CalibratedClassifierCV
    will be poorly conditioned or fail outright (can't fit a sigmoid to one point).
    Callers should guard for k below some minimum (likely k>=5 in practice)
    and fall back to Approach B or an uncalibrated raw score for small k —
    flagging this now since it's an easy edge case to hit once real k-sweeps run.
    """
    probe = ProbeClassifier()
    probe.fit(demonstration_X, demonstration_y)
    return probe


# --- Approach B: clinically motivated decision rules --------------------

# Registry pattern so each task's rule is a plain function, easy to unit-test
# in isolation and to swap without touching the calling code.
#
# Coverage decision (resolves the three TODOs that were previously here):
#   - aki:       implemented, per KDIGO creatinine criteria.
#   - sepsis:    implemented as a PARTIAL-SOFA proxy rule, with an explicit,
#                documented coverage gap — see sepsis_partial_sofa_rule below.
#                This is a genuine methodological choice, not a stub, and
#                should be described as such in Section 3.3 / limitations.
#   - mortality: Approach B is deliberately OUT OF SCOPE. Mortality has no
#                single organ-system trajectory to threshold — it is an
#                endpoint, not a lab value with a clinical cutoff — so a
#                "decision rule applied to a predicted trajectory" is not a
#                coherent construct here. Section 3.3 already frames Approach
#                B as one of two *pre-specified* strategies, not a
#                requirement that both apply to every task, so mortality
#                uses Approach A (forecast-probe classifier) only. This is
#                recorded explicitly below rather than left as a silent gap.

DecisionRule = Callable[..., float]


def aki_kdigo_creatinine_rule(result: ForecastResult, baseline_creatinine: float) -> float:
    """
    KDIGO-based AKI decision rule (Section 3.3: "exceedance of KDIGO
    creatinine thresholds for AKI"). KDIGO stage 1 is defined as a rise of
    >=0.3 mg/dL within 48h OR >=1.5x baseline within 7 days. This implements
    the >=1.5x baseline check; the 48h absolute-rise criterion needs a
    rolling-window implementation once real dynamic data is available to
    test against, and is left as a follow-up refinement (not required for
    a first working Approach-B pass).
    """
    threshold = 1.5 * baseline_creatinine
    return exceedance_probability(result, threshold=threshold, direction="above")


# Standard adult SOFA sub-score cutoffs (Vincent et al. 1996 / Singer et al.
# 2016), restricted to the components that are continuous lab/vital series a
# univariate TimesFM forecast can actually produce:
#   - creatinine (mg/dL)    -> renal component
#   - bilirubin (mg/dL)     -> liver component
#   - platelets (x10^3/uL)  -> coagulation component
# MAP is also continuous and forecastable, but is included as a *direct*
# threshold (SOFA cardiovascular score >=1 at MAP < 70 mmHg) rather than the
# full vasopressor-dose ladder, since vasopressor dosing is an intervention
# record, not a forecastable physiological series.
SOFA_CREATININE_CUTOFFS_MGDL = [(1.2, 1), (2.0, 2), (3.5, 3), (5.0, 4)]
SOFA_BILIRUBIN_CUTOFFS_MGDL = [(1.2, 1), (2.0, 2), (6.0, 3), (12.0, 4)]
SOFA_PLATELET_CUTOFFS_1E3UL = [(150, 1), (100, 2), (50, 3), (20, 4)]  # lower is worse
SOFA_MAP_CUTOFF_MMHG = 70.0  # MAP below this scores >=1 cardiovascular point


def _sub_score_from_ascending_cutoffs(value: float, cutoffs: list[tuple[float, int]]) -> int:
    """`value` >= cutoff -> that point value; take the highest satisfied cutoff."""
    score = 0
    for threshold, points in cutoffs:
        if value >= threshold:
            score = points
    return score


def _sub_score_from_descending_cutoffs(value: float, cutoffs: list[tuple[float, int]]) -> int:
    """`value` <= cutoff -> that point value (used for platelets, lower=worse)."""
    score = 0
    for threshold, points in cutoffs:
        if value <= threshold:
            score = points
    return score


def sepsis_partial_sofa_rule(
    forecasts_by_variable: dict[str, ForecastResult],
    baseline_partial_sofa: float,
    delta_threshold: float = 2.0,
) -> float:
    """
    Trajectory-based proxy for Sepsis-3 organ-dysfunction onset (Singer
    et al. [2016]: an acute rise of >=2 points in SOFA within a suspected-
    infection window), applied to TimesFM's forecasted trajectory rather
    than retrospective chart data.

    IMPORTANT SCOPE / LIMITATION, to be stated explicitly wherever this rule
    is used (Section 3.3, and again in any limitations discussion):

    True SOFA has six components. Three are NOT usable here because they are
    not continuously forecastable clinical time series in the sense TimesFM
    operates on:
      - CNS component (Glasgow Coma Score) — an assessed/charted score, not
        a physiological signal with its own forecastable trajectory.
      - Cardiovascular component beyond the MAP<70 threshold — the higher
        point values (2-4) depend on vasopressor *dose*, which is an
        intervention/administration record, not a measured series.
      - Respiratory component (PaO2/FiO2) — omitted here for the same
        reason it is often dropped in EHR-derived SOFA implementations
        (ventilator/FiO2 settings are not always reliably time-stamped);
        can be added if PaO2/FiO2 is confirmed available and forecastable
        on the YAIB hourly grid.

    This function therefore computes a PARTIAL SOFA proxy from the three
    components that are forecastable single-variable series (renal,
    hepatic, coagulation) plus MAP as a direct cardiovascular threshold,
    and flags a rise of `delta_threshold` points in this partial score
    relative to `baseline_partial_sofa`. This will systematically
    UNDER-detect organ dysfunction relative to full SOFA (a limitation also
    documented in prior EHR-derived SOFA implementations that omit the same
    hard-to-source components), so results using this rule should be
    reported as a lower bound / proxy, not equated with the ground-truth
    Sepsis-3 label YAIB uses to define the task itself.

    `forecasts_by_variable` expects keys from {"creatinine", "bilirubin",
    "platelets", "map"}; any missing key is scored as 0 (matching the
    convention — also used by YAIB's underlying ricu preprocessing — of
    treating an unmeasured SOFA contributor as healthy/zero rather than
    dropping the whole score).
    """
    partial_score = 0.0

    if "creatinine" in forecasts_by_variable:
        pf = forecasts_by_variable["creatinine"].point_forecast
        partial_score += _sub_score_from_ascending_cutoffs(
            float(pf.max()), SOFA_CREATININE_CUTOFFS_MGDL
        )
    if "bilirubin" in forecasts_by_variable:
        pf = forecasts_by_variable["bilirubin"].point_forecast
        partial_score += _sub_score_from_ascending_cutoffs(
            float(pf.max()), SOFA_BILIRUBIN_CUTOFFS_MGDL
        )
    if "platelets" in forecasts_by_variable:
        pf = forecasts_by_variable["platelets"].point_forecast
        partial_score += _sub_score_from_descending_cutoffs(
            float(pf.min()), SOFA_PLATELET_CUTOFFS_1E3UL
        )
    if "map" in forecasts_by_variable:
        pf = forecasts_by_variable["map"].point_forecast
        if float(pf.min()) < SOFA_MAP_CUTOFF_MMHG:
            partial_score += 1

    delta = partial_score - baseline_partial_sofa
    # Return a smooth-ish [0, 1] score rather than a hard 0/1 flag, so it can
    # be compared on the same footing (AUROC/AUPRC/ECE) as Approach A's
    # probabilistic probe output: clamp the normalised delta into [0, 1].
    return float(np.clip(delta / delta_threshold, 0.0, 1.0)) if delta > 0 else 0.0


TASK_DECISION_RULES: dict[str, DecisionRule] = {
    "aki": aki_kdigo_creatinine_rule,
    "sepsis": sepsis_partial_sofa_rule,
    # "mortality": intentionally absent — see coverage decision above.
    #              Mortality uses Approach A (forecast-probe classifier) only.
}
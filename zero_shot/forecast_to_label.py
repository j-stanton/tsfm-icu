"""Turn TimesFM forecasts into a label score: a trained probe or a clinical rule."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from timesfm_loader import ForecastResult


# Approach A: forecast-probe classifier

def forecast_to_probe_features(result: ForecastResult) -> np.ndarray:
    """Summary statistics of a single-variable forecast, used as input to the lightweight probe classifier."""
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
    _clf: Pipeline | None = None
    _calibrated: CalibratedClassifierCV | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._clf = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=5000)),
        ])
        self._clf.fit(X, y)
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
    """Few-shot ICL: probe trained on the same k demonstrations used in-context."""
    probe = ProbeClassifier()
    probe.fit(demonstration_X, demonstration_y)
    return probe


# Approach B: clinically motivated decision rules


DecisionRule = Callable[..., float]


def aki_kdigo_creatinine_rule(result: ForecastResult, baseline_creatinine: float) -> float:
    threshold = 1.5 * baseline_creatinine
    return exceedance_probability(result, threshold=threshold, direction="above")


SOFA_CREATININE_CUTOFFS_MGDL = [(1.2, 1), (2.0, 2), (3.5, 3), (5.0, 4)]
SOFA_BILIRUBIN_CUTOFFS_MGDL = [(1.2, 1), (2.0, 2), (6.0, 3), (12.0, 4)]
SOFA_PLATELET_CUTOFFS_1E3UL = [(150, 1), (100, 2), (50, 3), (20, 4)]  # lower is worse
SOFA_MAP_CUTOFF_MMHG = 70.0


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
    return float(np.clip(delta / delta_threshold, 0.0, 1.0)) if delta > 0 else 0.0


TASK_DECISION_RULES: dict[str, DecisionRule] = {
    "aki": aki_kdigo_creatinine_rule,
    "sepsis": sepsis_partial_sofa_rule,
}

"""
Smoke test for the TSFM adapter scaffolding, using synthetic data.

Deliberately does NOT require the `timesfm` package to be installed —
it exercises demo_selection.py and forecast_to_label.py against
hand-built ForecastResult objects, standing in for what TimesFMForecaster
would produce. This lets the plumbing (selection -> feature extraction ->
probe/rule) be validated now, independent of the cluster/checkpoint-download
step, and independent of real cohort data being available yet.

Run with: python smoke_test.py
"""

import numpy as np

from demo_selection import (
    Demonstration,
    select_random_demonstrations,
    select_similarity_demonstrations,
    build_summary_feature_vector,
)
from forecast_to_label import (
    forecast_to_probe_features,
    exceedance_probability,
    fit_zero_shot_probe,
    fit_few_shot_probe,
    aki_kdigo_creatinine_rule,
    sepsis_partial_sofa_rule,
    TASK_DECISION_RULES,
)
from timesfm_loader import ForecastResult


def make_synthetic_demonstration(patient_id: str, label: int, rng: np.random.Generator) -> Demonstration:
    """Fabricates a plausible-shaped patient episode for plumbing tests only —
    NOT meant to resemble real clinical distributions."""
    n_hours = 24
    base = 90 if label == 0 else 70  # crude signal: "sicker" patients trend lower
    hr_series = base + rng.normal(0, 5, size=n_hours)
    creat_series = 1.0 + label * 0.4 + rng.normal(0, 0.1, size=n_hours)
    return Demonstration(
        patient_id=patient_id,
        label=label,
        dynamic_series={"hr": hr_series, "crea": creat_series},
        static_features={"age": float(rng.integers(40, 80))},
    )


def make_synthetic_forecast(variable_name: str, horizon: int, rng: np.random.Generator) -> ForecastResult:
    point = rng.normal(1.0, 0.2, size=horizon).astype(np.float32)
    quantiles = np.stack([point + rng.normal(0, 0.1, size=horizon) for _ in range(5)], axis=1)
    return ForecastResult(
        variable_name=variable_name,
        point_forecast=point,
        quantile_forecast=quantiles.astype(np.float32),
        context_len_used=48,
        demonstrations_used=0,
    )


def test_demonstration_selection():
    rng = np.random.default_rng(2222)
    pool = [make_synthetic_demonstration(f"p{i}", label=int(i % 5 == 0), rng=rng) for i in range(50)]
    # ~20% positive rate, roughly mimicking real class imbalance

    for k in [1, 5, 10, 20]:
        selected = select_random_demonstrations(pool, k=k, rng=rng, stratify_on_label=True)
        assert len(selected) == min(k, len(pool)), f"k={k}: wrong count"
        if k >= 2:
            assert any(d.label == 1 for d in selected), f"k={k}: no positive demonstration guaranteed"
        print(f"  random k={k}: OK ({sum(d.label for d in selected)} positive)")

    query = pool[0]
    for k in [1, 5, 10]:
        selected = select_similarity_demonstrations(
            query, pool[1:], k=k, variable_order=["hr", "crea"]
        )
        assert len(selected) == k, f"similarity k={k}: wrong count"
        print(f"  similarity k={k}: OK")

    vec = build_summary_feature_vector(pool[0], variable_order=["hr", "crea"])
    assert vec.shape == (12,), f"expected 6 stats x 2 variables = 12, got {vec.shape}"
    print(f"  summary feature vector shape: OK {vec.shape}")


def test_forecast_to_label():
    rng = np.random.default_rng(2222)
    horizon = 12

    result = make_synthetic_forecast("crea", horizon, rng)
    features = forecast_to_probe_features(result)
    assert features.shape == (5,), f"expected 5 probe features, got {features.shape}"
    print(f"  probe features shape: OK {features.shape}")

    exceed_p = exceedance_probability(result, threshold=1.2, direction="above")
    assert 0.0 <= exceed_p <= 1.0
    print(f"  exceedance probability: OK ({exceed_p:.3f})")

    rule_score = aki_kdigo_creatinine_rule(result, baseline_creatinine=0.8)
    assert 0.0 <= rule_score <= 1.0
    print(f"  AKI KDIGO rule score: OK ({rule_score:.3f})")

    # Approach A: fit probe on synthetic demonstration-derived features
    n_demo = 20
    X = rng.normal(0, 1, size=(n_demo, 5))
    y = (rng.random(n_demo) > 0.7).astype(int)
    probe = fit_few_shot_probe(X, y)
    proba = probe.predict_proba(features.reshape(1, -1))
    assert proba.shape == (1,)
    print(f"  few-shot probe predict_proba: OK ({proba[0]:.3f})")

    X_val = rng.normal(0, 1, size=(30, 5))
    y_val = (rng.random(30) > 0.8).astype(int)
    probe_zs = fit_zero_shot_probe(X_val, y_val)
    proba_zs = probe_zs.predict_proba(features.reshape(1, -1))
    print(f"  zero-shot probe predict_proba: OK ({proba_zs[0]:.3f})")


def test_task_decision_rules():
    rng = np.random.default_rng(2222)
    horizon = 12

    # sepsis: partial-SOFA proxy rule, worsening trajectory should score high
    def make_forecast_with_level(var: str, level: float) -> ForecastResult:
        point = np.full(horizon, level, dtype=np.float32)
        quantiles = np.stack([point] * 5, axis=1).astype(np.float32)
        return ForecastResult(var, point, quantiles, context_len_used=48)

    # "sick" trajectory: elevated creatinine, elevated bilirubin, low platelets, low MAP
    sick_forecasts = {
        "creatinine": make_forecast_with_level("creatinine", 4.0),  # -> 3 pts
        "bilirubin": make_forecast_with_level("bilirubin", 7.0),    # -> 3 pts
        "platelets": make_forecast_with_level("platelets", 40.0),   # -> 3 pts
        "map": make_forecast_with_level("map", 55.0),               # -> 1 pt
    }
    score_sick = sepsis_partial_sofa_rule(sick_forecasts, baseline_partial_sofa=0.0)
    assert score_sick == 1.0, f"expected clamped max score 1.0, got {score_sick}"
    print(f"  sepsis partial-SOFA rule (sick trajectory): OK ({score_sick:.3f})")

    # "healthy" trajectory: nothing crosses any SOFA sub-score threshold
    healthy_forecasts = {
        "creatinine": make_forecast_with_level("creatinine", 0.8),
        "bilirubin": make_forecast_with_level("bilirubin", 0.5),
        "platelets": make_forecast_with_level("platelets", 250.0),
        "map": make_forecast_with_level("map", 85.0),
    }
    score_healthy = sepsis_partial_sofa_rule(healthy_forecasts, baseline_partial_sofa=0.0)
    assert score_healthy == 0.0, f"expected 0.0 for healthy trajectory, got {score_healthy}"
    print(f"  sepsis partial-SOFA rule (healthy trajectory): OK ({score_healthy:.3f})")

    # missing components should score as 0 contribution, not error (matches
    # the "unmeasured -> healthy" convention documented in the docstring)
    partial_forecasts = {"creatinine": make_forecast_with_level("creatinine", 4.0)}
    score_partial = sepsis_partial_sofa_rule(partial_forecasts, baseline_partial_sofa=0.0)
    assert 0.0 < score_partial <= 1.0
    print(f"  sepsis partial-SOFA rule (missing components): OK ({score_partial:.3f})")

    # task coverage: aki and sepsis have rules, mortality deliberately does not
    assert "aki" in TASK_DECISION_RULES
    assert "sepsis" in TASK_DECISION_RULES
    assert "mortality" not in TASK_DECISION_RULES, (
        "mortality should NOT have an Approach-B rule — see coverage decision "
        "documented in forecast_to_label.py (Approach A only for mortality)"
    )
    print("  task decision rule coverage (aki+sepsis present, mortality absent by design): OK")


if __name__ == "__main__":
    print("Testing demonstration selection...")
    test_demonstration_selection()
    print("\nTesting forecast-to-label mapping...")
    test_forecast_to_label()
    print("\nTesting task decision rule coverage...")
    test_task_decision_rules()
    print("\nAll smoke tests passed.")
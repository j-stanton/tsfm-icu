"""Print the true labels and probe scores for the AKI dry run, to check the low AUROC."""

from pathlib import Path

from timesfm_loader import TimesFMForecaster, TimesFMRunConfig
from yaib_data import load_yaib_cohort, TASK_HORIZON_HOURS
from zero_shot_pipeline import run_zero_shot_evaluation

COHORT_DIR = Path(
    "/home-mscluster/jstanton/tsfm-icu-early-warning/external/YAIB-cohorts/data/aki/mimic"
)
SPLIT_FILE = Path(
    "/home-mscluster/jstanton/tsfm-icu-early-warning/splits/shared_split.csv"
)

print("Loading AKI cohort...")
cohort = load_yaib_cohort(
    COHORT_DIR, task_name="aki", split_assignment=SPLIT_FILE
)

print("Loading TimesFM (this is the real checkpoint, same as the dry run)...")
forecaster = TimesFMForecaster(
    TimesFMRunConfig(max_horizon=TASK_HORIZON_HOURS["aki"])
)
forecaster.load()

print("Running the SAME evaluation the dry run did (probe, max_patients=20, "
      "max_calibration_patients=50), single seed for a clean, reproducible look...")
import numpy as np
np.random.seed(0)

result = run_zero_shot_evaluation(
    cohort, forecaster, task_name="aki", approach="probe",
    max_patients=20, max_calibration_patients=50,
)

print(f"\nn_patients={result.n_patients} AUROC={result.auroc:.3f} AUPRC={result.auprc:.3f}")
print()
print(f"{'patient_id':20s} {'true_label':10s} {'probe_score':12s}")
for p in result.predictions:
    print(f"{p.patient_id:20s} {p.true_label:<10d} {p.probe_score:.4f}")

print()
labels = [p.true_label for p in result.predictions]
scores = [p.probe_score for p in result.predictions]
print(f"label distribution: {sum(labels)} positive / {len(labels)} total")
print(f"score range: min={min(scores):.4f} max={max(scores):.4f} mean={sum(scores)/len(scores):.4f}")

print(f"\n1 - AUROC = {1 - result.auroc:.3f} (if this is much higher than "
      f"AUROC itself, the SCORES are ranking patients meaningfully, just "
      f"with the wrong sign somewhere -- worth searching for a flipped "
      f"comparison or inverted label encoding. If both AUROC and 1-AUROC "
      f"are unremarkable/close to a coin flip in different directions "
      f"given n=20's sampling noise, this is more likely a small-sample "
      f"result, not a bug.)")

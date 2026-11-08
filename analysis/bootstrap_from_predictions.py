#!/usr/bin/env python3
"""
Exact patient-level bootstrap intervals from the per-patient prediction CSVs
written by `zero_shot/run_zero_shot.py --predictions-dir ...`.

Resamples PATIENTS (all of a patient's scored rows move together), unlike the
instance-level bootstrap inside run_zero_shot.py.

Usage:
  # one method
  python bootstrap_from_predictions.py A.csv --score probe_score
  # paired comparison A vs B (rows must be the same instances in the same order)
  python bootstrap_from_predictions.py A.csv --score rule_score \
         --compare B.csv --compare-score rule_score
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def metrics(y, s):
    return roc_auc_score(y, s), average_precision_score(y, s)


def cluster_indices(pids):
    codes, uniq = pd.factorize(pids)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    return [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--score", default="probe_score")
    ap.add_argument("--compare", default=None)
    ap.add_argument("--compare-score", default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    A = pd.read_csv(a.csv)
    y = A["true_label"].to_numpy()
    sa = A[a.score].to_numpy(float)
    sb = None
    if a.compare:
        B = pd.read_csv(a.compare)
        if len(B) != len(A) or not (A["patient_id"].astype(str).values == B["patient_id"].astype(str).values).all() \
                or not (A["true_label"].values == B["true_label"].values).all():
            raise SystemExit("--compare file is not the same instances in the same order; cannot pair.")
        sb = B[a.compare_score or a.score].to_numpy(float)

    clusters = cluster_indices(A["patient_id"].astype(str).to_numpy())
    rng = np.random.default_rng(a.seed)
    boots = []
    for _ in range(a.n_boot):
        pick = rng.integers(0, len(clusters), len(clusters))
        idx = np.concatenate([clusters[i] for i in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        row = list(metrics(y[idx], sa[idx]))
        if sb is not None:
            row += list(metrics(y[idx], sb[idx]))
        boots.append(row)
    boots = np.array(boots)
    ci = lambda v: np.percentile(v, [2.5, 97.5])

    au, ap_ = metrics(y, sa)
    print(f"n_rows={len(y)} n_patients={len(clusters)} prevalence={y.mean():.4f} boots_used={len(boots)}")
    print(f"A  AUROC {au:.3f} [{ci(boots[:,0])[0]:.3f}, {ci(boots[:,0])[1]:.3f}]   "
          f"AUPRC {ap_:.3f} [{ci(boots[:,1])[0]:.3f}, {ci(boots[:,1])[1]:.3f}]")
    if sb is not None:
        bu, bp = metrics(y, sb)
        print(f"B  AUROC {bu:.3f} [{ci(boots[:,2])[0]:.3f}, {ci(boots[:,2])[1]:.3f}]   "
              f"AUPRC {bp:.3f} [{ci(boots[:,3])[0]:.3f}, {ci(boots[:,3])[1]:.3f}]")
        d = boots[:, 0] - boots[:, 2]
        print(f"A-B AUROC diff {au-bu:+.3f} [{ci(d)[0]:+.3f}, {ci(d)[1]:+.3f}]  "
              f"(paired patient-level bootstrap; excludes 0 => difference unlikely to be noise)")


if __name__ == "__main__":
    main()

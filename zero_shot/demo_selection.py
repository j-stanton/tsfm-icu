"""Pick demonstrations for few-shot prompts (random or similarity-based)."""

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np


class SelectionStrategy(str, Enum):
    RANDOM = "random"
    SIMILARITY = "similarity"


@dataclass
class Demonstration:
    """One labelled training-set episode used as an in-context example."""
    patient_id: str
    label: int
    dynamic_series: dict[str, np.ndarray]  # variable_name -> hourly series
    static_features: dict[str, float]


def select_random_demonstrations(
    train_pool: Sequence[Demonstration],
    k: int,
    rng: np.random.Generator,
    stratify_on_label: bool = True,
) -> list[Demonstration]:
    """Random sampling without replacement from the training-cohort demonstration pool."""
    if k <= 0:
        return []

    positives = [d for d in train_pool if d.label == 1]
    negatives = [d for d in train_pool if d.label == 0]

    if not stratify_on_label or k < 2 or not positives:
        idx = rng.choice(len(train_pool), size=min(k, len(train_pool)), replace=False)
        return [train_pool[i] for i in idx]

    # Guarantee >= 1 positive, fill the rest randomly from the full pool
    # (excluding the guaranteed positive to avoid duplicate selection).
    guaranteed_positive = positives[rng.integers(0, len(positives))]
    remaining_pool = [d for d in train_pool if d.patient_id != guaranteed_positive.patient_id]
    n_remaining = min(k - 1, len(remaining_pool))
    idx = rng.choice(len(remaining_pool), size=n_remaining, replace=False)
    rest = [remaining_pool[i] for i in idx]

    selected = [guaranteed_positive] + rest
    rng.shuffle(selected)
    return selected


def build_summary_feature_vector(
    demo: Demonstration,
    variable_order: Sequence[str],
) -> np.ndarray:
    feature_blocks = []
    for var in variable_order:
        series = demo.dynamic_series.get(var, np.array([]))
        series = series[~np.isnan(series)]
        if series.size == 0:
            feature_blocks.append(np.zeros(6, dtype=np.float32))
            continue
        t = np.arange(series.size, dtype=np.float32)
        # np.polyfit degree-1 slope; guard against the degenerate single-point case
        slope = np.polyfit(t, series, 1)[0] if series.size >= 2 else 0.0
        feature_blocks.append(np.array([
            series.mean(),
            series.min(),
            series.max(),
            series[-1],
            slope,
            float(series.size),
        ], dtype=np.float32))
    return np.concatenate(feature_blocks)


def select_similarity_demonstrations(
    query: Demonstration,
    train_pool: Sequence[Demonstration],
    k: int,
    variable_order: Sequence[str],
    embedding: str = "summary_features",
) -> list[Demonstration]:
    if embedding != "summary_features":
        raise NotImplementedError(
            "Only 'summary_features' embedding is implemented so far. "
            "TimesFM-encoder-based retrieval is a planned ablation (Section 3.3) "
            "but needs a model forward pass, not just array math — deferred "
            "until the base pipeline is validated end-to-end."
        )
    if k <= 0:
        return []

    query_vec = build_summary_feature_vector(query, variable_order)
    query_norm = np.linalg.norm(query_vec) + 1e-8

    sims = []
    for demo in train_pool:
        vec = build_summary_feature_vector(demo, variable_order)
        denom = (np.linalg.norm(vec) + 1e-8) * query_norm
        cos_sim = float(np.dot(query_vec, vec) / denom) if denom > 0 else 0.0
        sims.append(cos_sim)

    top_k_idx = np.argsort(sims)[::-1][:k]
    return [train_pool[i] for i in top_k_idx]

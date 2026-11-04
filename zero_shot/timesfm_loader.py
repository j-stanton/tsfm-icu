"""
Thin wrapper around the TimesFM 2.5 checkpoint.

Per the proposal (Section 3.3): TimesFM is never fine-tuned or weight-updated
on MIMIC-III data at any stage. This module only ever loads the pretrained
checkpoint and exposes a forecasting interface; all adaptation happens via
in-context demonstrations (see timesfm_icl.py) or a downstream probe/rule
(see forecast_to_label.py), never via gradient updates to TimesFM itself.

The checkpoint identifier is recorded here as a named constant specifically
so it ends up logged in every experiment's config (per the methodology's
reproducibility requirement, Section 3.4) rather than hardcoded ad hoc at
each call site.
"""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# --- Checkpoint identity -----------------------------------------------
# Recorded explicitly so it is captured verbatim in run configs/logs.
# Swap this single constant if a different checkpoint size is used later
# (e.g. for the stretch comparison across TSFM architectures).
TIMESFM_CHECKPOINT_ID = "google/timesfm-2.5-200m-pytorch"
TIMESFM_MODEL_CLASS_NAME = "TimesFM_2p5_200M_torch"  # matches installed timesfm pkg


@dataclass
class TimesFMRunConfig:
    """
    Mirrors timesfm.ForecastConfig fields we actually use, kept here as a
    plain dataclass so it can be serialized into experiment logs without
    depending on the timesfm package at config-definition time (useful for
    e.g. analysis code that reads back past run configs without needing
    TimesFM installed).
    """
    max_context: int = 512          # matches the ~512-timepoint limit noted
                                     # for this checkpoint family; see notes
                                     # on context-length handling in
                                     # forecast_to_label.py / demo_selection.py
    max_horizon: int = 24           # placeholder default, NOT actually used
                                     # by any real call site — every caller
                                     # passes max_horizon=horizon explicitly
                                     # from yaib_data.TASK_HORIZON_HOURS.
                                     # CORRECTED 2026-09-14: this used to be
                                     # commented "24h, correct for
                                     # mortality" — that was the exact bug
                                     # that caused mortality's real
                                     # TASK_HORIZON_HOURS value to be
                                     # wrongly set to 24 too (a horizon
                                     # nearly as long as the entire
                                     # ~25-point series left almost no
                                     # context). See TASK_HORIZON_HOURS in
                                     # yaib_data.py for the corrected value
                                     # and full explanation.
    normalize_inputs: bool = True
    use_continuous_quantile_head: bool = True
    force_flip_invariance: bool = True
    infer_is_positive: bool = False  # clinical values (e.g. lab deltas) can be
                                      # negative; do NOT assume positivity by default
    fix_quantile_crossing: bool = True
    torch_compile: bool = False      # off by default for local/demo dev;
                                      # flip on for cluster runs
    per_core_batch_size: int = 8     # CRITICAL for small-VRAM GPUs (mscluster's
                                      # stampede partition: GTX1060, 6GB). Left
                                      # unset, TimesFM's own internal default
                                      # is sized for much larger cards -- the
                                      # upstream docs' own guidance table
                                      # ("GPU 8GB VRAM: per_core_batch_size=64")
                                      # implies an unset/high default is
                                      # exactly what caused a real
                                      # CUDA OOM (5.91GB allocated before a
                                      # single forecast call even completed)
                                      # on this hardware. 8 is a conservative
                                      # starting point for 6GB; raise it only
                                      # after confirming headroom via
                                      # nvidia-smi during a real run, not
                                      # before.

    def as_dict(self) -> dict:
        return {
            "max_context": self.max_context,
            "max_horizon": self.max_horizon,
            "normalize_inputs": self.normalize_inputs,
            "use_continuous_quantile_head": self.use_continuous_quantile_head,
            "force_flip_invariance": self.force_flip_invariance,
            "infer_is_positive": self.infer_is_positive,
            "fix_quantile_crossing": self.fix_quantile_crossing,
            "per_core_batch_size": self.per_core_batch_size,
        }


@dataclass
class ForecastResult:
    """
    Container for a single univariate forecast call. Keeping this as an
    explicit type (rather than passing raw arrays around) so downstream
    code — the forecast-probe classifier and the clinical decision rule
    (Approach A / B, Section 3.3) — has a single stable interface regardless
    of which adaptation regime (zero-shot / k-shot ICL) produced the forecast.
    """
    variable_name: str
    point_forecast: np.ndarray        # shape (horizon,)
    quantile_forecast: np.ndarray     # shape (horizon, 10): CORRECTED — the
                                       # installed timesfm package's own
                                       # documented convention is that these
                                       # 10 columns are the 10th-90th
                                       # percentile bands (NOT quantile-0 ==
                                       # mean, as an earlier version of this
                                       # docstring assumed; verified against
                                       # the upstream README/HF model card).
                                       # point_forecast is the separate median
                                       # forecast returned alongside this.
    context_len_used: int
    demonstrations_used: int = 0      # 0 for zero-shot; k for few-shot ICL
    quantile_levels: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
                                       # placeholder ascending labels for the
                                       # 10 columns above; TODO confirm exact
                                       # levels against the installed
                                       # package's ForecastConfig docs before
                                       # relying on any single specific
                                       # column (e.g. "the 90th percentile
                                       # column") rather than aggregate
                                       # statistics (min/max/mean over all
                                       # columns), which are robust to this
                                       # ambiguity and are what
                                       # forecast_to_label.py actually uses.


class TimesFMForecaster:
    """
    Loads the TimesFM checkpoint once and exposes forecasting for
    univariate clinical variable series.

    TimesFM is a univariate forecaster (Section 3.3.1): each input clinical
    variable is processed through its own forward pass. This class reflects
    that — `forecast_variable` takes one series at a time. Aggregation across
    variables happens one layer up, not here.
    """

    def __init__(self, run_config: TimesFMRunConfig | None = None):
        self.run_config = run_config or TimesFMRunConfig()
        self._model = None  # lazy-loaded; see .load()

    def load(self) -> None:
        """
        Loads the pretrained checkpoint and compiles it with the current
        run_config. Deliberately separate from __init__ so that scaffolding/
        unit tests can construct a TimesFMForecaster without triggering a
        ~925MB download, and so the checkpoint ID is logged before the
        (possibly slow) load actually happens.
        """
        import timesfm  # local import: keep this module importable in
                         # environments where the timesfm package isn't
                         # installed yet (e.g. during YAIB-only demo work)

        model_cls = getattr(timesfm, TIMESFM_MODEL_CLASS_NAME)
        self._model = model_cls.from_pretrained(
            TIMESFM_CHECKPOINT_ID,
            torch_compile=self.run_config.torch_compile,
        )
        self._model.compile(
            timesfm.ForecastConfig(**self.run_config.as_dict())
        )

    def _ensure_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError(
                "TimesFMForecaster.load() must be called before forecasting. "
                "This is intentionally not automatic so the checkpoint load "
                "is a visible, loggable step in experiment scripts."
            )

    def forecast_variable(
        self,
        variable_name: str,
        context: Sequence[float],
        horizon: int | None = None,
    ) -> ForecastResult:
        """
        Zero-shot forecast for a single clinical variable's series.

        `context` should already be on the hourly YAIB-preprocessed grid
        (Section 3.2) with missingness handled upstream — this function does
        no resampling or imputation itself, to keep a single preprocessing
        source of truth (YAIB's cache) rather than duplicating logic here.

        This single-series call exists for scaffolding/debugging convenience
        (e.g. the smoke test) but should NOT be used in a loop over a real
        cohort — `model.forecast()` accepts a batch of series in one call
        and a Python-level loop calling this repeatedly will be far slower
        than batching. Use `forecast_batch` for anything beyond a handful
        of series.
        """
        self._ensure_loaded()
        horizon = horizon or self.run_config.max_horizon

        context_arr = np.asarray(context, dtype=np.float32)
        point_forecast, quantile_forecast = self._model.forecast(
            horizon=horizon,
            inputs=[context_arr],
        )

        return ForecastResult(
            variable_name=variable_name,
            point_forecast=point_forecast[0],
            quantile_forecast=quantile_forecast[0],
            context_len_used=len(context_arr),
            demonstrations_used=0,
        )

    def forecast_batch(
        self,
        variable_names: Sequence[str],
        contexts: Sequence[Sequence[float]],
        horizon: int | None = None,
    ) -> list[ForecastResult]:
        """
        Batched forecast over multiple series in a single model call.

        This is the call the real pipeline (zero_shot_pipeline.py) should use:
        one `model.forecast()` invocation per batch rather than one per
        series, since TimesFM's own forecast() signature already accepts
        a list of input series and returns stacked point/quantile arrays
        (verified against the upstream package's documented usage:
        `inputs=[series_a, series_b]` -> `point_forecast.shape == (2, horizon)`).

        `variable_names` and `contexts` must be the same length and
        positionally aligned — variable_names[i] labels contexts[i].
        Series in `contexts` may have different lengths (TimesFM handles
        variable-length input contexts per Das et al. [2023]); this method
        does not pad or truncate — that is caller responsibility if a
        model-specific max_context limit needs enforcing (see
        max_context on TimesFMRunConfig, and the context-length-overflow
        handling documented in forecast_to_label.py / demo_selection.py).
        """
        if len(variable_names) != len(contexts):
            raise ValueError(
                f"variable_names ({len(variable_names)}) and contexts "
                f"({len(contexts)}) must be the same length."
            )
        self._ensure_loaded()
        horizon = horizon or self.run_config.max_horizon

        context_arrs = [np.asarray(c, dtype=np.float32) for c in contexts]
        point_forecasts, quantile_forecasts = self._model.forecast(
            horizon=horizon,
            inputs=context_arrs,
        )

        return [
            ForecastResult(
                variable_name=name,
                point_forecast=point_forecasts[i],
                quantile_forecast=quantile_forecasts[i],
                context_len_used=len(context_arrs[i]),
                demonstrations_used=0,
            )
            for i, name in enumerate(variable_names)
        ]
"""Wrapper around the TimesFM 2.5 checkpoint."""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

TIMESFM_CHECKPOINT_ID = "google/timesfm-2.5-200m-pytorch"
TIMESFM_MODEL_CLASS_NAME = "TimesFM_2p5_200M_torch"


@dataclass
class TimesFMRunConfig:
    max_context: int = 512
    max_horizon: int = 24
    normalize_inputs: bool = True
    use_continuous_quantile_head: bool = True
    force_flip_invariance: bool = True
    infer_is_positive: bool = False  # clinical values can be negative
    fix_quantile_crossing: bool = True
    torch_compile: bool = False
    per_core_batch_size: int = 8     # small, to fit the 6 GB GPUs

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
    """Container for a single univariate forecast call."""
    variable_name: str
    point_forecast: np.ndarray        # shape (horizon,)
    quantile_forecast: np.ndarray
    context_len_used: int
    demonstrations_used: int = 0      # 0 for zero-shot; k for few-shot ICL
    quantile_levels: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


class TimesFMForecaster:
    """Loads the TimesFM checkpoint once and exposes forecasting for univariate clinical variable series."""

    def __init__(self, run_config: TimesFMRunConfig | None = None):
        self.run_config = run_config or TimesFMRunConfig()
        self._model = None  # lazy-loaded; see .load()

    def load(self) -> None:
        """Loads the pretrained checkpoint and compiles it with the current run_config."""
        import timesfm  # imported here so the module loads without timesfm installed

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
        """Zero-shot forecast for a single clinical variable's series."""
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
        """Batched forecast over multiple series in a single model call."""
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

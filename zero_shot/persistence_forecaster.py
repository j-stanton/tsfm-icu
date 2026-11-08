"""
Persistence "forecaster": repeats the last observed value of each context for
the whole horizon. Same interface as TimesFMForecaster (load / forecast_batch /
forecast_variable), so it runs through exactly the same probe and clinical-rule
code. Used as an ablation: if the AKI rule scores about the same with persistence
as with TimesFM, the rule's signal comes from the recent history, not from
the model's forecast.

Quantile columns are all set to the last value, so the probe's variance feature
is 0 and its slope feature is 0 by construction.
"""

from typing import Sequence

import numpy as np

from timesfm_loader import ForecastResult, TimesFMRunConfig

N_QUANTILE_COLUMNS = 10  # matches TimesFM's quantile_forecast shape (horizon, 10)


class PersistenceForecaster:
    def __init__(self, run_config: TimesFMRunConfig | None = None):
        self.run_config = run_config or TimesFMRunConfig()

    def load(self) -> None:  # nothing to load
        return None

    def forecast_batch(
        self,
        variable_names: Sequence[str],
        contexts: Sequence[Sequence[float]],
        horizon: int | None = None,
    ) -> list[ForecastResult]:
        if len(variable_names) != len(contexts):
            raise ValueError("variable_names and contexts must be the same length.")
        horizon = horizon or self.run_config.max_horizon
        out = []
        for name, ctx in zip(variable_names, contexts):
            arr = np.asarray(ctx, dtype=np.float32)
            last = float(arr[-1])
            point = np.full(horizon, last, dtype=np.float32)
            quant = np.tile(point[:, None], (1, N_QUANTILE_COLUMNS))
            out.append(ForecastResult(
                variable_name=name,
                point_forecast=point,
                quantile_forecast=quant,
                context_len_used=len(arr),
                demonstrations_used=0,
            ))
        return out

    def forecast_variable(self, variable_name: str, context, horizon: int | None = None):
        return self.forecast_batch([variable_name], [context], horizon=horizon)[0]

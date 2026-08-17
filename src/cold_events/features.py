from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FEATURE_COLUMNS


def build_supervised_frame(
    frame: pd.DataFrame, lookback: int = 5, horizon: int = 1
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build issue-time features at t to predict the event label at t + horizon.

    All predictors use observations available on or before the issue date. Returned
    dates are target dates so temporal splits and yearly backtests represent the
    period being forecast, not the date on which the prediction was issued.
    """
    if lookback < 1:
        raise ValueError("lookback must be at least 1")
    if horizon not in {1, 3, 7}:
        raise ValueError("horizon must be one of 1, 3, or 7 days")

    ordered = frame.sort_values("DATE").reset_index(drop=True).copy()
    result = pd.DataFrame(index=ordered.index)

    # lag_0 is the issue-date observation; every other lag is further in the past.
    for lag in range(lookback):
        for column in FEATURE_COLUMNS:
            result[f"{column.lower()}_lag_{lag}"] = ordered[column].shift(lag)

    trailing_temp = ordered["TEMP"].rolling(lookback, min_periods=lookback)
    result["temp_mean_5d"] = trailing_temp.mean()
    result["temp_min_5d"] = trailing_temp.min()
    result["temp_max_5d"] = trailing_temp.max()
    result["temp_trend_5d"] = ordered["TEMP"] - ordered["TEMP"].shift(lookback - 1)

    # Calendar seasonality is known at issuance and deliberately uses the issue date.
    issue_day = ordered["DATE"].dt.dayofyear
    result["doy_sin"] = np.sin(2 * np.pi * issue_day / 365.25)
    result["doy_cos"] = np.cos(2 * np.pi * issue_day / 365.25)

    target = ordered["extreme_cold"].shift(-horizon)
    target_date = ordered["DATE"].shift(-horizon)
    history_start = ordered["DATE"].shift(lookback - 1)
    has_contiguous_history = history_start.eq(
        ordered["DATE"] - pd.to_timedelta(lookback - 1, unit="D")
    )
    has_exact_target = target_date.eq(
        ordered["DATE"] + pd.to_timedelta(horizon, unit="D")
    )
    valid = has_contiguous_history & has_exact_target & target.notna()

    return (
        result.loc[valid].reset_index(drop=True),
        target.loc[valid].astype(int).reset_index(drop=True),
        target_date.loc[valid].reset_index(drop=True),
    )

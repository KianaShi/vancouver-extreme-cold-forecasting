import numpy as np
import pandas as pd

from cold_events.backtesting import walk_forward_fold_masks
from cold_events.data import load_gsod
from cold_events.features import build_supervised_frame
from cold_events.modeling import choose_threshold, temporal_partitions, temporal_split


def test_loader_filters_station_sorts_and_cleans_sentinels(tmp_path):
    source = pd.DataFrame({
        "STATION": [1, 2, 1],
        "DATE": ["2020-01-02", "2020-01-01", "2020-01-01"],
        "MIN": [9999.9, 10.0, 20.0], "MAX": [30.0, 20.0, 25.0],
        "TEMP": [25.0, 15.0, 22.0], "GUST": [10.0, 11.0, 12.0], "SNDP": [0.0, 0.0, 0.0],
    })
    path = tmp_path / "weather.xlsx"
    with pd.ExcelWriter(path) as writer:
        source.to_excel(writer, sheet_name="data", index=False)
    result = load_gsod(path, station=1)
    assert result["DATE"].is_monotonic_increasing
    assert len(result) == 2
    assert np.isnan(result.loc[1, "MIN"])


def test_windows_reject_calendar_gaps():
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05", "2020-01-06", "2020-01-08"])
    frame = pd.DataFrame({"DATE": dates, "STATION": 1})
    for column in ["MIN", "MAX", "TEMP", "GUST", "SNDP"]:
        frame[column] = 10.0
    frame["extreme_cold"] = True
    X, y, out_dates = build_supervised_frame(frame, lookback=5)
    assert len(X) == 1
    assert out_dates.iloc[0] == pd.Timestamp("2020-01-06")
    assert y.iloc[0] == 1


def test_temporal_split_has_no_overlap():
    dates = pd.Series(pd.date_range("2018-12-29", periods=6))
    X = pd.DataFrame({"x": range(6)})
    y = pd.Series([0, 0, 1, 0, 1, 0])
    X_train, X_test, _, _, test_dates = temporal_split(X, y, dates, "2019-01-01")
    assert len(X_train) == 3 and len(X_test) == 3
    assert dates.loc[X_train.index].max() < test_dates.min()


def test_temporal_partitions_are_ordered_and_disjoint():
    dates = pd.Series(pd.date_range("2000-01-01", "2020-12-31"))
    masks = temporal_partitions(dates, "2019-01-01")
    assert sum(mask.astype(int) for mask in masks.values()).eq(1).all()
    assert dates.loc[masks["train"]].max() < dates.loc[masks["validation"]].min()
    assert dates.loc[masks["validation"]].max() < dates.loc[masks["test"]].min()


def test_threshold_is_selected_from_supplied_validation_probabilities():
    labels = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.5, 0.9])
    assert choose_threshold(labels, probabilities) == 0.5


def test_one_three_and_seven_day_targets_use_issue_time_features_only():
    dates = pd.date_range("2020-01-01", periods=20)
    frame = pd.DataFrame({"DATE": dates, "STATION": 1})
    for column in ["MIN", "MAX", "TEMP", "GUST", "SNDP"]:
        frame[column] = np.arange(20, dtype=float)
    frame["extreme_cold"] = (np.arange(20) % 3 == 0)

    for horizon in (1, 3, 7):
        X, y, target_dates = build_supervised_frame(frame, lookback=5, horizon=horizon)
        issue_index = 4
        target_index = issue_index + horizon
        assert X.loc[0, "min_lag_0"] == frame.loc[issue_index, "MIN"]
        assert X.loc[0, "min_lag_4"] == frame.loc[0, "MIN"]
        assert frame.loc[target_index, "MIN"] not in X.loc[0, X.columns.str.startswith("min_")].values
        assert y.iloc[0] == int(frame.loc[target_index, "extreme_cold"])
        assert target_dates.iloc[0] == frame.loc[target_index, "DATE"]


def test_walk_forward_training_and_threshold_periods_precede_evaluation_year():
    dates = pd.Series(pd.date_range("2000-01-01", "2020-12-31"))
    masks = walk_forward_fold_masks(dates, evaluation_year=2019, validation_years=3)
    evaluation_start = pd.Timestamp("2019-01-01")
    assert dates.loc[masks["train"]].max() < dates.loc[masks["validation"]].min()
    assert dates.loc[masks["validation"]].max() < evaluation_start
    assert dates.loc[masks["evaluation"]].min() == evaluation_start
    assert not (masks["validation"] & masks["evaluation"]).any()

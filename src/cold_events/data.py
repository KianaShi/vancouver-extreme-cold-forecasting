from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["MIN", "MAX", "TEMP", "GUST", "SNDP"]
SENTINELS = {"MIN": 9999.9, "MAX": 9999.9, "TEMP": 9999.9, "GUST": 999.9, "SNDP": 999.9}
DEFAULT_STATION = 71892099999


def _read_source(path: Path) -> pd.DataFrame:
    """Read the original workbook, one NOAA GSOD CSV, or a directory of yearly CSVs."""
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
        if not files:
            raise ValueError(f"No .csv files found in {path}")
        return pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name="data")


def load_gsod(
    path: str | Path, station: int = DEFAULT_STATION, end_date: str | None = None
) -> pd.DataFrame:
    """Load, validate, clean, and chronologically order one GSOD station."""
    frame = _read_source(Path(path))
    required = {"STATION", "DATE", *FEATURE_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = frame.loc[frame["STATION"].eq(station), ["STATION", "DATE", *FEATURE_COLUMNS]].copy()
    if frame.empty:
        raise ValueError(f"Station {station} is absent from the workbook")
    frame["DATE"] = pd.to_datetime(frame["DATE"], errors="raise")
    if end_date is not None:
        frame = frame.loc[frame["DATE"].le(pd.Timestamp(end_date))]
    frame = frame.sort_values("DATE").drop_duplicates(["STATION", "DATE"], keep="last")

    for column, sentinel in SENTINELS.items():
        frame.loc[frame[column].ge(sentinel), column] = np.nan

    # The target can only be defined when daily minimum temperature is observed.
    frame["extreme_cold"] = frame["MIN"].lt(23.0).where(frame["MIN"].notna())
    return frame.reset_index(drop=True)


def data_quality_report(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": len(frame),
        "start": frame["DATE"].min().date().isoformat(),
        "end": frame["DATE"].max().date().isoformat(),
        "duplicate_station_dates": int(frame.duplicated(["STATION", "DATE"]).sum()),
        "chronological": bool(frame["DATE"].is_monotonic_increasing),
        "missing_by_feature": {k: int(v) for k, v in frame[FEATURE_COLUMNS].isna().sum().items()},
        "positive_labels": int(frame["extreme_cold"].fillna(False).sum()),
    }

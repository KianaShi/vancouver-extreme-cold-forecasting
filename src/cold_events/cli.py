from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import DEFAULT_STATION, data_quality_report, load_gsod
from .backtesting import run_forecasting_analysis
from .features import build_supervised_frame
from .modeling import train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train leakage-safe cold-event classifiers")
    parser.add_argument("--data", type=Path, required=True, help="Path to the source .xlsx workbook")
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--station", type=int, default=DEFAULT_STATION)
    parser.add_argument("--test-start", default="2019-01-01")
    parser.add_argument("--backtest-start-year", type=int, default=2011)
    parser.add_argument("--skip-backtest", action="store_true")
    args = parser.parse_args()

    frame = load_gsod(args.data, args.station)
    quality = data_quality_report(frame)
    X, y, dates = build_supervised_frame(frame, horizon=1)
    metrics = train_and_evaluate(X, y, dates, args.output, args.test_start)
    forecasting = run_forecasting_analysis(
        frame,
        args.output,
        test_start=args.test_start,
        backtest_start_year=args.backtest_start_year,
        run_backtest=not args.skip_backtest,
    )
    print(
        json.dumps(
            {"data_quality": quality, "modeling": metrics, "forecasting": forecasting},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

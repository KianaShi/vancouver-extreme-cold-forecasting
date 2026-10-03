from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import DEFAULT_STATION, data_quality_report, load_gsod
from .backtesting import run_forecasting_analysis
from .features import build_supervised_frame
from .modeling import train_and_evaluate
from .original_setup import run_original_setup


def main() -> None:
    parser = argparse.ArgumentParser(description="Train leakage-safe cold-event classifiers")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Source .xlsx workbook, a GSOD .csv file, or a directory of yearly GSOD .csv files",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--station", type=int, default=DEFAULT_STATION)
    parser.add_argument("--end-date", default=None, help="Ignore observations after this date")
    parser.add_argument("--test-start", default="2019-01-01")
    parser.add_argument("--backtest-start-year", type=int, default=2011)
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument(
        "--original-setup",
        action="store_true",
        help="Only run the corrected capstone design (LR, RF, RBF-SVM on 25 lag features)",
    )
    args = parser.parse_args()

    frame = load_gsod(args.data, args.station, args.end_date)
    quality = data_quality_report(frame)
    if args.original_setup:
        original = run_original_setup(frame, args.output, args.test_start)
        print(json.dumps({"data_quality": quality, "original_setup": original}, indent=2))
        return

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

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .features import build_supervised_frame
from .modeling import (
    THRESHOLD_METHOD,
    candidate_models,
    choose_threshold,
    classification_metrics,
    temporal_partitions,
)

HORIZONS = (1, 3, 7)
TREE_MODELS = ("random_forest", "lightgbm")
WALK_FORWARD_VALIDATION_YEARS = 3


def walk_forward_fold_masks(
    dates: pd.Series, evaluation_year: int, validation_years: int = WALK_FORWARD_VALIDATION_YEARS
) -> dict[str, pd.Series]:
    """Return temporal-safe train, validation, and evaluation masks for one year."""
    evaluation_start = pd.Timestamp(evaluation_year, 1, 1)
    evaluation_end = pd.Timestamp(evaluation_year + 1, 1, 1)
    validation_start = evaluation_start - pd.DateOffset(years=validation_years)
    masks = {
        "train": dates.lt(validation_start),
        "validation": dates.ge(validation_start) & dates.lt(evaluation_start),
        "evaluation": dates.ge(evaluation_start) & dates.lt(evaluation_end),
    }
    nonempty = all(mask.any() for mask in masks.values())
    if not nonempty:
        raise ValueError(f"Insufficient observations for walk-forward year {evaluation_year}")
    if dates.loc[masks["train"]].max() >= dates.loc[masks["validation"]].min():
        raise AssertionError("Walk-forward training overlaps validation")
    if dates.loc[masks["validation"]].max() >= dates.loc[masks["evaluation"]].min():
        raise AssertionError("Walk-forward validation overlaps evaluation year")
    return masks


def _tree_candidates(columns: list[str]):
    candidates = candidate_models(columns)
    return {name: candidates[name] for name in TREE_MODELS}


def _safe_yearly_metrics(y_true, probability, threshold: float) -> dict[str, object]:
    positives = int(y_true.sum())
    samples = len(y_true)
    prediction = (probability >= threshold).astype(int)
    has_both_classes = y_true.nunique() == 2
    has_positives = positives > 0
    return {
        "samples": samples,
        "positive_events": positives,
        "positive_rate": float(y_true.mean()),
        "average_precision": (
            float(average_precision_score(y_true, probability)) if has_positives else None
        ),
        "roc_auc": float(roc_auc_score(y_true, probability)) if has_both_classes else None,
        "precision": (
            float(precision_score(y_true, prediction, zero_division=0))
            if has_positives
            else None
        ),
        "recall": (
            float(recall_score(y_true, prediction, zero_division=0))
            if has_positives
            else None
        ),
        "f1": (
            float(f1_score(y_true, prediction, zero_division=0)) if has_positives else None
        ),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=[0, 1]).tolist(),
    }


def run_horizon_benchmark(frame, output_dir: str | Path, test_start: str) -> dict:
    output = Path(output_dir)
    benchmark: dict[str, object] = {
        "target_definition": "features available through issue date t predict label at t+h",
        "threshold_policy": THRESHOLD_METHOD,
        "selection_policy": "validation average precision; test is report-only",
        "horizons": {},
    }
    for horizon in HORIZONS:
        X, y, target_dates = build_supervised_frame(frame, horizon=horizon)
        masks = temporal_partitions(target_dates, test_start)
        pretest = masks["train"] | masks["validation"]
        model_results, fitted = {}, {}
        for name, template in _tree_candidates(list(X.columns)).items():
            validation_model = clone(template)
            validation_model.fit(X.loc[masks["train"]], y.loc[masks["train"]])
            validation_probability = validation_model.predict_proba(
                X.loc[masks["validation"]]
            )[:, 1]
            threshold = choose_threshold(y.loc[masks["validation"]], validation_probability)

            final_model = clone(template)
            final_model.fit(X.loc[pretest], y.loc[pretest])
            test_probability = final_model.predict_proba(X.loc[masks["test"]])[:, 1]
            model_results[name] = {
                "threshold": threshold,
                "validation": classification_metrics(
                    y.loc[masks["validation"]], validation_probability, threshold
                ),
                "test": classification_metrics(y.loc[masks["test"]], test_probability, threshold),
            }
            fitted[name] = final_model

        selected = max(
            TREE_MODELS,
            key=lambda name: model_results[name]["validation"]["average_precision"],
        )
        joblib.dump(
            {"model": fitted[selected], "threshold": model_results[selected]["threshold"]},
            output / f"model_h{horizon}.joblib",
        )
        benchmark["horizons"][str(horizon)] = {
            "issue_to_target_days": horizon,
            "samples": len(y),
            "target_start": target_dates.min().date().isoformat(),
            "target_end": target_dates.max().date().isoformat(),
            "test_samples": int(masks["test"].sum()),
            "test_positive_rate": float(y.loc[masks["test"]].mean()),
            "selected_model": selected,
            "models": model_results,
        }

    (output / "horizon_metrics.json").write_text(
        json.dumps(benchmark, indent=2), encoding="utf-8"
    )
    _plot_horizon_performance(benchmark, output / "horizon_performance.png")
    return benchmark


def run_walk_forward(
    frame,
    output_dir: str | Path,
    start_year: int = 2011,
) -> pd.DataFrame:
    output = Path(output_dir)
    records: list[dict[str, object]] = []
    prior_thresholds: dict[tuple[int, str], float] = {}

    for horizon in HORIZONS:
        X, y, target_dates = build_supervised_frame(frame, horizon=horizon)
        final_year = int(target_dates.dt.year.max())
        for evaluation_year in range(start_year, final_year + 1):
            masks = walk_forward_fold_masks(target_dates, evaluation_year)
            visible_before_year = target_dates.lt(pd.Timestamp(evaluation_year, 1, 1))
            for name, template in _tree_candidates(list(X.columns)).items():
                threshold_model = clone(template)
                threshold_model.fit(X.loc[masks["train"]], y.loc[masks["train"]])
                validation_probability = threshold_model.predict_proba(
                    X.loc[masks["validation"]]
                )[:, 1]
                key = (horizon, name)
                if y.loc[masks["validation"]].nunique() == 2:
                    threshold = choose_threshold(
                        y.loc[masks["validation"]], validation_probability
                    )
                    threshold_source = f"trailing {WALK_FORWARD_VALIDATION_YEARS}-year validation"
                    prior_thresholds[key] = threshold
                elif key in prior_thresholds:
                    threshold = prior_thresholds[key]
                    threshold_source = "previous temporally safe threshold"
                else:
                    threshold = 0.5
                    threshold_source = "0.5 fallback; validation contained one class"

                final_model = clone(template)
                final_model.fit(X.loc[visible_before_year], y.loc[visible_before_year])
                probability = final_model.predict_proba(X.loc[masks["evaluation"]])[:, 1]
                metrics = _safe_yearly_metrics(
                    y.loc[masks["evaluation"]], probability, threshold
                )
                records.append(
                    {
                        "year": evaluation_year,
                        "horizon": horizon,
                        "model": name,
                        "threshold": threshold,
                        "threshold_source": threshold_source,
                        "training_end": target_dates.loc[visible_before_year].max().date().isoformat(),
                        "validation_start": target_dates.loc[masks["validation"]]
                        .min()
                        .date()
                        .isoformat(),
                        "validation_end": target_dates.loc[masks["validation"]]
                        .max()
                        .date()
                        .isoformat(),
                        **metrics,
                    }
                )

    result = pd.DataFrame(records)
    result.to_csv(output / "walk_forward_metrics.csv", index=False)
    _plot_walk_forward_metric(result, "average_precision", output / "walk_forward_ap.png")
    _plot_walk_forward_metric(result, "recall", output / "walk_forward_recall.png")
    _plot_positive_frequency(result, output / "positive_event_frequency.png")
    return result


def temporal_drift_summary(frame, walk_forward: pd.DataFrame, output_dir: str | Path) -> dict:
    output = Path(output_dir)
    X, y, target_dates = build_supervised_frame(frame, horizon=1)
    annual = pd.DataFrame(
        {
            "year": target_dates.dt.year,
            "positive": y,
            "min_lag_0": X["min_lag_0"],
            "max_lag_0": X["max_lag_0"],
            "temp_lag_0": X["temp_lag_0"],
        }
    )
    annual = annual.loc[annual["year"].ge(int(walk_forward["year"].min()))]
    yearly = annual.groupby("year", as_index=False).agg(
        positive_rate=("positive", "mean"),
        min_mean=("min_lag_0", "mean"),
        max_mean=("max_lag_0", "mean"),
        temp_mean=("temp_lag_0", "mean"),
    )
    yearly_counts = annual.groupby("year").size()
    complete_years = yearly_counts.loc[yearly_counts.ge(300)].index
    partial_years = yearly_counts.loc[yearly_counts.lt(300)].index.astype(int).tolist()
    complete_yearly = yearly.loc[yearly["year"].isin(complete_years)]
    years = complete_yearly["year"].to_numpy()
    prevalence_slope = float(
        np.polyfit(years, complete_yearly["positive_rate"], 1)[0]
    )

    feature_shifts = {}
    early = complete_yearly.head(5)
    recent = complete_yearly.tail(5)
    for column in ("min_mean", "max_mean", "temp_mean"):
        scale = float(complete_yearly[column].std(ddof=0))
        feature_shifts[column] = {
            "early_5y_mean": float(early[column].mean()),
            "recent_5y_mean": float(recent[column].mean()),
            "standardized_shift": (
                float((recent[column].mean() - early[column].mean()) / scale)
                if scale > 0
                else 0.0
            ),
        }

    performance_trends = {}
    for (horizon, model), group in walk_forward.groupby(["horizon", "model"]):
        valid = group.dropna(subset=["average_precision"])
        full_year_valid = valid.loc[valid["samples"].ge(300)]
        slope = (
            float(
                np.polyfit(
                    full_year_valid["year"], full_year_valid["average_precision"], 1
                )[0]
            )
            if len(full_year_valid) >= 2
            else None
        )
        performance_trends[f"h{horizon}_{model}"] = {
            "mean_ap_all_evaluations": float(valid["average_precision"].mean()),
            "mean_ap_complete_years": float(full_year_valid["average_precision"].mean()),
            "ap_slope_per_year_complete_years": slope,
            "years_with_no_positive_events": group.loc[
                group["positive_events"].eq(0), "year"
            ].astype(int).tolist(),
        }

    summary = {
        "period": {
            "start_year": int(yearly["year"].min()),
            "end_year": int(yearly["year"].max()),
        },
        "partial_years_excluded_from_trend": partial_years,
        "complete_year_minimum_samples": 300,
        "positive_rate_slope_per_year_complete_years": prevalence_slope,
        "feature_distribution_shift": feature_shifts,
        "walk_forward_performance": performance_trends,
    }
    (output / "temporal_drift.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _plot_horizon_performance(benchmark: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, (metric, label) in zip(
        axes,
        (
            ("average_precision", "Average Precision"),
            ("recall", "Recall"),
            ("brier_score", "Brier Score (lower is better)"),
        ),
        strict=True,
    ):
        for model in TREE_MODELS:
            values = [
                benchmark["horizons"][str(h)]["models"][model]["test"][metric]
                for h in HORIZONS
            ]
            axis.plot(HORIZONS, values, marker="o", label=model)
        axis.set_xticks(HORIZONS, [f"{h}-day" for h in HORIZONS])
        axis.set_title(label)
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("Forecast performance versus lead time")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_walk_forward_metric(result: pd.DataFrame, metric: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for (horizon, model), group in result.groupby(["horizon", "model"]):
        ax.plot(
            group["year"],
            group[metric],
            marker="o",
            markersize=3,
            label=f"{horizon}-day {model}",
        )
    ax.set_title(f"Walk-forward yearly {metric.replace('_', ' ').title()}")
    ax.set_xlabel("Evaluation year")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_positive_frequency(result: pd.DataFrame, path: Path) -> None:
    unique = result.drop_duplicates(["year", "horizon"])
    fig, ax = plt.subplots(figsize=(11, 5))
    for horizon, group in unique.groupby("horizon"):
        ax.plot(group["year"], group["positive_rate"], marker="o", label=f"{horizon}-day")
    ax.set_title("Yearly extreme-cold event frequency")
    ax.set_xlabel("Target year")
    ax.set_ylabel("Positive rate")
    ax.grid(alpha=0.25)
    ax.legend()
    partial = unique.loc[unique["samples"].lt(300), "year"].unique()
    for year in partial:
        ax.axvline(year, color="grey", linestyle="--", alpha=0.5)
        ax.annotate(
            f"{int(year)} partial year",
            xy=(year, unique.loc[unique["year"].eq(year), "positive_rate"].max()),
            xytext=(-85, -25),
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "->", "color": "grey"},
        )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_forecasting_analysis(
    frame,
    output_dir: str | Path,
    test_start: str = "2019-01-01",
    backtest_start_year: int = 2011,
    run_backtest: bool = True,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    benchmark = run_horizon_benchmark(frame, output, test_start)
    summary: dict[str, object] = {
        "horizon_selected_models": {
            horizon: details["selected_model"]
            for horizon, details in benchmark["horizons"].items()
        }
    }
    if run_backtest:
        walk_forward = run_walk_forward(frame, output, backtest_start_year)
        drift = temporal_drift_summary(frame, walk_forward, output)
        summary["walk_forward_rows"] = len(walk_forward)
        summary["walk_forward_years"] = [
            int(walk_forward["year"].min()),
            int(walk_forward["year"].max()),
        ]
        summary["drift"] = drift
    return summary

"""Corrected re-run of the Spring 2025 capstone design.

The capstone compared Logistic Regression, Random Forest, and an RBF-kernel SVM using the
previous five days of MIN/MAX/TEMP/GUST/SNDP (25 features) to predict whether the next day
is an extreme cold day, with a 2019-01-01 chronological split and each model's default
decision rule. This module keeps that design and changes only what the audit found broken:
a single station, GSOD sentinels treated as missing, calendar-contiguous windows, and
standardized inputs for the scale-sensitive models.

Because cold conditions persist from day to day, the models are compared with a persistence
baseline (the previous day's minimum temperature) and are also scored on winter days only and
on onset days, when the previous day was not yet extremely cold.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import build_supervised_frame

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "svm_rbf": "SVM (RBF kernel)",
}


def original_models() -> dict[str, Pipeline]:
    """The capstone's three untuned models; imputation and scaling are fitted on train only."""
    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2000)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)),
        ]),
        "svm_rbf": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", SVC(kernel="rbf", class_weight="balanced", random_state=42)),
        ]),
    }


WINTER_MONTHS = (11, 12, 1, 2, 3)
BASELINE_LABEL = "Persistence baseline"


def _scores(model: Pipeline, X: pd.DataFrame):
    # Ranking metrics only need an ordering; SVC's decision function avoids Platt scaling.
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def _ranking(y_true: pd.Series, score) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, score)),
        "average_precision": float(average_precision_score(y_true, score)),
    }


def _classification(y_true: pd.Series, prediction) -> dict:
    return {
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, prediction).tolist(),
    }


def _subsets(y_test: pd.Series, score, prediction, winter, onset) -> dict:
    """Ranking skill in winter only, and at the onset of cold events."""
    result = {"winter": _ranking(y_test[winter], score[winter])}
    result["onset"] = {
        **_ranking(y_test[onset], score[onset]),
        "positives": int(y_test[onset].sum()),
        "detected": int((prediction[onset] == 1)[y_test[onset].to_numpy() == 1].sum()),
    }
    return result


def run_original_setup(
    frame: pd.DataFrame, output_dir: str | Path, test_start: str = "2019-01-01"
) -> dict:
    output = Path(output_dir) / "original_setup"
    output.mkdir(parents=True, exist_ok=True)

    X_all, y, dates = build_supervised_frame(frame, lookback=5, horizon=1)
    lag_columns = [column for column in X_all.columns if "_lag_" in column]
    X = X_all[lag_columns]
    test = dates.ge(pd.Timestamp(test_start))
    train = ~test
    y_test = y.loc[test]
    X_test = X.loc[test]
    winter = dates.loc[test].dt.month.isin(WINTER_MONTHS).to_numpy()
    # An onset is a target day whose previous day was not observed below the threshold.
    previous_cold = X_test["min_lag_0"].lt(23.0).to_numpy()
    onset = ~previous_cold

    results, scores = {}, {}
    for name, model in original_models().items():
        model.fit(X.loc[train], y.loc[train])
        prediction = model.predict(X_test)
        score = _scores(model, X_test)
        scores[name] = score
        results[name] = {
            **_classification(y_test, prediction),
            **_ranking(y_test, score),
            **_subsets(y_test, score, prediction, winter, onset),
        }

    # Persistence: colder previous day means higher risk; the rule flags tomorrow when today
    # is already below the threshold. Missing values get the training median, as in the models.
    previous_min = X_test["min_lag_0"].fillna(X.loc[train, "min_lag_0"].median())
    baseline_score = -previous_min.to_numpy()
    rule = previous_cold.astype(int)
    baseline = {
        **_ranking(y_test, baseline_score),
        **_subsets(y_test, baseline_score, rule, winter, onset),
        "rule_today_cold_means_tomorrow_cold": _classification(y_test, rule),
    }

    payload = {
        "design": "capstone setup (25 lag features, 1-day horizon, default decision rule)",
        "fixes": [
            "single station",
            "GSOD sentinels converted to missing, median imputation fitted on train only",
            "calendar-contiguous 5-day windows and target day",
            "standardized inputs for Logistic Regression and SVM",
        ],
        "data_end": frame["DATE"].max().date().isoformat(),
        "train": {
            "samples": int(train.sum()),
            "positives": int(y.loc[train].sum()),
            "start": dates.loc[train].min().date().isoformat(),
            "end": dates.loc[train].max().date().isoformat(),
        },
        "test": {
            "samples": int(test.sum()),
            "positives": int(y.loc[test].sum()),
            "positive_rate": float(y.loc[test].mean()),
            "start": dates.loc[test].min().date().isoformat(),
            "end": dates.loc[test].max().date().isoformat(),
        },
        "overall_positive_rate": float(y.mean()),
        "subsets": {
            "winter_months": list(WINTER_MONTHS),
            "winter_days": int(winter.sum()),
            "onset_definition": "previous day's MIN not observed below 23F",
            "onset_days": int(onset.sum()),
            "onset_positives": int(y_test[onset].sum()),
            "continuation_positives": int(y_test[previous_cold].sum()),
        },
        "persistence_baseline": baseline,
        "models": results,
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _plot(y_test, scores, baseline_score, results, output)
    return payload


def _plot(y_test: pd.Series, scores: dict, baseline_score, results: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (name, metrics) in zip(axes, results.items()):
        ConfusionMatrixDisplay(
            pd.DataFrame(metrics["confusion_matrix"]).to_numpy(),
            display_labels=["Not extreme", "Extreme cold"],
        ).plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
        ax.set_title(MODEL_LABELS[name])
    fig.tight_layout()
    fig.savefig(output / "confusion_matrices.png", dpi=160)
    plt.close(fig)

    for display, filename, title in (
        (RocCurveDisplay, "roc_curves.png", "ROC curves, 2019-2025 test"),
        (PrecisionRecallDisplay, "pr_curves.png", "Precision-recall curves, 2019-2025 test"),
    ):
        fig, ax = plt.subplots(figsize=(6, 5))
        for name, score in scores.items():
            display.from_predictions(y_test, score, name=MODEL_LABELS[name], ax=ax)
        display.from_predictions(
            y_test, baseline_score, name=BASELINE_LABEL, ax=ax, color="gray", linestyle="--"
        )
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)

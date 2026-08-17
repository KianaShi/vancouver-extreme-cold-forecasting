from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

THRESHOLD_METHOD = "maximum F1 on the validation period"

LIGHTGBM_PARAMETERS = {
    "objective": "binary",
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 30,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}


def temporal_split(X: pd.DataFrame, y: pd.Series, dates: pd.Series, test_start: str):
    boundary = pd.Timestamp(test_start)
    train = dates.lt(boundary)
    test = dates.ge(boundary)
    if not train.any() or not test.any():
        raise ValueError("test_start must leave observations on both sides")
    if dates.loc[train].max() >= dates.loc[test].min():
        raise AssertionError("Temporal split overlaps")
    return X.loc[train], X.loc[test], y.loc[train], y.loc[test], dates.loc[test]


def temporal_partitions(dates: pd.Series, test_start: str = "2019-01-01") -> dict[str, pd.Series]:
    """Create chronological train, validation, and strict OOT test masks."""
    test_mask = dates.ge(pd.Timestamp(test_start))
    pretest_index = dates.index[~test_mask]
    if len(pretest_index) < 10 or not test_mask.any():
        raise ValueError("Not enough observations for temporal partitions")

    validation_start = int(len(pretest_index) * 0.8)
    masks = {
        "train": pd.Series(dates.index.isin(pretest_index[:validation_start]), index=dates.index),
        "validation": pd.Series(
            dates.index.isin(pretest_index[validation_start:]), index=dates.index
        ),
        "test": pd.Series(test_mask.to_numpy(), index=dates.index),
    }
    if dates.loc[masks["train"]].max() >= dates.loc[masks["validation"]].min():
        raise AssertionError("Train and validation overlap")
    if dates.loc[masks["validation"]].max() >= dates.loc[masks["test"]].min():
        raise AssertionError("Validation and test overlap")
    return masks


def candidate_models(columns: list[str]) -> dict[str, Pipeline]:
    scaled = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                columns,
            )
        ]
    )
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocess", scaled),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000, class_weight="balanced", random_state=42
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "lightgbm": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median").set_output(transform="pandas"),
                ),
                ("model", LGBMClassifier(**LIGHTGBM_PARAMETERS)),
            ]
        ),
    }


def choose_threshold(y_true: pd.Series, probability) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[scores.argmax()])


def classification_metrics(y_true, probability, threshold) -> dict:
    prediction = (probability >= threshold).astype(int)
    return {
        "positive_rate": float(y_true.mean()),
        "average_precision": float(average_precision_score(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "confusion_matrix": confusion_matrix(y_true, prediction).tolist(),
    }


def _period(dates: pd.Series, mask: pd.Series) -> dict[str, object]:
    selected = dates.loc[mask]
    return {
        "rows": len(selected),
        "start": selected.min().date().isoformat(),
        "end": selected.max().date().isoformat(),
    }


def train_and_evaluate(
    X, y, dates, output_dir: str | Path, test_start: str = "2019-01-01"
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    masks = temporal_partitions(dates, test_start)
    pretest_mask = masks["train"] | masks["validation"]

    results, final_models = {}, {}
    for name, template in candidate_models(list(X.columns)).items():
        validation_model = clone(template)
        validation_model.fit(X.loc[masks["train"]], y.loc[masks["train"]])
        validation_probability = validation_model.predict_proba(X.loc[masks["validation"]])[:, 1]
        threshold = choose_threshold(y.loc[masks["validation"]], validation_probability)

        final_model = clone(template)
        started = time.perf_counter()
        final_model.fit(X.loc[pretest_mask], y.loc[pretest_mask])
        fit_seconds = time.perf_counter() - started
        test_probability = final_model.predict_proba(X.loc[masks["test"]])[:, 1]
        results[name] = {
            "threshold": threshold,
            "threshold_method": THRESHOLD_METHOD,
            "final_fit_seconds": fit_seconds,
            "validation": classification_metrics(
                y.loc[masks["validation"]], validation_probability, threshold
            ),
            "test": classification_metrics(
                y.loc[masks["test"]], test_probability, threshold
            ),
        }
        final_models[name] = final_model

    # Test scores are reported once and never used for model or threshold selection.
    tree_candidates = ("random_forest", "lightgbm")
    winner = max(
        tree_candidates,
        key=lambda name: results[name]["validation"]["average_precision"],
    )
    joblib.dump(
        {"model": final_models[winner], "threshold": results[winner]["threshold"]},
        output / "model.joblib",
    )
    payload = {
        "split_policy": {
            "test_start": test_start,
            "validation_fraction_of_pretest": 0.2,
            "partitions": {name: _period(dates, mask) for name, mask in masks.items()},
            "final_refit": "train plus validation, after threshold and model selection",
        },
        "selection_policy": "highest validation average precision among Random Forest and LightGBM",
        "selected_model": winner,
        "threshold_policy": THRESHOLD_METHOD,
        "probability_calibration": "not fitted; reliability diagram is diagnostic only",
        "test_positive_rate": float(y.loc[masks["test"]].mean()),
        "lightgbm_parameters": LIGHTGBM_PARAMETERS,
        "models": results,
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 6))
    for name in tree_candidates:
        CalibrationDisplay.from_predictions(
            y.loc[masks["test"]],
            final_models[name].predict_proba(X.loc[masks["test"]])[:, 1],
            n_bins=8,
            name=name,
            ax=ax,
        )
    ax.set_title("Strict out-of-time reliability diagram (diagnostic only)")
    fig.tight_layout()
    fig.savefig(output / "calibration.png", dpi=160)
    plt.close(fig)
    return payload

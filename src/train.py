"""Train baselines and candidate models, select on validation data, lock test evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_pipeline import FEATURES, TARGET, load_dataset, split_xy, temporal_split
from src.evaluate import (
    classification_metrics,
    save_confusion_plot,
    save_precision_recall_plot,
    stratified_bootstrap_pr_auc,
    tune_threshold_by_cost,
)

SEED = 42
FN_COST = 25.0
FP_COST = 1.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_candidates() -> dict[str, Pipeline]:
    return {
        "dummy_prior": Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("model", DummyClassifier(strategy="prior"))]
        ),
        "weighted_logistic": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=SEED
                    ),
                ),
            ]
        ),
        "weighted_hist_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.08,
                        max_iter=200,
                        max_leaf_nodes=31,
                        min_samples_leaf=30,
                        l2_regularization=1.0,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "weighted_extra_trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=50,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }


def reference_profile(frame: pd.DataFrame) -> dict:
    profile: dict[str, dict] = {}
    for column in FEATURES:
        values = frame[column].to_numpy(dtype=float)
        edges = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
        if len(edges) < 3:
            edges = np.array([values.min() - 1e-9, values.max() + 1e-9])
        edges[0], edges[-1] = -np.inf, np.inf
        counts, _ = np.histogram(values, bins=edges)
        proportions = (counts / counts.sum()).tolist()
        profile[column] = {
            "bin_edges": [None if not np.isfinite(x) else float(x) for x in edges],
            "bin_proportions": [float(x) for x in proportions],
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/creditcard.csv"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (args.report_dir / "figures").mkdir(parents=True, exist_ok=True)

    raw = load_dataset(args.data)
    partitions = temporal_split(raw)
    X_train, y_train = split_xy(partitions.train)
    X_val, y_val = split_xy(partitions.validation)
    X_test, y_test = split_xy(partitions.test)

    trained: dict[str, Pipeline] = {}
    metrics: dict[str, dict] = {}
    for name, model in model_candidates().items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        val_scores = model.predict_proba(X_val)[:, 1]
        operating_point = tune_threshold_by_cost(y_val, val_scores, FN_COST, FP_COST)
        threshold = operating_point["threshold"]
        test_scores = model.predict_proba(X_test)[:, 1]
        metrics[name] = {
            "fit_seconds": fit_seconds,
            "validation": classification_metrics(y_val, val_scores, threshold),
            "test": classification_metrics(y_test, test_scores, threshold),
            "threshold_selection": operating_point,
        }
        trained[name] = model

    # Selection rule is fixed before reading locked-test metrics: minimize validation
    # business cost; break ties by higher validation PR-AUC, then simpler name order.
    selected_name = min(
        metrics,
        key=lambda name: (
            metrics[name]["validation"]["business_cost"],
            -metrics[name]["validation"]["pr_auc"],
            name,
        ),
    )
    selected_model = trained[selected_name]
    selected_threshold = metrics[selected_name]["validation"]["threshold"]
    selected_test_scores = selected_model.predict_proba(X_test)[:, 1]
    bootstrap = stratified_bootstrap_pr_auc(y_test, selected_test_scores, n_bootstrap=1000, seed=SEED)

    data_hash = sha256(args.data)
    model_version = f"2026.11.23-{selected_name}-{data_hash[:8]}"
    metadata = {
        "model_version": model_version,
        "selected_model": selected_name,
        "selection_rule": "minimum validation cost (25*FN + 1*FP), tie-break by validation PR-AUC",
        "decision_threshold": float(selected_threshold),
        "fallback_fraud_prior": float(y_train.mean()),
        "feature_order": FEATURES,
        "target": TARGET,
        "data_sha256": data_hash,
        "data_rows_raw": int(len(raw)),
        "random_seed": SEED,
        "false_negative_cost": FN_COST,
        "false_positive_cost": FP_COST,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "leakage_audit": partitions.audit,
        "test_pr_auc_bootstrap_ci": bootstrap,
    }

    joblib.dump(selected_model, args.artifact_dir / "model.joblib", compress=3)
    (args.artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (args.artifact_dir / "reference_profile.json").write_text(json.dumps(reference_profile(X_train), indent=2))
    (args.report_dir / "metrics" / "model_metrics.json").write_text(
        json.dumps({"selected_model": selected_name, "models": metrics, "bootstrap": bootstrap}, indent=2)
    )
    (args.report_dir / "metrics" / "leakage_audit.json").write_text(json.dumps(partitions.audit, indent=2))

    rows = []
    test_curves = {}
    for name, model in trained.items():
        test_scores = model.predict_proba(X_test)[:, 1]
        test_curves[name] = (y_test.to_numpy(), test_scores)
        for split in ("validation", "test"):
            rows.append({"model": name, "split": split, **metrics[name][split]})
    pd.DataFrame(rows).to_csv(args.report_dir / "metrics" / "model_metrics.csv", index=False)
    save_precision_recall_plot(test_curves, args.report_dir / "figures" / "test_precision_recall.png")
    save_confusion_plot(metrics[selected_name]["test"], args.report_dir / "figures" / "test_confusion_matrix.png")

    test_output = pd.DataFrame(
        {
            "record_id": partitions.test.index,
            "Time": X_test["Time"].to_numpy(),
            "label": y_test.to_numpy(),
            "fraud_probability": selected_test_scores,
            "prediction": (selected_test_scores >= selected_threshold).astype(int),
        }
    )
    test_output.to_csv(args.report_dir / "metrics" / "locked_test_predictions.csv", index=False)
    print(json.dumps({"selected_model": selected_name, "metadata": metadata, "metrics": metrics[selected_name]}, indent=2))


if __name__ == "__main__":
    main()

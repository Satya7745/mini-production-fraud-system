"""Empirical monitoring experiment: natural temporal holdout and injected drift."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from monitoring.drift import evaluate_drift
from src.data_pipeline import FEATURES, load_dataset, temporal_split


def summarize(report):
    actionable = report.loc[report.feature != "Time"]
    return {
        "critical_features": actionable.loc[actionable.severity == "critical", "feature"].tolist(),
        "warning_features": actionable.loc[actionable.severity == "warning", "feature"].tolist(),
        "top_5_by_psi": actionable.head(5)[["feature", "psi", "ks_statistic", "ks_pvalue"]].to_dict(orient="records"),
        "note": "Time is monitored for range/monotonicity but excluded from drift paging because elapsed time must advance.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/creditcard.csv"))
    parser.add_argument("--profile", type=Path, default=Path("artifacts/reference_profile.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/metrics"))
    args = parser.parse_args()
    parts = temporal_split(load_dataset(args.data))
    reference = parts.train[FEATURES]
    current = parts.test[FEATURES]
    natural = evaluate_drift(reference, current, args.profile)
    shifted = current.copy()
    shifted["Amount"] = shifted["Amount"] * 3.0 + 25.0
    shifted["V14"] = shifted["V14"] + 2.5
    injected = evaluate_drift(reference, shifted, args.profile)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    natural.to_csv(args.output_dir / "natural_temporal_drift.csv", index=False)
    injected.to_csv(args.output_dir / "injected_drift.csv", index=False)
    summary = {"natural_temporal_holdout": summarize(natural), "injected_shift": summarize(injected)}
    (args.output_dir / "drift_experiment.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

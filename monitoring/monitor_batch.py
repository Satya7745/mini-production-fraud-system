"""CLI for reproducible data-drift analysis and alert generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from monitoring.drift import evaluate_drift
from src.data_pipeline import FEATURES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=Path("artifacts/reference_profile.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/metrics/drift_report.csv"))
    args = parser.parse_args()
    reference = pd.read_csv(args.reference, usecols=FEATURES)
    current = pd.read_csv(args.current, usecols=FEATURES)
    report = evaluate_drift(reference, current, args.profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    actionable = report.loc[report.feature != "Time"]
    summary = {
        "status": "critical" if (actionable.severity == "critical").any() else "warning" if (actionable.severity == "warning").any() else "stable",
        "critical_features": actionable.loc[actionable.severity == "critical", "feature"].tolist(),
        "warning_features": actionable.loc[actionable.severity == "warning", "feature"].tolist(),
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

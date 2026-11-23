"""Vectorized batch inference CLI with record IDs and throughput reporting."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from serving.predictor import Predictor
from src.data_pipeline import FEATURES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    features = frame[FEATURES].apply(pd.to_numeric, errors="raise")
    predictor = Predictor(args.artifact_dir)
    start = time.perf_counter()
    probabilities = predictor.model.predict_proba(features)[:, 1]
    elapsed = time.perf_counter() - start
    record_id = frame["record_id"] if "record_id" in frame else frame.index.astype(str)
    output = pd.DataFrame(
        {
            "record_id": record_id,
            "fraud_probability": probabilities,
            "fraud_prediction": (probabilities >= predictor.threshold).astype(int),
            "model_version": predictor.version,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(json.dumps({"rows": len(output), "elapsed_seconds": elapsed, "records_per_second": len(output) / elapsed, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

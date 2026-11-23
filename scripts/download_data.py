"""Download the public OpenML credit-card fraud dataset (OpenML data_id=1597)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

EXPECTED_ROWS = 284_807
EXPECTED_FRAUDS = 492
EXPECTED_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/creditcard.csv"))
    args = parser.parse_args()
    dataset = fetch_openml(data_id=1597, as_frame=True, parser="auto")
    frame = dataset.data.copy()
    frame["Class"] = pd.to_numeric(dataset.target, errors="raise").astype(int)
    frame = frame[EXPECTED_COLUMNS]
    if len(frame) != EXPECTED_ROWS or int(frame["Class"].sum()) != EXPECTED_FRAUDS:
        raise RuntimeError("Downloaded dataset failed row/class-count integrity checks")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame):,} rows to {args.output}")


if __name__ == "__main__":
    main()

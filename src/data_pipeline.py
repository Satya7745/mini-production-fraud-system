"""Dataset validation and leakage-safe temporal partitioning.

The credit-card dataset is already ordered by elapsed transaction time. We retain
that causal order, remove only exact feature duplicates by keeping the first
occurrence, and create contiguous train/validation/test windows. No target
information is used to derive the split boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TARGET = "Class"
FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
REQUIRED_COLUMNS = [*FEATURES, TARGET]


@dataclass(frozen=True)
class TemporalPartitions:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    audit: dict[str, Any]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and strictly validate the canonical transaction schema."""
    frame = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(REQUIRED_COLUMNS))
    if missing or extra:
        raise ValueError(f"Schema mismatch; missing={missing}, extra={extra}")
    frame = frame[REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Features contain NaN or infinite values")
    if not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Class must be binary {0, 1}")
    if (frame["Time"] < 0).any() or (frame["Amount"] < 0).any():
        raise ValueError("Time and Amount must be non-negative")
    return frame


def temporal_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> TemporalPartitions:
    """Create contiguous temporal partitions and a machine-readable leakage audit.

    Quantile cutoffs depend only on the observed time index, never on the label.
    Equal timestamps stay in the same partition. Duplicate feature vectors keep
    their earliest occurrence, preventing an identical record from crossing a
    boundary.
    """
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1):
        raise ValueError("Fractions must be in (0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than 1")

    ordered = frame.sort_values("Time", kind="mergesort").reset_index(drop=True)
    duplicate_count = int(ordered.duplicated(subset=FEATURES, keep="first").sum())
    clean = ordered.drop_duplicates(subset=FEATURES, keep="first").reset_index(drop=True)

    train_cutoff = float(clean["Time"].quantile(train_fraction, interpolation="higher"))
    val_cutoff = float(
        clean["Time"].quantile(train_fraction + validation_fraction, interpolation="higher")
    )
    train = clean.loc[clean["Time"] <= train_cutoff].copy()
    validation = clean.loc[(clean["Time"] > train_cutoff) & (clean["Time"] <= val_cutoff)].copy()
    test = clean.loc[clean["Time"] > val_cutoff].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("A temporal partition is empty")
    if not (train["Time"].max() < validation["Time"].min() <= validation["Time"].max() < test["Time"].min()):
        raise AssertionError("Temporal windows overlap")

    feature_hashes = {
        name: set(pd.util.hash_pandas_object(part[FEATURES], index=False).astype(str))
        for name, part in {"train": train, "validation": validation, "test": test}.items()
    }
    overlaps = {
        "train_validation": len(feature_hashes["train"] & feature_hashes["validation"]),
        "train_test": len(feature_hashes["train"] & feature_hashes["test"]),
        "validation_test": len(feature_hashes["validation"] & feature_hashes["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"Duplicate feature vectors cross partitions: {overlaps}")

    partition_stats = {}
    for name, part in (("train", train), ("validation", validation), ("test", test)):
        partition_stats[name] = {
            "rows": int(len(part)),
            "fraud_count": int(part[TARGET].sum()),
            "fraud_rate": float(part[TARGET].mean()),
            "time_min": float(part["Time"].min()),
            "time_max": float(part["Time"].max()),
        }

    audit = {
        "raw_rows": int(len(frame)),
        "deduplicated_rows": int(len(clean)),
        "feature_duplicates_removed": duplicate_count,
        "train_cutoff_seconds": train_cutoff,
        "validation_cutoff_seconds": val_cutoff,
        "partitions": partition_stats,
        "cross_partition_feature_overlap": overlaps,
        "target_used_for_cutoffs": False,
        "preprocessing_fit_scope": "train_only",
    }
    return TemporalPartitions(train=train, validation=validation, test=test, audit=audit)


def split_xy(partition: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return partition[FEATURES].copy(), partition[TARGET].astype(int).copy()

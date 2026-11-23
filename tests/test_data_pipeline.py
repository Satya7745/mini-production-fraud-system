import numpy as np
import pandas as pd

from src.data_pipeline import FEATURES, TARGET, temporal_split


def synthetic_frame(rows: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(rows, len(FEATURES))), columns=FEATURES)
    frame["Time"] = np.arange(rows, dtype=float)
    frame["Amount"] = np.abs(frame["Amount"])
    frame[TARGET] = (np.arange(rows) % 17 == 0).astype(int)
    return frame


def test_temporal_split_is_ordered_disjoint_and_target_free():
    parts = temporal_split(synthetic_frame())
    assert parts.train.Time.max() < parts.validation.Time.min()
    assert parts.validation.Time.max() < parts.test.Time.min()
    assert all(value == 0 for value in parts.audit["cross_partition_feature_overlap"].values())
    assert parts.audit["target_used_for_cutoffs"] is False
    assert parts.audit["preprocessing_fit_scope"] == "train_only"


def test_duplicate_feature_vector_keeps_earliest_once():
    frame = synthetic_frame()
    duplicate = frame.iloc[[10]].copy()
    doubled = pd.concat([frame, duplicate], ignore_index=True)
    parts = temporal_split(doubled)
    total = len(parts.train) + len(parts.validation) + len(parts.test)
    assert total == len(frame)
    assert parts.audit["feature_duplicates_removed"] == 1

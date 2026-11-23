"""Feature drift checks using train-reference PSI and two-sample KS tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.data_pipeline import FEATURES

EPSILON = 1e-6


def population_stability_index(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(np.asarray(expected, dtype=float), EPSILON, None)
    actual = np.clip(np.asarray(actual, dtype=float), EPSILON, None)
    expected /= expected.sum()
    actual /= actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def evaluate_drift(
    reference_frame: pd.DataFrame,
    current_frame: pd.DataFrame,
    reference_profile_path: str | Path,
) -> pd.DataFrame:
    profile = json.loads(Path(reference_profile_path).read_text())
    records = []
    for feature in FEATURES:
        spec = profile[feature]
        edges = np.array([
            -np.inf if value is None and i == 0 else np.inf if value is None else float(value)
            for i, value in enumerate(spec["bin_edges"])
        ])
        counts, _ = np.histogram(current_frame[feature].to_numpy(dtype=float), bins=edges)
        actual = counts / max(1, counts.sum())
        psi = population_stability_index(np.array(spec["bin_proportions"]), actual)
        ks = ks_2samp(
            reference_frame[feature].to_numpy(dtype=float),
            current_frame[feature].to_numpy(dtype=float),
            alternative="two-sided",
            method="asymp",
        )
        severity = "critical" if psi >= 0.25 else "warning" if psi >= 0.10 else "stable"
        records.append(
            {
                "feature": feature,
                "psi": psi,
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "severity": severity,
            }
        )
    return pd.DataFrame(records).sort_values(["psi", "feature"], ascending=[False, True])

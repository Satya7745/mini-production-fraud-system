"""Thread-safe artifact loading and fail-safe scoring."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.data_pipeline import FEATURES

LOGGER = logging.getLogger(__name__)


@dataclass
class ScoreResult:
    fraud_probability: float
    fraud_prediction: int
    decision_threshold: float
    model_version: str
    fallback_used: bool


class Predictor:
    """Load artifacts once; return prior-risk fallback if scoring raises."""

    def __init__(self, artifact_dir: str | Path) -> None:
        artifact_dir = Path(artifact_dir)
        self.model = joblib.load(artifact_dir / "model.joblib")
        # Tree ensembles trained with n_jobs=-1 are efficient for large batches but
        # incur heavy process/thread scheduling overhead on one-row API calls.
        # Use one inference thread per request; horizontal API concurrency supplies
        # parallelism at the service layer.
        estimator = getattr(self.model, "named_steps", {}).get("model")
        if estimator is not None and hasattr(estimator, "n_jobs"):
            estimator.n_jobs = 1
        self.metadata: dict[str, Any] = json.loads((artifact_dir / "metadata.json").read_text())
        self.threshold = float(self.metadata["decision_threshold"])
        self.prior = float(self.metadata["fallback_fraud_prior"])
        self.version = str(self.metadata["model_version"])

    def score(self, payload: dict[str, float]) -> ScoreResult:
        try:
            frame = pd.DataFrame([[payload[name] for name in FEATURES]], columns=FEATURES)
            probability = float(self.model.predict_proba(frame)[0, 1])
            fallback = False
        except Exception:  # pragma: no cover - exercised by explicit failure injection test
            LOGGER.exception("Model scoring failed; serving conservative training-prior fallback")
            probability = self.prior
            fallback = True
        return ScoreResult(
            fraud_probability=probability,
            fraud_prediction=int(probability >= self.threshold),
            decision_threshold=self.threshold,
            model_version=self.version,
            fallback_used=fallback,
        )

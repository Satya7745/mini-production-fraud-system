"""Strict request and response contracts for online inference."""
from __future__ import annotations

import math
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFiniteFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    Time: NonNegativeFiniteFloat
    V1: FiniteFloat
    V2: FiniteFloat
    V3: FiniteFloat
    V4: FiniteFloat
    V5: FiniteFloat
    V6: FiniteFloat
    V7: FiniteFloat
    V8: FiniteFloat
    V9: FiniteFloat
    V10: FiniteFloat
    V11: FiniteFloat
    V12: FiniteFloat
    V13: FiniteFloat
    V14: FiniteFloat
    V15: FiniteFloat
    V16: FiniteFloat
    V17: FiniteFloat
    V18: FiniteFloat
    V19: FiniteFloat
    V20: FiniteFloat
    V21: FiniteFloat
    V22: FiniteFloat
    V23: FiniteFloat
    V24: FiniteFloat
    V25: FiniteFloat
    V26: FiniteFloat
    V27: FiniteFloat
    V28: FiniteFloat
    Amount: NonNegativeFiniteFloat

    @field_validator("Time", "Amount", *[f"V{i}" for i in range(1, 29)])
    @classmethod
    def finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("value must be finite")
        return value


class PredictionResponse(BaseModel):
    fraud_probability: float = Field(ge=0, le=1)
    fraud_prediction: int = Field(ge=0, le=1)
    decision_threshold: float = Field(ge=0, le=1)
    model_version: str
    fallback_used: bool
    request_id: str


class HealthResponse(BaseModel):
    status: str
    model_version: str | None
    artifact_loaded: bool

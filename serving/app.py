"""FastAPI application exposing health and transaction scoring endpoints."""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from serving.predictor import Predictor
from serving.schemas import HealthResponse, PredictionResponse, Transaction

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger(__name__)
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact_dir = Path(os.getenv("MODEL_ARTIFACT_DIR", str(DEFAULT_ARTIFACT_DIR)))
    try:
        app.state.predictor = Predictor(artifact_dir)
        LOGGER.info("Loaded model artifact from %s", artifact_dir)
    except Exception:
        app.state.predictor = None
        LOGGER.exception("Model artifact failed to load")
    yield


app = FastAPI(
    title="Leak-Free Credit-Card Fraud Scoring API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def observability(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    request.state.request_id = request_id
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        LOGGER.exception("Unhandled request failure", extra={"request_id": request_id})
        response = JSONResponse(status_code=500, content={"detail": "Internal scoring error", "request_id": request_id})
    latency_ms = (time.perf_counter() - start) * 1000
    response.headers["x-request-id"] = request_id
    response.headers["x-latency-ms"] = f"{latency_ms:.3f}"
    LOGGER.info(
        "request_complete method=%s path=%s status=%s latency_ms=%.3f request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
        request_id,
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    predictor = getattr(request.app.state, "predictor", None)
    return HealthResponse(
        status="ok" if predictor is not None else "degraded",
        model_version=predictor.version if predictor is not None else None,
        artifact_loaded=predictor is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction, request: Request) -> PredictionResponse:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model artifact unavailable")
    result = predictor.score(transaction.model_dump())
    return PredictionResponse(
        **result.__dict__,
        request_id=request.state.request_id,
    )

# Mini Production ML System - Credit-Card Fraud Detection

**Author:** Akella Sri Satya Vijay  
**BITS ID:** 2025EM1200154  
**Course:** ML Model Engineering, BITS Pilani WILP  
**Repository:** https://github.com/v-srakella_microsoft/mini-production-fraud-system

A reproducible, leakage-audited fraud scoring system built from the public ULB European credit-card dataset. The repository contains deterministic temporal splitting, train-only preprocessing, baseline/candidate comparisons, validation-only cost thresholding, a serialized artifact, strict FastAPI contracts, empirical latency and drift experiments, batch scoring, tests, CI, and optional Docker packaging.

## Headline results

| Item | Locked result |
|---|---:|
| Raw data | 284,807 transactions; 492 frauds (0.173%) |
| Temporal train / validation / test | 170,236 / 56,746 / 56,744 after feature deduplication |
| Selected model | Class-weighted ExtraTrees, 50 trees |
| Decision threshold | 0.07984, selected only on validation cost `25*FN + FP` |
| Test PR-AUC | 0.7905 (95% stratified bootstrap CI 0.7021-0.8721) |
| Test precision / recall / F1 | 0.5043 / 0.7973 / 0.6178 |
| Test confusion matrix | TN 56,612; FP 58; FN 15; TP 59 |
| HTTP benchmark | 2,000 requests, 8 concurrent, 100 warm-up; 0 errors |
| p50 / p95 / p99 | 123.09 / 190.26 / 222.02 ms |
| Throughput | 63.15 requests/s |

Accuracy is intentionally not the selection metric: at 0.13% test prevalence it would obscure fraud recall. PR-AUC measures minority-class ranking, while a validation-only asymmetric cost translates missed-fraud harm into the deployed threshold.

## Leakage controls

1. Sort by observed `Time` and retain causal order; no shuffle.
2. Remove exact duplicate feature vectors by keeping the earliest event.
3. Keep equal timestamps in one partition; create contiguous 60/20/20 windows.
4. Derive cutoffs from `Time` only, not `Class`.
5. Fit imputers, scaler, class-weighted estimator, and threshold using train/validation only.
6. Open the test window once after model/threshold selection.
7. Record zero feature-hash overlap across train/validation/test in `reports/metrics/leakage_audit.json`.

## Reproduce from a clean environment

```bash
git clone https://github.com/v-srakella_microsoft/mini-production-fraud-system.git
cd mini-production-fraud-system
python3.12 -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/download_data.py --output data/creditcard.csv
python -m src.train --data data/creditcard.csv --artifact-dir artifacts --report-dir reports
pytest
```

The canonical input used for the reported run had SHA-256 `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`. The downloader verifies 284,807 rows, 492 positive labels, and the expected 31-column schema. If OpenML is temporarily unavailable, download the same ULB/Kaggle dataset and place `creditcard.csv` under `data/`.

## Serve and score

```bash
uvicorn serving.app:app --host 0.0.0.0 --port 8000
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data @data/sample_request.json
```

Generate a request from the checked-in sample:

```bash
python - <<'PY'
import json, pandas as pd
row = pd.read_csv('data/sample_transactions.csv').iloc[0].drop('record_id')
open('data/sample_request.json','w').write(json.dumps({k: float(v) for k,v in row.items()}))
PY
```

The Pydantic schema requires exactly `Time`, `V1`-`V28`, and `Amount`; it forbids unknown fields, missing fields, negative `Time`/`Amount`, NaN, and infinity. Validation failures return HTTP 422. Missing artifacts return 503. Every response has a request ID and measured latency header. If runtime scoring raises after successful startup, the response is marked `fallback_used=true` and returns the recorded training fraud prior rather than a fabricated model score.

### Batch path

```bash
python -m serving.predict_batch \
  --input data/sample_transactions.csv \
  --output reports/metrics/sample_predictions.csv \
  --artifact-dir artifacts
```

## Benchmark and monitoring

With the API running:

```bash
python -m serving.benchmark --requests 2000 --warmup 100 --concurrency 8
python -m scripts.run_drift_experiment --data data/creditcard.csv
python -m monitoring.monitor_batch \
  --reference data/reference.csv --current data/current.csv \
  --profile artifacts/reference_profile.json
```

Operational policy:

- **System:** page when 5-minute p95 exceeds 250 ms, error rate exceeds 1%, artifact health fails, or fallback exceeds 0.1%.
- **Data:** PSI 0.10-0.25 is warning; PSI >=0.25 is critical. Confirm with KS statistic/p-value and require two consecutive windows before retraining. `Time` is range/monotonicity monitored but excluded from PSI paging because elapsed time must advance.
- **Prediction:** monitor score quantiles, alert rate, fallback rate, and threshold crossings by amount band.
- **Concept/performance:** when delayed labels arrive, track PR-AUC, recall, precision, and cost weekly; retrain if PR-AUC falls >10% relative or recall falls below 0.70 for two labeled windows.
- **Reliability:** reject malformed inputs; 503 on startup artifact failure; training-prior runtime fallback with an explicit flag; retain the previous signed artifact for rollback.

The injected-drift experiment shifts `Amount` and `V14`; both cross the critical PSI threshold, demonstrating that the monitoring code detects known distribution changes.

## Repository layout

```text
artifacts/                 serialized model, metadata, reference bins
src/                       schema checks, temporal split, train/evaluate
serving/                   FastAPI, Pydantic, predictor, batch, benchmark
monitoring/                PSI/KS drift implementation and CLI
scripts/                   OpenML download and drift experiment
tests/                     split, schema, inference, fallback, PSI tests
reports/figures/           PR curve and confusion matrix
reports/metrics/           metrics, predictions, latency and drift evidence
data/sample_transactions.csv safe example records; full CSV ignored
.github/workflows/ci.yml    clean-environment CI test workflow
```

## Modeling decisions

The comparison includes a prior-only dummy, class-weighted logistic regression, class-weighted histogram gradient boosting, and class-weighted ExtraTrees. The model family and threshold are selected only from validation evidence. A 50-tree ExtraTrees deployment is retained after a matched-load ablation: compared with 300 trees, p95 decreased from 636.35 ms to 199.86 ms, while locked-test PR-AUC changed from 0.8054 to 0.7905 and test recall increased slightly from 0.7838 to 0.7973. The reported 2,000-request final run produced p95 190.26 ms.

## Limitations

- The PCA features are anonymized, which limits semantic root-cause analysis and fairness auditing.
- The data spans about two days in September 2013; current payment behavior may differ materially.
- A 25:1 false-negative/false-positive cost is an explicit illustrative assumption and must be replaced by issuer economics.
- A single historical temporal holdout is not a substitute for shadow deployment, label-delay handling, calibration review, and champion/challenger evaluation.
- The runtime prior fallback preserves availability but is not a fraud decision substitute; downstream systems must consume `fallback_used` and apply manual-review policy.

## Academic integrity and attribution

See `ACADEMIC_INTEGRITY.md`. The implementation and analysis are original for this submission. Standard libraries are pinned in `requirements.txt`; dataset and literature sources are explicitly attributed. Generated metric values are produced by the checked-in scripts rather than invented.

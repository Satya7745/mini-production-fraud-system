.PHONY: install data train test serve benchmark batch

install:
	python -m pip install -r requirements.txt

data:
	python scripts/download_data.py --output data/creditcard.csv

train:
	python -m src.train --data data/creditcard.csv --artifact-dir artifacts --report-dir reports

test:
	pytest

serve:
	uvicorn serving.app:app --host 0.0.0.0 --port 8000

benchmark:
	python -m serving.benchmark --requests 1000 --warmup 50 --concurrency 8

batch:
	python -m serving.predict_batch --input data/sample_transactions.csv --output reports/metrics/sample_predictions.csv

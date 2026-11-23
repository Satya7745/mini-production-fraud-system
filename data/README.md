# Data

The repository intentionally excludes `creditcard.csv` (approximately 144 MB).
Run `python scripts/download_data.py` to retrieve OpenML dataset **1597**, or place
the canonical ULB/Kaggle file at `data/creditcard.csv`.

Expected integrity checks:

- 284,807 rows
- 30 features: `Time`, anonymized PCA components `V1`-`V28`, and `Amount`
- Binary target `Class`
- 492 fraud observations

The training pipeline additionally computes and records the local source SHA-256,
then removes exact duplicate feature vectors by retaining their first temporal
occurrence before deriving contiguous time windows.

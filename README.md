# MILK10k skin-lesion classification — Milestone 1 (data pipeline)

## 1. Project

Goal: predict `diagnosis_1` (**Benign / Malignant / Indeterminate**) for a skin lesion from its images.

Dataset: **MILK10k** (ISIC), 5,240 lesions and 10,480 images — exactly one dermoscopic and one
clinical close-up image per lesion. Labels: `diagnosis_1` (3 classes) and an 11-class grouping in
`training_gt.csv` (AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL, NV, SCCKA, VASC).

- Source: ISIC Archive / MILK10k challenge download (https://challenge.isic-archive.com/data/)
- License: **CC BY-NC 4.0** (see `licenses/CC-BY-NC.txt` in the download)
- Attribution: *MILK study team* (from `attribution.txt` in the download)

## 2. Setup and run

Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Point the code at the data folder (never hard-coded in the code)
export MILK10K_DIR=~/data/milk10k    # folder containing metadata.csv, images/, supplements/training_gt.csv
```

Run order (every script writes only to `outputs/`):

```bash
python scripts/check_integrity.py    # B1
python scripts/eda.py                # B2
python scripts/quality_report.py     # B4
python scripts/make_splits.py        # B5
python scripts/compute_stats.py      # B6/B8: norm stats + class weights
pytest tests/                        # unit tests
```

## 3. Repository structure

TODO (tree)

## 4. Data handling

TODO

## 5. Key decisions and results

TODO — filled in after Part B.

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

```
milk10k-cv/
├── README.md                 project description, setup, how to run
├── SUBMISSION.md             name, repo URL, links to every deliverable
├── requirements.txt          packages (pinned versions)
├── src/                      reusable, importable code
│   ├── config.py             data path (MILK10K_DIR), seed, image size, class names
│   ├── data.py               load CSVs, labels, lesion table, image paths, availability check
│   ├── viz.py                image grid, class-balance plot
│   ├── splits.py             split_lesions(): lesion-level stratified train/val/test
│   ├── transforms.py         train_transform, eval_transform, augmentation table
│   └── dataset.py            LesionDataset, MilkImageDataset, get_dataloaders, sampler, aggregate_predictions
├── scripts/                  run once to create outputs
│   ├── check_integrity.py    B1  files exist and decode; image size summary
│   ├── eda.py                B2  class balance, findings table, gallery
│   ├── make_label_map.py     B3  label counts and label_map.json
│   ├── quality_report.py     B4  missing values, structure checks, leaky columns
│   ├── make_splits.py        B5  train/val/test CSVs with checks
│   ├── compute_stats.py      B6/B8 normalisation stats and class weights (train only)
│   └── check_pipeline.py     B7/B8 augmentation figure, batch checks, epoch time
├── notebooks/
│   ├── homework_part_a.ipynb Part A (A1–A3), executed
│   ├── homework_part_a.pdf   PDF export of the notebook
│   └── session2_eda.ipynb    scratch EDA (Session 2)
├── tests/                    pytest tests (splits, transforms, datasets)
├── outputs/                  generated files only: splits/, figures/, *.json, *.csv, *.md
└── reports/
    └── milestone1_report.md  Milestone 1 report (B9)
```

Reusable logic lives in `src/` and is imported by both the notebook and the scripts, so nothing is written twice. One-off steps that produce files are in `scripts/`; exploration is in `notebooks/`; everything generated goes to `outputs/`. No path is hard-coded: all code reads the data folder from `MILK10K_DIR` via `src/config.py`.

## 4. Data handling

- Images and the raw CSVs are **not** committed (see `.gitignore`); they stay in `MILK10K_DIR`.
- Everything derived from them **is** committed: `outputs/splits/{train,val,test}.csv`, `label_map.json`, `norm_stats.json`, `class_weights.json`, `lesions.csv`, `image_size_summary.csv`, the reports and the figures.
- Splits are made **by lesion** (both images of a lesion always stay together), stratified on the 11-class label.
- **Seed = 42.** Splits created on **2026-09-24** with `python scripts/make_splits.py`: 3,742 / 749 / 749 lesions (71.4 / 14.3 / 14.3 %).
- Normalisation statistics and class weights are computed on the **train split only**.

## 5. Key decisions and results

Full reasoning: [reports/milestone1_report.md](reports/milestone1_report.md).

- **Target:** `diagnosis_1` with 3 classes. Indeterminate (123 lesions) is kept as its own class, because it still needs follow-up. The 11-class `dx` is the stretch goal, with all classes kept and class weights.
- **Split:** lesion-level `StratifiedGroupKFold` in two steps. No overlap, and the largest class-proportion deviation is 0.10 pp. A3.1 shows why: a naive image split puts 80 % of test lesions' sibling images in train.
- **Inputs:** both views per lesion (`LesionDataset`). Leaky columns are never used: `diagnosis_2–4`, `diagnosis_confirm_type`, `image_manipulation`.
- **Preprocessing:** all images are 600×450 → `Resize(256)` + `CenterCrop(224)`. Normalisation: mean (0.686, 0.523, 0.473), std (0.124, 0.131, 0.151), from train.
- **Augmentation:** random crop (scale 0.8–1), flips, 90° rotations, mild ColorJitter (brightness/contrast 0.1, hue 0.02). Hue is kept below the Benign/Malignant colour gap measured in A3.5.
- **Imbalance:** 29 : 1 on train (Malignant vs Indeterminate). `WeightedRandomSampler` by default, or class-weighted loss (`outputs/class_weights.json`), not both.
- **Metrics:** balanced accuracy and Malignant recall, evaluated per lesion. The dataset is biopsy-enriched (69 % malignant), so accuracy and precision do not transfer to clinics.

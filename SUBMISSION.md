# Submission — Milestone 1 (MILK10k)

- **Name:** Alexander Nakashidze
- **Repository:** https://github.com/Lio02/milk10k-cv

## Deliverables

| item | link |
|---|---|
| Part A notebook | [notebooks/homework_part_a.ipynb](notebooks/homework_part_a.ipynb) |
| Part A PDF | [notebooks/homework_part_a.pdf](notebooks/homework_part_a.pdf) |
| README | [README.md](README.md) |
| Report (B9) | [reports/milestone1_report.md](reports/milestone1_report.md) |
| Splits | [train.csv](outputs/splits/train.csv) · [val.csv](outputs/splits/val.csv) · [test.csv](outputs/splits/test.csv) |
| Label map | [outputs/label_map.json](outputs/label_map.json) |
| Normalisation stats | [outputs/norm_stats.json](outputs/norm_stats.json) |
| Class weights | [outputs/class_weights.json](outputs/class_weights.json) |
| Data-quality report (B4) | [outputs/quality_report.md](outputs/quality_report.md) |
| EDA findings (B2) | [outputs/eda_findings.md](outputs/eda_findings.md) |
| Figures | [outputs/figures/](outputs/figures/) |
| Session 2 EDA | [notebooks/session2_eda.ipynb](notebooks/session2_eda.ipynb) |
| Tests | [tests/](tests/) |

## Source code (`src/`)

| file | content |
|---|---|
| [src/config.py](src/config.py) | data path from `MILK10K_DIR`, seed, image size |
| [src/data.py](src/data.py) | CSV loading, labels, lesion table, image paths |
| [src/viz.py](src/viz.py) | `show_grid`, `plot_class_balance` |
| [src/splits.py](src/splits.py) | `split_lesions` |
| [src/transforms.py](src/transforms.py) | `train_transform`, `eval_transform` |
| [src/dataset.py](src/dataset.py) | `LesionDataset`, `MilkImageDataset`, `get_dataloaders`, `aggregate_predictions` |

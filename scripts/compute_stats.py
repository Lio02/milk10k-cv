"""B6/B8 - Statistics computed on the TRAIN split only.

Usage:  python scripts/compute_stats.py      (run after make_splits.py)
Writes: outputs/norm_stats.json     per-channel mean/std of train images after Resize(256)+CenterCrop(224)
        outputs/class_weights.json  n_samples / (n_classes * count_per_class), for diagnosis_1 and dx
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from src import config
from src.dataset import load_rgb


class _PlainImages(Dataset):
    """Train images resized and cropped exactly like eval_transform, without normalisation."""

    def __init__(self, ids):
        self.ids = list(ids)
        self.tf = T.Compose([T.Resize(256), T.CenterCrop(config.IMG_SIZE), T.ToTensor()])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.tf(load_rgb(self.ids[i]))


def channel_stats(ids, batch_size=64, num_workers=min(4, os.cpu_count() or 1)):
    """Exact per-channel mean and std over all pixels (streamed sums, no need to hold images in RAM)."""
    loader = DataLoader(_PlainImages(ids), batch_size=batch_size, num_workers=num_workers)
    n, s, s2 = 0, torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    for x in loader:
        x = x.double()
        n += x.numel() // 3
        s += x.sum(dim=(0, 2, 3))
        s2 += (x ** 2).sum(dim=(0, 2, 3))
    mean = s / n
    std = (s2 / n - mean ** 2).sqrt()
    return mean.tolist(), std.tolist()


def class_weights(labels: pd.Series, classes) -> dict:
    counts = labels.value_counts().reindex(classes).fillna(0)
    weights = len(labels) / (len(classes) * counts)
    return {c: round(float(w), 6) for c, w in weights.items()}


def main():
    train = pd.read_csv(config.SPLIT_DIR / "train.csv")
    print(f"train split: {len(train):,} images, {train.lesion_id.nunique():,} lesions")

    mean, std = channel_stats(train.isic_id)
    stats = {"mean": [round(m, 6) for m in mean], "std": [round(s, 6) for s in std],
             "computed_on": "train split, all images (both views), after Resize(256) + CenterCrop(224), pixel values in [0, 1]",
             "n_images": int(len(train))}
    with open(config.OUTPUT_DIR / "norm_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("norm stats:", stats["mean"], stats["std"])

    with open(config.OUTPUT_DIR / "label_map.json") as f:
        label_map = json.load(f)
    weights = {
        "diagnosis_1": class_weights(train.diagnosis_1, list(label_map["diagnosis_1"])),
        "dx": class_weights(train.dx, list(label_map["dx"])),
        "formula": "n_samples / (n_classes * count_per_class), train split only",
    }
    with open(config.OUTPUT_DIR / "class_weights.json", "w") as f:
        json.dump(weights, f, indent=2)
    counts = train.drop_duplicates("lesion_id").diagnosis_1.value_counts()
    counts_dx = train.drop_duplicates("lesion_id").dx.value_counts()
    print("class weights diagnosis_1:", weights["diagnosis_1"])
    print(f"imbalance ratio on TRAIN (largest / smallest class, lesions): diagnosis_1 {counts.max()}/{counts.min()} = "
          f"{counts.max() / counts.min():.1f} | dx {counts_dx.max()}/{counts_dx.min()} = {counts_dx.max() / counts_dx.min():.1f}")
    print("saved outputs/norm_stats.json and outputs/class_weights.json")


if __name__ == "__main__":
    main()

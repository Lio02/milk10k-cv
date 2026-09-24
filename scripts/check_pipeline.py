"""B7/B8 - Visual and numerical checks of the finished data pipeline.

Usage:  python scripts/check_pipeline.py      (run after make_splits.py and compute_stats.py)
Writes: outputs/figures/augmentations.png   3 classes x (1 original + 7 augmented)
        outputs/figures/batch_check.png     16 transformed train images with their labels
        outputs/pipeline_check.txt          sanity prints (shapes, dtype, ranges, label histogram, epoch time)
"""
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import torch
import torchvision.transforms as T

from src import config
from src.dataset import get_dataloaders, load_label_map, load_rgb
from src.transforms import denormalize, train_transform_steps
from src.viz import show_grid

LOG = []


def log(*args):
    line = " ".join(str(a) for a in args)
    print(line)
    LOG.append(line)


def augmentation_figure():
    """One common, one medium and one rare class: original + 7 augmented versions."""
    train = pd.read_csv(config.SPLIT_DIR / "train.csv")
    derm = train[train.image_type == config.DERM]
    aug = T.Compose(train_transform_steps()[:-2])   # all augmentations, without ToTensor/Normalize
    torch.manual_seed(config.SEED)
    images, titles = [], []
    for dx in ["BCC", "MEL", "DF"]:
        isic_id = derm[derm.dx == dx].isic_id.iloc[0]
        img = load_rgb(isic_id)
        images.append(img.resize((config.IMG_SIZE * 4 // 3, config.IMG_SIZE)))
        titles.append(f"{dx} original")
        for k in range(7):
            images.append(aug(img))
            titles.append(f"{dx} aug {k + 1}")
    fig = show_grid(images, titles, ncols=8, size=1.9)
    fig.savefig(config.FIG_DIR / "augmentations.png", dpi=110, bbox_inches="tight")
    log("saved outputs/figures/augmentations.png")


def main():
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    augmentation_figure()

    label_map = load_label_map("diagnosis_1")
    inv = {v: k for k, v in label_map.items()}
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32, num_workers=4, balance="sampler")
    log(f"datasets: train {len(train_loader.dataset)} | val {len(val_loader.dataset)} | test {len(test_loader.dataset)} images")

    x, y, ids = next(iter(train_loader))
    log(f"train batch: x {tuple(x.shape)} {x.dtype}, min {x.min():.2f}, max {x.max():.2f}, mean {x.mean():.3f}; "
        f"y {tuple(y.shape)} {y.dtype}")
    xv, yv, _ = next(iter(val_loader))
    log(f"val batch:   x {tuple(xv.shape)} {xv.dtype}, min {xv.min():.2f}, max {xv.max():.2f}")

    # label histogram over 20 train batches: roughly balanced thanks to the sampler
    hist = Counter()
    for k, (_, yb, _) in enumerate(train_loader):
        hist.update(yb.tolist())
        if k == 19:
            break
    total = sum(hist.values())
    log("label histogram over 20 train batches (sampler):",
        {inv[c]: f"{n} ({n / total:.0%})" for c, n in sorted(hist.items())})
    raw = train_loader.dataset.labels
    log("label share in the train split itself:        ",
        {inv[c]: f"{(raw == c).mean():.0%}" for c in sorted(inv)})

    # time for one full train epoch (loading + augmentation only, no model)
    t0 = time.perf_counter()
    n = 0
    for xb, _, _ in train_loader:
        n += len(xb)
    log(f"one full train epoch: {n} images in {time.perf_counter() - t0:.1f} s "
        f"(batch_size=32, num_workers={train_loader.num_workers})")

    # visual check: 16 transformed train images, normalisation undone
    imgs = [denormalize(x[i]) for i in range(16)]
    fig = show_grid(imgs, [inv[int(v)] for v in y[:16]], ncols=8, size=1.9)
    fig.savefig(config.FIG_DIR / "batch_check.png", dpi=110, bbox_inches="tight")
    log("saved outputs/figures/batch_check.png")

    (config.OUTPUT_DIR / "pipeline_check.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()

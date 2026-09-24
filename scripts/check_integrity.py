"""B1 - Integrity check: every image listed in metadata.csv exists and can be decoded.

Usage:  python scripts/check_integrity.py
Writes: outputs/image_size_summary.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from PIL import Image

from src import config
from src.data import filter_available, image_path, load_metadata


def main():
    meta = load_metadata()

    # 1. Every file exists (raises FileNotFoundError listing the missing files otherwise)
    filter_available(meta, strict=True)
    print(f"{len(meta):,} / {len(meta):,} image files found")

    # 2. Every file can be decoded; read the size from a fresh handle (verify() makes the file unusable)
    rows, unreadable = [], []
    for k, isic_id in enumerate(meta.isic_id, 1):
        path = image_path(isic_id)
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                w, h = im.size
                rows.append({"isic_id": isic_id, "width": w, "height": h, "mode": im.mode, "format": im.format})
        except Exception as e:  # noqa: BLE001 - any decoding problem counts as unreadable
            unreadable.append((isic_id, repr(e)))
        if k % 2000 == 0:
            print(f"  checked {k:,} images")

    sizes = pd.DataFrame(rows)
    print(f"unreadable files: {len(unreadable)}")
    for isic_id, err in unreadable[:20]:
        print("  ", isic_id, err)

    summary = pd.DataFrame({
        "width": sizes.width.agg(["min", "median", "max"]),
        "height": sizes.height.agg(["min", "median", "max"]),
    })
    summary.loc["n_images"] = len(meta)
    summary.loc["n_unreadable"] = len(unreadable)
    summary.loc["share_short_side_ge_224"] = (sizes[["width", "height"]].min(axis=1) >= 224).mean()
    summary.loc["share_short_side_ge_256"] = (sizes[["width", "height"]].min(axis=1) >= 256).mean()
    print(summary)
    print("modes:", sizes["mode"].value_counts().to_dict(), "| formats:", sizes["format"].value_counts().to_dict())

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    summary.to_csv(config.OUTPUT_DIR / "image_size_summary.csv")
    print("saved outputs/image_size_summary.csv")


if __name__ == "__main__":
    main()

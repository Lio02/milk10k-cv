"""B5 - Create the lesion-level train / val / test split.

Usage:  python scripts/make_splits.py
Writes: outputs/splits/train.csv, val.csv, test.csv  (one row per IMAGE, both images of a lesion together)
        outputs/splits/split_summary.csv
"""
import itertools
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config
from src.data import build_lesion_table, image_table, load_gt, load_metadata
from src.splits import split_lesions

VAL_SIZE, TEST_SIZE = 0.15, 0.15


def main():
    lesions = build_lesion_table(load_metadata(), load_gt())
    ids = dict(zip(["train", "val", "test"],
                   split_lesions(lesions, val_size=VAL_SIZE, test_size=TEST_SIZE, seed=config.SEED, y_col="dx")))

    # Assign images afterwards: lesion list -> lesion rows -> both images
    splits = {name: image_table(lesions[lesions.lesion_id.isin(lesion_ids)]) for name, lesion_ids in ids.items()}

    # --- checks -------------------------------------------------------------------------------
    for a, b in itertools.combinations(splits, 2):
        overlap = set(splits[a].lesion_id) & set(splits[b].lesion_id)
        img_overlap = set(splits[a].isic_id) & set(splits[b].isic_id)
        assert not overlap and not img_overlap, f"overlap between {a} and {b}"
        print(f"overlap {a:5s}/{b:5s}: 0 lesions, 0 images")
    for name, df in splits.items():
        per_lesion = df.groupby("lesion_id").image_type.nunique()
        assert (df.groupby("lesion_id").size() == 2).all() and (per_lesion == 2).all()
    assert sum(len(d) for d in splits.values()) == 2 * len(lesions)
    print("every lesion has exactly 2 images (one per type) in its split\n")

    n = len(lesions)
    sizes = pd.DataFrame({name: [len(ids[name]), len(ids[name]) / n, len(df)] for name, df in splits.items()},
                         index=["lesions", "share", "images"]).T
    print(sizes.to_string(formatters={"lesions": "{:.0f}".format, "share": "{:.1%}".format, "images": "{:.0f}".format}), "\n")

    global_dx = lesions.dx.value_counts(normalize=True)
    dx_props = pd.DataFrame({name: lesions[lesions.lesion_id.isin(ids[name])].dx.value_counts(normalize=True)
                             for name in ids}).reindex(global_dx.index).fillna(0)
    dx_props["global"] = global_dx
    dev = (dx_props[["train", "val", "test"]].sub(dx_props["global"], axis=0)).abs() * 100
    dx_props["max_dev_pp"] = dev.max(axis=1)
    print("11-class proportions per split (%):")
    print((dx_props * [100, 100, 100, 100, 1]).round(2).to_string())
    print(f"\nmaximum deviation from the global proportions: {dev.values.max():.2f} percentage points\n")

    d1 = pd.DataFrame({name: lesions[lesions.lesion_id.isin(ids[name])].diagnosis_1.value_counts(normalize=True) * 100
                       for name in ids})
    d1["global"] = lesions.diagnosis_1.value_counts(normalize=True) * 100
    print("diagnosis_1 proportions per split (%):")
    print(d1.round(2).to_string())

    # --- save -----------------------------------------------------------------------------------
    config.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_csv(config.SPLIT_DIR / f"{name}.csv", index=False)
    summary = dx_props.round(5)
    summary.loc["_n_lesions"] = [len(ids["train"]), len(ids["val"]), len(ids["test"]), n, None]
    summary.to_csv(config.SPLIT_DIR / "split_summary.csv")
    print(f"\nsaved outputs/splits/{{train,val,test}}.csv  (seed={config.SEED}, created {date.today().isoformat()})")


if __name__ == "__main__":
    main()

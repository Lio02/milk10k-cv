"""Loading the MILK10k CSVs and resolving image paths.

Two tables are used throughout the project:
- the *image table*  (metadata.csv, one row per image, 10,480 rows)
- the *lesion table* (one row per lesion, 5,240 rows) built by build_lesion_table()
"""
from pathlib import Path

import pandas as pd

from src import config

IMG_EXT = ".jpg"  # checked: every file in images/ is ISIC_xxxxxxx.jpg


def load_metadata() -> pd.DataFrame:
    """Return metadata.csv as a DataFrame (one row per image)."""
    return pd.read_csv(config.METADATA_CSV)


def load_gt() -> pd.DataFrame:
    """Return training_gt.csv (one row per lesion, one-hot columns for the 11 classes)."""
    return pd.read_csv(config.GT_CSV)


def add_labels(meta: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Turn the one-hot ground truth into a single `dx` column and merge it on lesion_id.

    Returns a copy of `meta` with an extra `dx` column (e.g. "BCC").
    """
    labels = gt[["lesion_id"]].copy()
    labels["dx"] = gt[config.CLASSES].idxmax(axis=1)
    out = meta.merge(labels, on="lesion_id", how="left", validate="many_to_one")
    if out["dx"].isna().any():
        missing = out.loc[out["dx"].isna(), "lesion_id"].unique()[:10]
        raise ValueError(f"{out['dx'].isna().sum()} images have no ground truth, e.g. {list(missing)}")
    return out


def image_path(isic_id: str) -> Path:
    """Return the path of the image file for one isic_id."""
    return config.IMG_DIR / f"{isic_id}{IMG_EXT}"


def filter_available(df: pd.DataFrame, strict: bool = True, id_col: str = "isic_id") -> pd.DataFrame:
    """Check that the image file of every row exists.

    strict=True  -> raise FileNotFoundError listing the missing files ("fail loudly").
    strict=False -> silently keep only the rows whose file exists (handy for subsets).
    """
    exists = df[id_col].map(lambda i: image_path(i).exists())
    if strict:
        if not exists.all():
            missing = [str(image_path(i)) for i in df.loc[~exists, id_col]]
            shown = "\n  ".join(missing[:20])
            raise FileNotFoundError(f"{len(missing)} image files are missing:\n  {shown}")
        return df
    return df.loc[exists].copy()


def build_lesion_table(meta: pd.DataFrame) -> pd.DataFrame:
    """One row per lesion. Filled in during exercise A1.3."""
    raise NotImplementedError

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


LESION_COLS = ["age_approx", "sex", "anatom_site_general", "diagnosis_1", "diagnosis_2",
               "diagnosis_3", "diagnosis_confirm_type"]


def build_lesion_table(meta: pd.DataFrame, gt: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return one row per lesion (5,240 rows).

    Columns: lesion_id, derm_id, clinical_id (the isic_id of each view), the lesion-level
    metadata in LESION_COLS and the 11-class label `dx` from training_gt.csv.
    """
    if gt is None:
        gt = load_gt()
    wide = meta.pivot(index="lesion_id", columns="image_type", values="isic_id")
    wide = wide.rename(columns={config.DERM: "derm_id", config.CLINICAL: "clinical_id"})
    wide = wide[["derm_id", "clinical_id"]]
    wide.columns.name = None

    fields = meta.groupby("lesion_id")[LESION_COLS].first()
    dx = gt.set_index("lesion_id")[config.CLASSES].idxmax(axis=1).rename("dx")
    table = wide.join(fields).join(dx).reset_index()

    assert table.lesion_id.is_unique
    assert table[["derm_id", "clinical_id"]].notna().all().all(), "some lesions lack one of the two views"
    return table


def image_table(lesions: pd.DataFrame) -> pd.DataFrame:
    """Expand a lesion table into one row per image (isic_id, image_type + all lesion columns)."""
    parts = []
    for col, itype in [("derm_id", config.DERM), ("clinical_id", config.CLINICAL)]:
        part = lesions.drop(columns=["derm_id", "clinical_id"]).copy()
        part.insert(1, "isic_id", lesions[col].values)
        part.insert(2, "image_type", itype)
        parts.append(part)
    return pd.concat(parts, ignore_index=True).sort_values(["lesion_id", "image_type"], ignore_index=True)

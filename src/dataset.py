"""PyTorch Datasets for MILK10k.

LesionDataset  - one item per lesion with both views (dermoscopic + clinical).   (A3.6)
"""
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data import image_path


def load_rgb(isic_id: str) -> Image.Image:
    """Open one image as RGB, raising FileNotFoundError with the full path if it is missing."""
    path = image_path(isic_id)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    with Image.open(path) as im:
        return im.convert("RGB")


class LesionDataset(Dataset):
    """One item per lesion: {"derm", "clinical", "label", "lesion_id"}.

    lesions:   lesion table with columns lesion_id, derm_id, clinical_id and the label column
    label_map: dict label string -> int (e.g. {"Benign": 0, "Indeterminate": 1, "Malignant": 2})
    transform: applied separately to each view (so random augmentations differ per view)
    """

    def __init__(self, lesions: pd.DataFrame, label_map: dict, transform=None, label_col: str = "diagnosis_1"):
        self.table = lesions.reset_index(drop=True)
        self.label_map = label_map
        self.transform = transform
        self.labels = self.table[label_col].map(label_map)
        if self.labels.isna().any():
            unknown = self.table.loc[self.labels.isna(), label_col].unique()
            raise ValueError(f"labels not in label_map: {list(unknown)}")
        self.labels = self.labels.astype(int).to_numpy()

    def __len__(self):
        return len(self.table)

    def _view(self, isic_id):
        img = load_rgb(isic_id)
        return self.transform(img) if self.transform else img

    def __getitem__(self, idx):
        row = self.table.iloc[idx]
        return {
            "derm": self._view(row.derm_id),
            "clinical": self._view(row.clinical_id),
            "label": int(self.labels[idx]),
            "lesion_id": row.lesion_id,
        }


def aggregate_predictions(image_probs, image_table: pd.DataFrame) -> pd.DataFrame:
    """Average per-image class probabilities into one prediction per lesion.

    image_probs: array (n_images, n_classes), rows aligned with image_table
    image_table: DataFrame with a lesion_id column (same order as image_probs)
    Returns a DataFrame indexed by lesion_id with the mean probabilities and a `pred` column (argmax).
    """
    probs = pd.DataFrame(np.asarray(image_probs)).groupby(image_table.lesion_id.values).mean()
    probs.index.name = "lesion_id"
    probs["pred"] = probs.to_numpy().argmax(axis=1)
    return probs

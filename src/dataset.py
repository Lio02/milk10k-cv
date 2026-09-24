"""PyTorch Datasets and DataLoaders for MILK10k.

LesionDataset     - one item per lesion with both views (dermoscopic + clinical).   (A3.6)
MilkImageDataset  - one item per image, read from a split CSV.                     (B8)
get_dataloaders() - train / val / test loaders with seeding and imbalance handling. (B8)
"""
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src import config
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


class MilkImageDataset(Dataset):
    """One item per image: (image_tensor, label, isic_id).

    split_csv: path to outputs/splits/{train,val,test}.csv (or a DataFrame with the same columns)
    label_map: dict label string -> int
    label_col: "diagnosis_1" (primary target) or "dx" (11-class stretch goal)
    """

    def __init__(self, split_csv, label_map: dict, transform=None, label_col: str = "diagnosis_1"):
        self.table = (pd.read_csv(split_csv) if not isinstance(split_csv, pd.DataFrame) else split_csv).reset_index(drop=True)
        self.transform = transform
        labels = self.table[label_col].map(label_map)
        if labels.isna().any():
            raise ValueError(f"labels not in label_map: {list(self.table.loc[labels.isna(), label_col].unique())}")
        self.labels = labels.astype(int).to_numpy()
        missing = [str(image_path(i)) for i in self.table.isic_id if not image_path(i).exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} image files missing, e.g. {missing[:5]}")

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        isic_id = self.table.isic_id.iloc[idx]
        img = load_rgb(isic_id)
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx]), isic_id


def lesions_from_split(split_df: pd.DataFrame) -> pd.DataFrame:
    """Turn an image-level split CSV back into a lesion table (lesion_id, derm_id, clinical_id, labels)."""
    wide = split_df.pivot(index="lesion_id", columns="image_type", values="isic_id")
    wide = wide.rename(columns={config.DERM: "derm_id", config.CLINICAL: "clinical_id"})[["derm_id", "clinical_id"]]
    wide.columns.name = None
    labels = split_df.drop_duplicates("lesion_id").set_index("lesion_id")[["diagnosis_1", "dx"]]
    return wide.join(labels).reset_index()


# ---------------------------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------------------------
def seed_everything(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id):
    """worker_init_fn: give every DataLoader worker a seed derived from the torch seed."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = config.SEED):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ---------------------------------------------------------------------------------------------
# Imbalance
# ---------------------------------------------------------------------------------------------
def class_weights_from_labels(labels, n_classes: int) -> torch.Tensor:
    """n_samples / (n_classes * count_per_class) -> use as CrossEntropyLoss(weight=...)."""
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    return torch.tensor(len(labels) / (n_classes * counts), dtype=torch.float32)


def make_sampler(labels, seed: int = config.SEED) -> WeightedRandomSampler:
    """Oversample rare classes: weight 1/count[label] per sample, drawn with replacement."""
    counts = np.bincount(labels)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(labels),
                                 replacement=True, generator=make_generator(seed))


def load_label_map(target: str = "diagnosis_1") -> dict:
    with open(config.OUTPUT_DIR / "label_map.json") as f:
        return json.load(f)[target]


def get_dataloaders(batch_size: int = 32, num_workers: int = 4, target: str = "diagnosis_1",
                    balance: str = "sampler", seed: int = config.SEED, train_tf=None, eval_tf=None):
    """Return (train_loader, val_loader, test_loader) built from outputs/splits/*.csv.

    balance="sampler" -> WeightedRandomSampler on train (replaces shuffle=True);
    balance="shuffle" -> plain shuffling; use class weights in the loss instead
                         (outputs/class_weights.json). Use one of the two, not both.
    """
    from src.transforms import eval_transform, train_transform

    seed_everything(seed)
    num_workers = min(num_workers, os.cpu_count() or 1)
    label_map = load_label_map(target)
    ds = {name: MilkImageDataset(config.SPLIT_DIR / f"{name}.csv", label_map,
                                 transform=(train_tf or train_transform) if name == "train" else (eval_tf or eval_transform),
                                 label_col=target)
          for name in ["train", "val", "test"]}

    common = dict(batch_size=batch_size, num_workers=num_workers, worker_init_fn=seed_worker,
                  generator=make_generator(seed), pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
    if balance == "sampler":
        train_loader = DataLoader(ds["train"], sampler=make_sampler(ds["train"].labels, seed), **common)
    else:
        train_loader = DataLoader(ds["train"], shuffle=True, **common)
    val_loader = DataLoader(ds["val"], shuffle=False, **common)
    test_loader = DataLoader(ds["test"], shuffle=False, **common)
    return train_loader, val_loader, test_loader

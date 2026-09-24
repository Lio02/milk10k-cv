"""Image transforms (B7).

train_transform: random geometric augmentation + very mild colour jitter (see A3.5), then normalise.
eval_transform:  deterministic resize + centre crop, then normalise. Nothing random.

Normalisation statistics come from outputs/norm_stats.json (computed on the TRAIN split only by
scripts/compute_stats.py). If that file does not exist yet, ImageNet statistics are used.
"""
import json

import torchvision.transforms as T

from src import config

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def load_norm_stats():
    path = config.OUTPUT_DIR / "norm_stats.json"
    if path.exists():
        with open(path) as f:
            stats = json.load(f)
        return stats["mean"], stats["std"]
    return IMAGENET_MEAN, IMAGENET_STD


MEAN, STD = load_norm_stats()
SIZE = config.IMG_SIZE


def train_transform_steps(mean=MEAN, std=STD):
    return [
        T.RandomResizedCrop(SIZE, scale=(0.8, 1.0)),          # small zoom / crop jitter
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(180),                                # lesions have no orientation
        T.ColorJitter(brightness=0.1, contrast=0.1, hue=0.02),  # mild: colour is diagnostic (A3.5)
        T.ToTensor(),
        T.Normalize(mean, std),
    ]


def eval_transform_steps(mean=MEAN, std=STD):
    return [T.Resize(256), T.CenterCrop(SIZE), T.ToTensor(), T.Normalize(mean, std)]


train_transform = T.Compose(train_transform_steps())
eval_transform = T.Compose(eval_transform_steps())


def denormalize(tensor, mean=MEAN, std=STD):
    """Undo Normalize for plotting: CxHxW tensor -> HxWxC numpy array clipped to [0, 1]."""
    import torch

    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (tensor * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


# Table for the report / README: augmentation | parameters | why it does not change the diagnosis
AUGMENTATION_TABLE = [
    ("RandomResizedCrop", "224, scale=(0.8, 1.0)", "keeps ≥ 80 % of the image, the lesion stays in view; mimics distance/zoom differences"),
    ("RandomHorizontalFlip", "p=0.5", "lesions have no left/right orientation in the image"),
    ("RandomVerticalFlip", "p=0.5", "lesions have no up/down orientation in the image"),
    ("RandomRotation", "±180°", "the camera can be held at any angle; shape and pattern are rotation-invariant"),
    ("ColorJitter", "brightness=0.1, contrast=0.1", "mimics lighting/camera differences; the classes barely differ in brightness (A3.5)"),
    ("ColorJitter", "hue=0.02 (≈ ±7°, avg 3°)", "smaller than the ≈ 9° Benign/Malignant hue gap measured in A3.5; stronger hue shifts are NOT used"),
]

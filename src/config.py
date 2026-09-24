"""Central settings for the project.

Every script and notebook imports paths from here, so no absolute path is ever typed
anywhere else. The data folder is read from the environment variable MILK10K_DIR.
"""
import os
from pathlib import Path


def _data_dir() -> Path:
    value = os.environ.get("MILK10K_DIR")
    if not value:
        raise RuntimeError(
            "Environment variable MILK10K_DIR is not set.\n"
            "Point it at the MILK10k folder, e.g.  export MILK10K_DIR=~/data/milk10k"
        )
    path = Path(value).expanduser()
    if not path.is_dir():
        raise RuntimeError(f"MILK10K_DIR={value!r} does not exist or is not a folder.")
    return path


def _first_existing(*candidates: Path) -> Path:
    """Return the first path that exists (the download puts training_gt.csv in supplements/)."""
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


DATA_DIR = _data_dir()
IMG_DIR = DATA_DIR / "images"
METADATA_CSV = DATA_DIR / "metadata.csv"
GT_CSV = _first_existing(DATA_DIR / "training_gt.csv", DATA_DIR / "supplements" / "training_gt.csv")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
SPLIT_DIR = OUTPUT_DIR / "splits"

SEED = 42
IMG_SIZE = 224

# The 11 diagnostic classes (column order of training_gt.csv)
CLASSES = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
DX1_CLASSES = ["Benign", "Indeterminate", "Malignant"]

# Exact image_type strings in metadata.csv
DERM = "dermoscopic"
CLINICAL = "clinical: close-up"

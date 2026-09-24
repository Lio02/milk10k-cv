"""Lesion-level train / val / test splitting.

The unit of splitting is the lesion: both images of a lesion always land in the same split,
so a model can never see one view of a lesion in training and the other in evaluation.
"""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def _take_one_fold(df: pd.DataFrame, n_splits: int, y_col: str, seed: int):
    """Return (rest_index, fold_index) for the first fold of a StratifiedGroupKFold."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_idx, fold_idx = next(sgkf.split(df, y=df[y_col], groups=df["lesion_id"]))
    return rest_idx, fold_idx


def split_lesions(lesions: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15,
                  seed: int = 42, y_col: str = "dx"):
    """Split lesions into train / val / test, stratified on `y_col` and grouped by lesion_id.

    Step 1: StratifiedGroupKFold with n_splits = round(1 / test_size); one fold is the test set.
    Step 2: the same on the remaining lesions with n_splits = round((1 - test_size) / val_size);
            one fold is the validation set, the rest is train.

    Works on the lesion table (one row per lesion) or the image table (rows grouped by lesion_id).
    Returns three sorted lists of lesion_id: (train_ids, val_ids, test_ids).
    """
    df = lesions.reset_index(drop=True)

    rest_idx, test_idx = _take_one_fold(df, round(1 / test_size), y_col, seed)
    rest = df.iloc[rest_idx].reset_index(drop=True)

    train_idx, val_idx = _take_one_fold(rest, round((1 - test_size) / val_size), y_col, seed)

    test_ids = sorted(df.iloc[test_idx].lesion_id.unique())
    val_ids = sorted(rest.iloc[val_idx].lesion_id.unique())
    train_ids = sorted(rest.iloc[train_idx].lesion_id.unique())
    return train_ids, val_ids, test_ids

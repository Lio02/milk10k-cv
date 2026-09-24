"""Tests for src.dataset (run with: pytest tests/)."""
import numpy as np
import pandas as pd
import pytest

from src.data import build_lesion_table, load_gt, load_metadata
from src.dataset import LesionDataset, MilkImageDataset, aggregate_predictions, make_sampler

LABEL_MAP = {"Benign": 0, "Indeterminate": 1, "Malignant": 2}


def test_aggregate_predictions_known_answer():
    table = pd.DataFrame({"lesion_id": ["A", "A", "B", "B"]})
    probs = np.array([[0.9, 0.05, 0.05], [0.5, 0.1, 0.4], [0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
    out = aggregate_predictions(probs, table)
    assert out.pred.to_dict() == {"A": 0, "B": 2}
    assert np.allclose(out.loc["B", [0, 1, 2]].values, [0.2, 0.3, 0.5])


def test_missing_image_raises():
    lesions = build_lesion_table(load_metadata(), load_gt()).head(2).copy()
    lesions.loc[0, "derm_id"] = "ISIC_missing"
    with pytest.raises(FileNotFoundError):
        LesionDataset(lesions, LABEL_MAP)[0]
    images = pd.DataFrame({"isic_id": ["ISIC_missing"], "diagnosis_1": ["Benign"]})
    with pytest.raises(FileNotFoundError):
        MilkImageDataset(images, LABEL_MAP)


def test_sampler_balances_classes():
    labels = np.array([0] * 900 + [1] * 90 + [2] * 10)
    drawn = np.array(list(make_sampler(labels, seed=0)))
    shares = np.bincount(labels[drawn], minlength=3) / len(drawn)
    assert np.all(np.abs(shares - 1 / 3) < 0.06)

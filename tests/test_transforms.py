"""Tests for src.transforms (run with: pytest tests/)."""
import torch
from PIL import Image

from src import config
from src.data import image_path, load_metadata
from src.transforms import eval_transform, train_transform


def _img():
    return Image.open(image_path(load_metadata().isic_id.iloc[0])).convert("RGB")


def test_eval_transform_is_deterministic():
    img = _img()
    assert torch.equal(eval_transform(img), eval_transform(img))


def test_output_shapes():
    img = _img()
    for tf in (train_transform, eval_transform):
        out = tf(img)
        assert out.shape == (3, config.IMG_SIZE, config.IMG_SIZE)
        assert out.dtype == torch.float32


def test_train_transform_is_random():
    img = _img()
    torch.manual_seed(0)
    a = train_transform(img)
    b = train_transform(img)
    assert not torch.equal(a, b)

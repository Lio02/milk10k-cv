"""Tests for src.splits.split_lesions (run with: pytest tests/)."""
from functools import lru_cache

import pytest

from src.data import build_lesion_table, load_gt, load_metadata
from src.splits import split_lesions

SEEDS = range(10)
VAL, TEST = 0.15, 0.15


@pytest.fixture(scope="module")
def lesions():
    return build_lesion_table(load_metadata(), load_gt())


@lru_cache(maxsize=None)
def cached_split(seed):
    lesions = build_lesion_table(load_metadata(), load_gt())
    return split_lesions(lesions, VAL, TEST, seed)


@pytest.mark.parametrize("seed", SEEDS)
def test_no_overlap_and_complete(lesions, seed):
    train, val, test = map(set, cached_split(seed))
    assert not train & val
    assert not train & test
    assert not val & test
    assert train | val | test == set(lesions.lesion_id)


@pytest.mark.parametrize("seed", SEEDS)
def test_sizes_within_one_point(lesions, seed):
    n = len(lesions)
    _, val, test = cached_split(seed)
    # StratifiedGroupKFold uses a whole number of folds: 1/7 = 14.3 % for a 15 % request
    assert abs(len(test) / n - 1 / round(1 / TEST)) <= 0.01
    assert abs(len(test) / n - TEST) <= 0.01
    assert abs(len(val) / n - VAL) <= 0.01


@pytest.mark.parametrize("seed", SEEDS)
def test_same_seed_same_split(lesions, seed):
    assert split_lesions(lesions, VAL, TEST, seed) == cached_split(seed)


def test_different_seeds_differ(lesions):
    assert cached_split(0)[2] != cached_split(1)[2]

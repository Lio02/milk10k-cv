"""Small plotting helpers shared by the notebook and the scripts."""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def show_grid(images, titles=None, ncols=4, size=2.5):
    """Draw a grid of images with optional titles and return the figure.

    images: list of PIL images, HxWx3 uint8 arrays, float arrays in [0, 1],
            or None (leaves an empty slot).
    """
    n = len(images)
    nrows = max(1, math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= n or images[i] is None:
            continue
        img = np.asarray(images[i])
        ax.imshow(img)
        if titles is not None:
            ax.set_title(str(titles[i]), fontsize=9)
    fig.tight_layout()
    return fig


def plot_class_balance(series, log=False, title=None, ax=None):
    """Bar chart of class counts. `series` is a column of labels or a value_counts() result."""
    already_counts = pd.api.types.is_numeric_dtype(series) and not isinstance(series.index, pd.RangeIndex)
    counts = series if already_counts else series.value_counts()
    counts = counts.sort_values(ascending=False)
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, 0.6 * len(counts) + 2), 3.5))
    else:
        fig = ax.figure
    bars = ax.bar(counts.index.astype(str), counts.values, color="#4C72B0")
    ax.bar_label(bars, fontsize=8)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel("count" + (" (log scale)" if log else ""))
    ax.set_title(title or "Class balance")
    ax.tick_params(axis="x", rotation=45)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig

"""B2 - Label-centred EDA on the full dataset.

Usage:  python scripts/eda.py
Writes: outputs/figures/eda_diagnosis_1.png, eda_11_classes_log.png, eda_gallery.png
        outputs/class_to_diagnosis_1.csv, outputs/eda_findings.md
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

from src import config
from src.data import add_labels, build_lesion_table, image_path, load_gt, load_metadata
from src.viz import plot_class_balance, show_grid


def main():
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    meta = add_labels(load_metadata(), load_gt())
    lesions = build_lesion_table(meta)

    # 1. Class balance (per lesion)
    fig = plot_class_balance(lesions.diagnosis_1, title="diagnosis_1 — lesions")
    fig.savefig(config.FIG_DIR / "eda_diagnosis_1.png", dpi=120, bbox_inches="tight")
    fig = plot_class_balance(lesions.dx, log=True, title="11 classes — lesions (log scale)")
    fig.savefig(config.FIG_DIR / "eda_11_classes_log.png", dpi=120, bbox_inches="tight")
    plt.close("all")

    table = pd.crosstab(lesions.dx, lesions.diagnosis_1, margins=True)
    table.to_csv(config.OUTPUT_DIR / "class_to_diagnosis_1.csv")
    print(table, "\n")

    # 2. Re-check the Session 2 findings on the full data
    age = lesions.groupby("diagnosis_1").age_approx.median()
    bm = meta[meta.diagnosis_1.isin(["Benign", "Malignant"])].copy()
    bm["size"] = [os.stat(image_path(i)).st_size for i in bm.isic_id]
    auc = {t: roc_auc_score(g.diagnosis_1 == "Malignant", g["size"]) for t, g in bm.groupby("image_type")}
    manip = pd.crosstab(meta.image_manipulation, meta.diagnosis_1, normalize="index")
    altered_by_dx = meta.groupby("dx").image_manipulation.apply(lambda s: (s == "altered").mean())
    site_missing = lesions.anatom_site_general.isna().mean()
    mal_by_sex = pd.crosstab(lesions.sex, lesions.diagnosis_1, normalize="index")["Malignant"]

    # mean RGB of 150 dermoscopic images per class (downsized for speed)
    derm = meta[meta.image_type == config.DERM]
    mean_rgb = {}
    for c in ["Benign", "Malignant"]:
        ids = derm[derm.diagnosis_1 == c].isic_id.sample(150, random_state=config.SEED)
        px = np.stack([np.asarray(Image.open(image_path(i)).convert("RGB").resize((64, 48))).reshape(-1, 3).mean(0) for i in ids])
        mean_rgb[c] = px.mean(0)
    rg_ratio = {c: v[0] / v[1] for c, v in mean_rgb.items()}

    findings = pd.DataFrame([
        ["Malignant lesions come from older patients",
         f"yes — median age Malignant {age['Malignant']:.0f}, Indeterminate {age['Indeterminate']:.0f}, Benign {age['Benign']:.0f}",
         "age is a real signal, but the model must not rely on it alone (images only as input for now)"],
        ["File size differs by class (A2.2c)",
         f"no — AUC dermoscopic {auc[config.DERM]:.2f}, clinical {auc[config.CLINICAL]:.2f} (all images 600×450)",
         "no measurable shortcut in this copy, but still resize all images and never use file size"],
        ["Colour differs by class",
         f"yes, slightly — red/green ratio Benign {rg_ratio['Benign']:.3f} vs Malignant {rg_ratio['Malignant']:.3f}",
         "keep hue augmentation mild (hue ≤ 0.02, see A3.5)"],
        ["image_manipulation is linked to class",
         f"yes — 'altered' share ranges from {altered_by_dx.min():.1%} to {altered_by_dx.max():.1%} by class; "
         f"Malignant share {manip.loc['altered', 'Malignant']:.0%} (altered) vs {manip.loc['instrument only', 'Malignant']:.0%} (instrument only)",
         "do not use image_manipulation as an input"],
        ["Anatomical site often missing",
         f"yes — {site_missing:.0%} of lesions; class mix differs when missing (more NV)",
         "treat 'missing' as its own category if site is ever used; never drop these lesions"],
        ["Men have a higher malignant share",
         f"yes — {mal_by_sex['male']:.0%} (male) vs {mal_by_sex['female']:.0%} (female)",
         "check performance per sex later (subgroup bias)"],
    ], columns=["Session 2 finding", "Still true on the full data?", "Consequence"])
    md = findings.to_markdown(index=False)
    (config.OUTPUT_DIR / "eda_findings.md").write_text("# EDA findings (B2)\n\n" + md + "\n")
    print(findings.to_string(), "\n")

    # 3. Gallery: one dermoscopic image per class (11 classes + 1 empty slot)
    counts = lesions.dx.value_counts()
    picks = lesions.groupby("dx").sample(1, random_state=config.SEED).set_index("dx").loc[counts.index]
    imgs = [Image.open(image_path(i)).convert("RGB") for i in picks.derm_id] + [None]
    titles = [f"{dx} ({picks.loc[dx, 'diagnosis_1']}, n={counts[dx]})" for dx in picks.index] + [""]
    fig = show_grid(imgs, titles, ncols=4, size=3)
    fig.savefig(config.FIG_DIR / "eda_gallery.png", dpi=110, bbox_inches="tight")
    print("figures saved to outputs/figures/")


if __name__ == "__main__":
    main()

"""B4 - Data-quality report.

Usage:  python scripts/quality_report.py
Writes: outputs/quality_report.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config
from src.data import add_labels, load_gt, load_metadata

# Decision for every column that has missing values
MISSING_DECISIONS = {
    "age_approx": "impute the median age of the TRAIN split (if age is ever used as an input)",
    "anatom_site_general": 'fill with "unknown" — missingness is informative (more NV), so keep it as its own category',
    "anatom_site_special": "not used (> 95 % missing)",
    "diagnosis_3": "not used as input (label information); missing only means no finer sub-type was given",
    "diagnosis_4": "not used as input (label information)",
    "melanocytic": "not used as input (derived from the diagnosis)",
    "sex": 'fill with "unknown"',
}

# Columns never given to the model because they leak the label or are pure bookkeeping
LEAKY_COLUMNS = {
    "diagnosis_1": "the target itself",
    "diagnosis_2": "coarser/finer version of the diagnosis",
    "diagnosis_3": "coarser/finer version of the diagnosis",
    "diagnosis_4": "coarser/finer version of the diagnosis",
    "melanocytic": "derived from the diagnosis",
    "diagnosis_confirm_type": "how the diagnosis was made (biopsy ⇒ suspicious ⇒ mostly malignant)",
    "concomitant_biopsy": "same information as diagnosis_confirm_type",
    "image_manipulation": "linked to class (see crosstab) — an acquisition artefact, not biology",
    "attribution / copyright_license": "bookkeeping, constant",
    "isic_id / lesion_id": "identifiers",
}


def pct_table(a, b):
    return (pd.crosstab(a, b, normalize="index") * 100).round(1)


def main():
    meta = add_labels(load_metadata(), load_gt())
    lines = ["# Data-quality report (B4)", ""]

    # 1. Missing values
    missing = meta.isna().sum()
    missing = missing[missing > 0]
    lines += ["## 1. Missing values and decisions", "", "| column | missing images | % | decision |", "|---|---|---|---|"]
    for col, n in missing.items():
        lines.append(f"| {col} | {n:,} | {n / len(meta):.1%} | {MISSING_DECISIONS.get(col, 'not used')} |")

    # 2. Structural checks
    per_lesion = meta.groupby("lesion_id").image_type.agg(["size", "nunique"])
    two_images = (per_lesion["size"] == 2).all()
    one_of_each = (per_lesion["nunique"] == 2).all()
    types = meta.groupby("lesion_id").image_type.apply(frozenset).value_counts()
    assert two_images, "some lesions do not have exactly 2 images"
    assert one_of_each, "some lesions do not have one image of each type"
    assert types.index[0] == frozenset({config.DERM, config.CLINICAL})
    lines += ["", "## 2. Structure checks", "",
              f"- every lesion has exactly 2 images: **{two_images}** ({meta.lesion_id.nunique():,} lesions, {len(meta):,} images)",
              f"- every lesion has one `{config.DERM}` and one `{config.CLINICAL}` image: **{one_of_each}**",
              f"- `isic_id` unique: **{meta.isic_id.is_unique}**"]

    # 3. Acquisition columns vs label
    manip = pct_table(meta.image_manipulation, meta.diagnosis_1)
    itype = pct_table(meta.image_type, meta.diagnosis_1)
    manip_by_type = pd.crosstab(meta.image_type, meta.image_manipulation)
    lines += ["", "## 3. Acquisition columns vs diagnosis_1 (row %)", "",
              "### image_manipulation", "", manip.to_markdown(), "",
              "### image_type", "", itype.to_markdown(), "",
              "### image_manipulation by image_type (counts)", "", manip_by_type.to_markdown(), "",
              f"**Conclusion:** `image_type` carries no label information (each lesion has one of each, so the class mix is identical). "
              f"`image_manipulation` does: 'altered' images are malignant only {manip.loc['altered', 'Malignant']:.0f} % of the time vs "
              f"{manip.loc['instrument only', 'Malignant']:.0f} % for 'instrument only', and almost all altered images are clinical photos. "
              f"It must not be used as an input."]

    # 4. Leaky columns
    lines += ["", "## 4. Columns NOT used as inputs", "", "| column | reason |", "|---|---|"]
    lines += [f"| {c} | {r} |" for c, r in LEAKY_COLUMNS.items()]
    lines += ["", "Inputs for the model: the two images only. Age, sex and site may be added later as auxiliary inputs."]

    out = config.OUTPUT_DIR / "quality_report.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nsaved {out.relative_to(config.REPO_ROOT)}")


if __name__ == "__main__":
    main()

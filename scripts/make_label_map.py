"""B3 - Label strategy: write outputs/label_map.json and print the counts behind the decision.

Usage:  python scripts/make_label_map.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data import build_lesion_table, load_gt, load_metadata

RARE_THRESHOLD = 50  # lesions


def main():
    lesions = build_lesion_table(load_metadata(), load_gt())

    print("diagnosis_1 (lesions):")
    print(lesions.diagnosis_1.value_counts().to_string(), "\n")
    counts = lesions.dx.value_counts()
    print("11 classes (lesions):")
    print(counts.to_string())
    print(f"\nclasses under {RARE_THRESHOLD} lesions:", list(counts[counts < RARE_THRESHOLD].index))

    # Exact label strings copied from the data, in a fixed (alphabetical) order
    label_map = {
        "diagnosis_1": {c: i for i, c in enumerate(sorted(lesions.diagnosis_1.unique()))},
        "dx": {c: i for i, c in enumerate(config.CLASSES)},
    }
    assert set(label_map["diagnosis_1"]) == set(config.DX1_CLASSES)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    with open(config.OUTPUT_DIR / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    print("\nsaved outputs/label_map.json:", label_map)


if __name__ == "__main__":
    main()

# Milestone 1 — MILK10k data pipeline

All numbers come from the scripts in `scripts/` (seed 42). Figures are in `outputs/figures/`.

## 1. Label strategy (B3)

**Primary target: `diagnosis_1` with 3 classes** — Benign 1,483, Malignant 3,634, **Indeterminate 123** lesions (2.3 %).
We keep Indeterminate as its own class instead of merging it into Benign. Clinically these lesions (mostly AKIEC) still need treatment or follow-up, so calling them "benign" would hide risk. 123 lesions (89 in train) is little but enough to learn with class weighting or oversampling. Note that AKIEC maps to *both* Indeterminate (123) and Malignant (180) (A1.1c), so `diagnosis_1` cannot be derived from an 11-class prediction; we train on it directly.

**11-class stretch goal.** Three classes have fewer than 50 lesions: VASC 47, BEN_OTH 44, MAL_OTH 9 (DF 52 and INF 50 are borderline).
- *Keep them with class weights*: this preserves clinically distinct diagnoses, but test metrics for them rest on 1–8 lesions and are very noisy.
- *Merge them into "OTHER"*: this is more stable, but it mixes a malignant class (MAL_OTH) with benign ones, which is clinically meaningless.

We **keep all 11 classes with class weights** and report per-class results with their counts (and ideally pooled over cross-validation folds). We only merge if training proves unstable, and then only within the same `diagnosis_1` group. Label indices: `outputs/label_map.json`.

## 2. Split design and verification (B5)

Split **by lesion** with `split_lesions()`, a two-step `StratifiedGroupKFold` stratified on the 11-class `dx` (`src/splits.py`). Then both images of each lesion are attached to that lesion's split.

| split | lesions | share | images |
|---|---|---|---|
| train | 3,742 | 71.4 % | 7,484 |
| val | 749 | 14.3 % | 1,498 |
| test | 749 | 14.3 % | 1,498 |

(15 % is not a whole number of folds, so `round(1/0.15) = 7` folds gives 14.3 %.)

Verification:
- zero lesion or image overlap between any two splits
- every lesion has exactly two images (one per type) in its split
- **maximum deviation of any 11-class proportion from the global proportion: 0.10 percentage points** (MAL_OTH: 6 / 1 / 2 lesions)
- `diagnosis_1` shares within 0.2 pp across splits

`tests/test_splits.py` checks overlap, sizes and reproducibility for 10 seeds. A3.1 showed why grouping matters: in a naive image split, 80 % of test images have their sibling in train, and a "sibling oracle" reaches 0.85 balanced accuracy without learning anything.

## 3. Imbalance on TRAIN

`diagnosis_1`: largest/smallest = Malignant 2,595 / Indeterminate 89 = **29 : 1** (lesions). 11 classes: BCC 1,802 / MAL_OTH 6 = **300 : 1**.
Handling (`src/dataset.py`), computed on train only:
- (i) class weights `n / (k · count)` in `outputs/class_weights.json` (Benign 1.18, Indeterminate 14.0, Malignant 0.48)
- (ii) a `WeightedRandomSampler` (weight 1/count), which gives about 1/3 of each class per batch (checked over 20 batches)

We use **one of the two**, not both at once: the sampler is the default and the weighted loss is the alternative.

## 4. Quality issues (B1, B4)

- All 10,480 files are present and decodable (0 unreadable). All are RGB JPEGs of **600 × 450**, so this copy was already resized by the course.
- Structure is clean: every lesion has exactly one dermoscopic and one clinical close-up image, IDs are unique, and the lesion-level fields agree between the two images.
- Missing values:
  - `anatom_site_general` 37 % → kept as "unknown", because missingness is informative (more NV)
  - `age_approx` 0.4 % → train median
  - `anatom_site_special` 98 % → dropped
- **Leaky columns excluded as inputs:** `diagnosis_2–4`, `melanocytic`, `diagnosis_confirm_type`, `concomitant_biopsy` and `image_manipulation`. The last one is linked to the class ("altered" images are 48 % malignant vs 70 % otherwise, and are almost all clinical photos).
- File size shows **no** shortcut in this copy (AUC 0.48 / 0.51). We still never use it.

## 5. Preprocessing (B6, B7)

- **Resolution:** 100 % of images have a short side ≥ 256 px (all are 450 px), so `Resize(256) → CenterCrop(224)` only downsamples and never upsamples. We use **224**, the standard input size for pretrained CNNs; the 4:3 → 1:1 crop keeps the centred lesion.
- **Views:** option (c), **both views per lesion** (`LesionDataset`). (a) Independent images with the lesion label give twice the samples but must be averaged per lesion at evaluation (`aggregate_predictions`); `MilkImageDataset` supports this for a first baseline. (b) Dermoscopic only is the simplest but loses clinical context. Per-image evaluation counts each lesion twice and inflates confidence.
- **Normalisation (train only):** mean = (0.686, 0.523, 0.473), std = (0.124, 0.131, 0.151) → `outputs/norm_stats.json`.
- **Augmentation** (train only; eval = Resize + CenterCrop + Normalize, deterministic and tested):

| augmentation | parameters | why it does not change the diagnosis |
|---|---|---|
| RandomResizedCrop | 224, scale 0.8–1.0 | keeps ≥ 80 % of the image; mimics distance/zoom |
| Horizontal / vertical flip | p = 0.5 each | a lesion has no anatomical orientation in the image |
| RandomRotate90 | 0/90/180/270° | any camera angle; unlike free rotation, no black corners that only training images would have |
| ColorJitter | brightness 0.1, contrast 0.1 | lighting/camera variation; the classes barely differ in brightness (A3.5) |
| ColorJitter | hue 0.02 | shifts hue by about 3°, below the 9° Benign/Malignant gap (A3.5); stronger hue is not used |

## 6. Biopsy enrichment (B2)

MILK10k is not a sample of the lesions a GP sees. 96 % of lesions were confirmed by histopathology, which means someone found them suspicious enough to biopsy, and 72 % of those turned out malignant. As a result the dataset is **69 % malignant**, and BCC alone is 48 % of all lesions. In primary care the large majority of lesions are benign (nevi, seborrhoeic keratoses) and are never biopsied. The few non-biopsied lesions here, confirmed only by clinical assessment, are 87 % benign.

This has two consequences:
- **Accuracy and precision measured here will not transfer.** Precision (PPV) depends on prevalence: with the same sensitivity and specificity, a model that looks precise on a 69 %-malignant test set will produce mostly false alarms where malignancy is rare. Even a constant "Malignant" prediction reaches 69 % accuracy here (A3.4).
- **The model learns to separate *suspicious* lesions from each other**, not "any lesion" from cancer. Obviously benign lesions that never get biopsied are under-represented, so its behaviour on them is untested.

We therefore report prevalence-independent metrics (balanced accuracy, per-class recall, ROC/PR per class). A deployment would need recalibration to the real prevalence and a validation set from the target setting.

## 7. What could still go wrong

- **Near-duplicates across lesions:** the same patient can have several lesions, and there is no patient ID, so visually similar skin or lesions from one person could still end up in both train and test.
- **Acquisition shortcuts:** device, clinic, image manipulation, rulers, pen marks or dermoscope vignetting can correlate with class.
- **Bias:** skin tone is unevenly represented (61 % of lesions fall in one `skin_tone_class` in `training_input.csv`), and site and age distributions differ by class, so performance must be checked per subgroup.
- **Too few rare-class cases:** MAL_OTH has 2 test lesions, and INF, VASC, BEN_OTH and DF have about 7 each, so their metrics are anecdotal.
- **Clinical use:** the model is **decision support, not a diagnosis**. A clinician keeps the final decision, especially for lesions predicted Benign.

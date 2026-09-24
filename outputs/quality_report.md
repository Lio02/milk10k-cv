# Data-quality report (B4)

## 1. Missing values and decisions

| column | missing images | % | decision |
|---|---|---|---|
| age_approx | 40 | 0.4% | impute the median age of the TRAIN split (if age is ever used as an input) |
| anatom_site_general | 3,912 | 37.3% | fill with "unknown" — missingness is informative (more NV), so keep it as its own category |
| anatom_site_special | 10,274 | 98.0% | not used (> 95 % missing) |
| diagnosis_3 | 158 | 1.5% | not used as input (label information); missing only means no finer sub-type was given |
| diagnosis_4 | 8,958 | 85.5% | not used as input (label information) |
| melanocytic | 8,088 | 77.2% | not used as input (derived from the diagnosis) |

## 2. Structure checks

- every lesion has exactly 2 images: **True** (5,240 lesions, 10,480 images)
- every lesion has one `dermoscopic` and one `clinical: close-up` image: **True**
- `isic_id` unique: **True**

## 3. Acquisition columns vs diagnosis_1 (row %)

### image_manipulation

| image_manipulation   |   Benign |   Indeterminate |   Malignant |
|:---------------------|---------:|----------------:|------------:|
| altered              |     39.7 |            12.8 |        47.5 |
| instrument only      |     27.9 |             2   |        70.1 |

### image_type

| image_type         |   Benign |   Indeterminate |   Malignant |
|:-------------------|---------:|----------------:|------------:|
| clinical: close-up |     28.3 |             2.3 |        69.4 |
| dermoscopic        |     28.3 |             2.3 |        69.4 |

### image_manipulation by image_type (counts)

| image_type         |   altered |   instrument only |
|:-------------------|----------:|------------------:|
| clinical: close-up |       327 |              4913 |
| dermoscopic        |         8 |              5232 |

**Conclusion:** `image_type` carries no label information (each lesion has one of each, so the class mix is identical). `image_manipulation` does: 'altered' images are malignant only 48 % of the time vs 70 % for 'instrument only', and almost all altered images are clinical photos. It must not be used as an input.

## 4. Columns NOT used as inputs

| column | reason |
|---|---|
| diagnosis_1 | the target itself |
| diagnosis_2 | coarser/finer version of the diagnosis |
| diagnosis_3 | coarser/finer version of the diagnosis |
| diagnosis_4 | coarser/finer version of the diagnosis |
| melanocytic | derived from the diagnosis |
| diagnosis_confirm_type | how the diagnosis was made (biopsy ⇒ suspicious ⇒ mostly malignant) |
| concomitant_biopsy | same information as diagnosis_confirm_type |
| image_manipulation | linked to class (see crosstab) — an acquisition artefact, not biology |
| attribution / copyright_license | bookkeeping, constant |
| isic_id / lesion_id | identifiers |

Inputs for the model: the two images only. Age, sex and site may be added later as auxiliary inputs.

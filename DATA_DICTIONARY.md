# Dataset Documentation — Mango Ripeness & Quality Assessment

**Assignment option:** C, a custom tabular dataset.
**Raw file:** `data/mango_quality_raw.csv` (338 rows × 14 columns: 330 unique mangoes plus 8 double-entered rows).
**Generator:** `data/generate_dataset.py` (seed 2025). Run `python data/generate_dataset.py` to regenerate the same file.
**Feature file (output of the notebook):** `data/mango_features.csv` (330 rows: `sample_id`, `split`, 9 selected features, `ripeness_class`).
**All engineered candidates:** `data/mango_features_all_candidates.csv` (21 candidate features).

## 1. Purpose
This dataset simulates a post-harvest grading session for Pakistani mangoes. Graders need to decide whether a fruit is
**Unripe**, **Ripe** or **Overripe** from measurements that are quick to take. The dataset supports:
- supervised classification (KNN), and
- unsupervised discovery of groups (K-Means, Agglomerative).

## 2. Collection / generation procedure
The assignment allows "generating meaningful custom data". The data come from a generative model that I designed myself
from post-harvest physiology. No public dataset and no internet data were used.

1. **Hidden ripening stage** `r ~ Beta(1.1, 1.1)` on [0, 1]: 0 means freshly harvested and hard green; 1 means fully overripe.
2. **Variety** is drawn with probabilities Sindhri 0.36, Chaunsa 0.38 and Anwar Ratol 0.26. Each variety has its own mean size, sugar offset, peel-colour offset and acidity multiplier.
3. **Storage temperature** `~ N(29, 3.5)` °C, clipped to 18–38 °C. The ripening rate is proportional to (T − 10 °C). **Days after harvest** = r / rate + N(0, 1.2).
4. **Measurements**, each one a function of r plus a variety effect plus instrument noise:

| Attribute | Model | Noise σ |
|---|---|---|
| firmness_n | 62·e^(−3.1r) + 2.5 | 3.2 N |
| tss_brix | 7.5 + 13 / (1 + e^(−9(r − 0.45))) + variety offset | 0.9 °Brix |
| acidity_pct | variety mult · (1.55·e^(−2.6r) + 0.12) | 0.07 % |
| peel_hue_deg | 112 − 52·r^0.9 + variety offset | 4.5° |
| aroma_score | 1 + 8.2·r^1.3, clipped to 1–10 | 0.8 |
| dark_spots | Poisson(250·max(r − 0.55, 0)^1.5) | Poisson |
| weight_g | variety mean · size factor · (1 − 0.07r) | 12 g |
| length_mm, width_mm | N(variety mean, 7 / 5 mm) | — |

5. **Ground-truth label** (what a grader would assign): r_obs = r + Logistic(0, 0.025).
   The label is Unripe if r_obs < 0.38, Ripe if 0.38 ≤ r_obs < 0.70, and Overripe otherwise.
   The logistic term models how unsure a grader is near a boundary.
6. **Collection problems added on purpose**, so that cleaning is needed:
   - Missing values: `firmness_n` 9, `tss_brix` 10, `acidity_pct` 13, `aroma_score` 6.
   - Six data-entry errors: MG018 weight entered in kg, MG089 firmness with an extra digit, MG144 Brix = 48, MG202 hue = 290°, MG257 length with a decimal slip, MG302 acidity in the wrong unit.
   - Eight duplicated rows, as if the same sheet was typed in twice.

## 3. Data dictionary

| Attribute | Type | Unit / values | Instrument | Description |
|---|---|---|---|---|
| sample_id | ID | MG001–MG330 | label | Unique fruit ID (not a feature) |
| variety | categorical | Sindhri / Chaunsa / Anwar Ratol | visual | Cultivar |
| storage_temp_c | float | °C | thermometer | Mean storage temperature |
| days_after_harvest | float | days | log book | Time since harvest |
| weight_g | float | g | digital scale | Fruit mass |
| length_mm | float | mm | vernier caliper | Longest axis |
| width_mm | float | mm | vernier caliper | Widest diameter |
| firmness_n | float | N | penetrometer (8 mm probe) | Puncture force |
| tss_brix | float | °Brix | refractometer | Total soluble solids (sugar) |
| acidity_pct | float | % citric acid | titration | Titratable acidity |
| peel_hue_deg | float | degrees | colorimeter | Peel hue angle (≈110 green → ≈60 orange) |
| aroma_score | float | 1–10 | sensory panel | Aroma intensity |
| dark_spots | int | count | visual | Dark/brown spots on peel |
| ripeness_class | target | Unripe / Ripe / Overripe | grader | Ground-truth label |

### Engineered features (Phase II)
| Feature | Formula | Meaning |
|---|---|---|
| brix_acid_ratio | tss_brix / acidity_pct | Sugar–acid maturity index |
| sweet_soft_index | tss_brix / firmness_n | Sweetness per unit firmness |
| firmness_per_100g | firmness_n / (weight_g / 100) | Firmness adjusted for fruit size |
| thermal_time | days_after_harvest × (storage_temp_c − 10) | Degree-days of heat accumulated |
| volume_cm3 | π/6 · L · W² / 1000 | Volume, treating the fruit as an ellipsoid |
| density_g_cm3 | weight_g / volume_cm3 | Apparent density |
| log_dark_spots | ln(1 + dark_spots) | Spot count on a log scale to reduce skew |
| variety_* | one-hot | Cultivar indicator |

**Selected for modelling (9):** thermal_time, tss_brix, acidity_pct, firmness_n, peel_hue_deg, aroma_score,
log_dark_spots, days_after_harvest, firmness_per_100g.

## 4. Labels
| Class | Count (after removing duplicates) | Typical fruit |
|---|---|---|
| Unripe | 127 (38 %) | hard (~35 N), green (~96°), sour, ~10 °Brix |
| Ripe | 108 (33 %) | ~15 N, yellow (~80°), ~17.6 °Brix, aromatic |
| Overripe | 95 (29 %) | soft (~7 N), orange (~65°), ~21.6 °Brix, dark spots |

## 5. Assumptions
- The model's coefficients are realistic approximations taken from general mango physiology. They are not measurements from one specific orchard.
- The grader's label is treated as ground truth, even though it is noisy near the class boundaries.
- Values are missing completely at random.
- Readings outside physical limits are data-entry errors.
- No personal, confidential or sensitive information is collected.

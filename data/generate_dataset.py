"""
Custom dataset generator: Mango Ripeness & Quality Assessment (Option C - Custom Tabular Dataset).

The assignment allows "generating meaningful custom data". This script generates the raw dataset
from a documented, domain-driven simulation model so that every value is reproducible (fixed seed)
and traceable to a physical/biological assumption.

Generation model (summary - full details in DATA_DICTIONARY.md):
  1. Each mango gets a hidden continuous "ripening stage" r in [0, 1]
     (0 = just harvested / hard green, 1 = fully over-ripe).
  2. The ground-truth class is decided from r with grader uncertainty near the boundaries:
        r < 0.38 -> Unripe, 0.38 <= r < 0.70 -> Ripe, r >= 0.70 -> Overripe
     A logistic "grader noise" term makes labels near a boundary occasionally fall on the other side,
     just like a human grader would.
  3. Every measurable attribute is a function of r + variety effect + measurement noise, following
     known post-harvest physiology of mango:
        - firmness drops sharply as cell walls soften,
        - total soluble solids (Brix) rise as starch converts to sugar,
        - titratable acidity falls,
        - peel hue moves from green (~110 deg) to yellow/orange (~60 deg),
        - aroma volatiles increase,
        - weight falls slightly due to moisture loss.
  4. Realistic data-collection problems are injected on purpose so the EDA/preprocessing phases
     have real work to do: ~3% missing values in some instrument columns, 8 duplicated rows
     (double data entry) and 6 gross data-entry errors (outliers).

Run:  python data/generate_dataset.py      -> writes data/mango_quality_raw.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2025
N = 330
rng = np.random.default_rng(SEED)

VARIETIES = {
    # name: (mean weight g, Brix offset, hue offset deg, acidity multiplier, length mm, width mm)
    "Sindhri":     (420, 0.8, -3.0, 0.95, 128, 82),
    "Chaunsa":     (300, 1.5, -5.0, 0.85, 112, 76),
    "Anwar Ratol": (190, 2.2,  0.0, 1.10,  92, 64),
}

rows = []
for i in range(N):
    variety = rng.choice(list(VARIETIES), p=[0.36, 0.38, 0.26])
    w_mu, brix_off, hue_off, acid_mul, len_mu, wid_mu = VARIETIES[variety]

    # hidden ripening stage; Beta(1.1, 1.1) gives roughly uniform coverage of all stages
    r = rng.beta(1.1, 1.1)

    # storage conditions drive how many days it took to reach stage r
    storage_temp = float(np.clip(rng.normal(29, 3.5), 18, 38))          # deg C
    rate = 0.055 * (storage_temp - 10) / 19                              # stage per day
    days = float(np.clip(r / rate + rng.normal(0, 1.2), 0, 21))

    # grader label with boundary uncertainty (logistic noise)
    r_obs = r + rng.logistic(0, 0.025)
    label = "Unripe" if r_obs < 0.38 else ("Ripe" if r_obs < 0.70 else "Overripe")

    length = rng.normal(len_mu, 7)
    width = rng.normal(wid_mu, 5)
    size_factor = (length / len_mu) * (width / wid_mu) ** 2
    weight = w_mu * size_factor * (1 - 0.07 * r) + rng.normal(0, 12)     # moisture loss

    firmness = 62 * np.exp(-3.1 * r) + 2.5 + rng.normal(0, 3.2)          # Newton (penetrometer)
    brix = 7.5 + 13.0 / (1 + np.exp(-9 * (r - 0.45))) + brix_off + rng.normal(0, 0.9)
    acidity = acid_mul * (1.55 * np.exp(-2.6 * r) + 0.12) + rng.normal(0, 0.07)   # % citric acid
    hue = 112 - 52 * r ** 0.9 + hue_off + rng.normal(0, 4.5)             # degrees
    aroma = 1 + 8.2 * r ** 1.3 + rng.normal(0, 0.8)                      # 1-10 panel score
    spots = max(0, rng.poisson(25 * max(r - 0.55, 0) ** 1.5 * 10))       # dark spots count

    rows.append({
        "sample_id": f"MG{i + 1:03d}",
        "variety": variety,
        "storage_temp_c": round(storage_temp, 1),
        "days_after_harvest": round(days, 1),
        "weight_g": round(weight, 1),
        "length_mm": round(length, 1),
        "width_mm": round(width, 1),
        "firmness_n": round(max(firmness, 0.8), 2),
        "tss_brix": round(brix, 2),
        "acidity_pct": round(max(acidity, 0.05), 3),
        "peel_hue_deg": round(hue, 1),
        "aroma_score": round(float(np.clip(aroma, 1, 10)), 1),
        "dark_spots": int(spots),
        "ripeness_class": label,
    })

df = pd.DataFrame(rows)

# ---- inject realistic data-collection problems (documented in DATA_DICTIONARY.md) ----
# 1) missing values: instrument not available / reading not recorded
for col, frac in [("firmness_n", 0.03), ("tss_brix", 0.03), ("acidity_pct", 0.04), ("aroma_score", 0.02)]:
    idx = rng.choice(df.index, size=int(frac * len(df)), replace=False)
    df.loc[idx, col] = np.nan

# 2) gross data-entry errors (outliers)
df.loc[17, "weight_g"] = df.loc[17, "weight_g"] / 1000        # typed in kg instead of g
df.loc[88, "firmness_n"] = df.loc[88, "firmness_n"] * 10      # extra digit
df.loc[143, "tss_brix"] = 48.0                                # impossible Brix for mango
df.loc[201, "peel_hue_deg"] = 290.0                           # colorimeter mis-read
df.loc[256, "length_mm"] = 12.4                               # decimal slip (124 -> 12.4)
df.loc[301, "acidity_pct"] = 9.5                              # wrong unit

# 3) duplicate rows: same sheet entered twice
dups = df.sample(8, random_state=SEED)
df = pd.concat([df, dups]).sort_index(kind="stable").reset_index(drop=True)

out = Path(__file__).with_name("mango_quality_raw.csv")
df.to_csv(out, index=False)
print(f"Wrote {out} with shape {df.shape}")
print(df["ripeness_class"].value_counts())

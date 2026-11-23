# Advanced AI: Mango Ripeness Assignment

This repository covers custom dataset generation, EDA, feature engineering, KNN classification, K-Means clustering and agglomerative clustering.

**Dataset:** Option C, a custom tabular dataset. It describes the post-harvest ripeness of Pakistani mangoes (Sindhri, Chaunsa, Anwar Ratol). The target has three classes: **Unripe, Ripe, Overripe**.

## Deliverables

| # | Deliverable | File |
|---|---|---|
| 1 | Raw dataset | [`data/mango_quality_raw.csv`](data/mango_quality_raw.csv), with generator [`data/generate_dataset.py`](data/generate_dataset.py) |
| 2 | Dataset documentation | [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) and Section 1 of the notebook |
| 3 | EDA notebook | [`Assignment2_Mango_Ripeness.ipynb`](Assignment2_Mango_Ripeness.ipynb), Phase I |
| 4 | Preprocessing and feature extraction | Notebook Phase II, plus [`data/mango_features.csv`](data/mango_features.csv) (final features) and `data/mango_features_all_candidates.csv` |
| 5 | KNN implementation | Notebook Phase III |
| 6 | K-Means implementation | Notebook Phase IV |
| 7 | Agglomerative implementation | Notebook Phase V |
| 8 | Comparative report | Notebook Phase VI and [`report/Assignment2_Report.pdf`](report/Assignment2_Report.pdf) |

A step-by-step study guide with viva questions is in [`LEARNING_GUIDE.md`](LEARNING_GUIDE.md).

All figures are saved in [`figures/`](figures/). All numeric results are saved in [`results.json`](results.json).

## Key results

| | KNN (K=3, Manhattan) | K-Means (K=3) | Agglomerative (Ward, 3) |
|---|---|---|---|
| Labels used for fitting | Yes | No | No |
| Test accuracy / macro-F1 | 0.880 / 0.876 (CV 0.931) | n/a | n/a |
| Silhouette | n/a | 0.442 | 0.445 |
| ARI / NMI | n/a | 0.478 / 0.580 | 0.397 / 0.529 |

## Reproduce

```bash
pip install -r requirements.txt
python data/generate_dataset.py                 # regenerates the raw CSV (seed 2025)
jupyter nbconvert --to notebook --execute --inplace Assignment2_Mango_Ripeness.ipynb
python report/build_report.py                   # rebuilds the PDF report from results.json + figures
```

The notebook runs from top to bottom with no manual steps. It uses fixed random seeds. Every preprocessing step is fitted on the training data only.

"""Builds report/Assignment2_Report.pdf from results.json and figures/ (run after executing the notebook).
Usage (from repo root):  python report/build_report.py
"""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
                                KeepTogether)

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "results.json").read_text())
FIG = ROOT / "figures"
OUT = ROOT / "report" / "Assignment2_Report.pdf"

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#1f3b5c"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#1f3b5c"))
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=13, spaceAfter=4)
BL = ParagraphStyle("BL", parent=B, leftIndent=12, bulletIndent=2)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8.5, textColor=colors.grey, alignment=TA_CENTER, spaceAfter=8)
TC = ParagraphStyle("TC", parent=B, fontSize=8.2, leading=10.5, spaceAfter=0)
W = A4[0] - 3.6 * cm

story = []
p = lambda t, s=B: story.append(Paragraph(t, s))
bullets = lambda items: [story.append(Paragraph(i, BL, bulletText="•")) for i in items]


def fig(name, caption, width=W):
    path = FIG / f"{name}.png"
    img = Image(str(path)); ratio = img.imageHeight / img.imageWidth
    h = width * ratio
    if h > 11.5 * cm:  # keep tall figures on the page
        h = 11.5 * cm; width = h / ratio
    img.drawWidth, img.drawHeight = width, h
    story.append(KeepTogether([img, Paragraph(caption, CAP)]))


def table(rows, col_widths=None, header=True):
    data = [[Paragraph(str(c), TC) for c in r] for r in rows]
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0b7c3")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe7f2"))]
    t.setStyle(TableStyle(st)); story.append(t); story.append(Spacer(1, 6))


K, M, A = R["knn"], R["kmeans"], R["agg"]
f3 = lambda x: f"{x:.3f}"

# ------------------------------------------------------------------ title
story.append(Spacer(1, 1.2 * cm))
p("Advanced Artificial Intelligence – Integrated Practical Assignment", ParagraphStyle("T", parent=H1, fontSize=18, alignment=TA_CENTER))
p("Mango Ripeness &amp; Quality Assessment: Custom Dataset Generation, EDA, Feature Engineering, "
  "KNN Classification, K-Means and Agglomerative Clustering", ParagraphStyle("ST", parent=B, fontSize=11.5, alignment=TA_CENTER, leading=15))
story.append(Spacer(1, 0.5 * cm))
table([["Item", "Detail"],
       ["Assignment type", "Individual – 40 marks"],
       ["Dataset option", "Option C – Custom tabular dataset generated with my own documented domain model"],
       ["Domain", "Post-harvest ripeness grading of Pakistani mangoes (Sindhri, Chaunsa, Anwar Ratol)"],
       ["Size", f"{R['n_raw']} raw records → {R['n_clean']} unique mangoes; 12 raw inputs; 3 classes"],
       ["Deliverables", "Assignment2_Mango_Ripeness.ipynb, data/mango_quality_raw.csv, data/generate_dataset.py, "
                        "DATA_DICTIONARY.md, data/mango_features.csv, this report"]],
      [4 * cm, W - 4 * cm])
p("<b>Summary.</b> I generated a 3-class tabular dataset of mango measurements, explored and cleaned it, "
  f"engineered 7 physiological features and selected 9 final features. KNN (K = {K['K']}, {K['metric']}) reaches "
  f"<b>{K['test_acc']:.1%}</b> test accuracy (5-fold CV {K['cv_acc']:.1%}); all errors are between adjacent ripeness stages. "
  f"Without labels, K-Means (K = 3) and Ward agglomerative clustering recover an ordered ripening gradient "
  f"(ARI {M['ARI']:.2f} / {A['ARI']:.2f}, NMI {M['NMI']:.2f} / {A['NMI']:.2f}).")

# ------------------------------------------------------------------ 1 dataset
p("1. Dataset Collection and Documentation", H1)
p("<b>Research question.</b> Can easily measured physical and chemical properties tell a mango's ripeness stage, and do natural "
  "groups in the measurements match the stages a grader assigns? This matters to exporters, because shipping fruit at the wrong stage causes large losses.")
p("<b>Generation procedure.</b> The assignment allows generated custom data. I wrote a generative model "
  "(<i>data/generate_dataset.py</i>, seed 2025) that imitates a grading session:")
bullets([
    "Each mango has a hidden ripening stage r ∈ [0, 1] ~ Beta(1.1, 1.1). Its variety is drawn as Sindhri 36 %, Chaunsa 38 % or Anwar Ratol 26 %, and each variety has its own size, sugar, colour and acidity offsets.",
    "Storage temperature ~ N(29, 3.5) °C. Days after harvest = r / rate(T) + noise, because warmer storage ripens fruit faster.",
    "Each attribute follows a physiological curve in r plus instrument noise. Firmness decays exponentially (penetrometer). Brix rises along a sigmoid (refractometer). Acidity decays (titration). Peel hue moves from about 110° to about 60° (colorimeter). Aroma rises (sensory panel). Dark spots appear late in ripening. Weight drops slightly as moisture is lost.",
    "Label: Unripe if r &lt; 0.38, Ripe if 0.38 ≤ r &lt; 0.70, otherwise Overripe. A logistic grader-noise term is added to r first, so fruit near a boundary are sometimes labelled like the neighbouring stage.",
    "Collection problems were added on purpose: 38 missing instrument readings, 6 gross data-entry errors (kg typed instead of g, an extra digit, impossible Brix or hue) and 8 rows entered twice.",
])
table([["Attribute", "Unit", "Instrument", "Attribute", "Unit", "Instrument"],
       ["variety", "category", "visual", "firmness_n", "N", "penetrometer"],
       ["storage_temp_c", "°C", "thermometer", "tss_brix", "°Brix", "refractometer"],
       ["days_after_harvest", "days", "log book", "acidity_pct", "% citric", "titration"],
       ["weight_g", "g", "scale", "peel_hue_deg", "degrees", "colorimeter"],
       ["length_mm / width_mm", "mm", "caliper", "aroma_score", "1–10", "sensory panel"],
       ["ripeness_class (target)", "Unripe/Ripe/Overripe", "grader", "dark_spots", "count", "visual"]],
      [3.4 * cm, 2.6 * cm, 2.3 * cm, 2.7 * cm, 2.0 * cm, W - 13 * cm])
p("The full data dictionary, the model formulas and the assumptions are in <i>DATA_DICTIONARY.md</i> and in Section 1 of the notebook.")

# ------------------------------------------------------------------ 2 EDA
p("2. Phase I – Exploratory Data Analysis", H1)
table([["#", "Required analysis", "Finding"],
       ["1", "Shape, structure, types", f"{R['n_raw']} × 14 table: 11 numeric inputs, 1 categorical input, target and ID. All data types are correct. describe() already shows impossible minimum and maximum values."],
       ["2", "Missing values", f"{R['n_missing_raw']} cells (≈0.8 %), all in firmness, Brix, acidity and aroma. They are scattered and the rate is similar in every class (MCAR). Handled by median imputation fitted on the training split only."],
       ["3", "Duplicates", f"{R['n_dups']} exact duplicate rows (sheets typed twice) were removed, leaving {R['n_clean']} unique mangoes. If kept, a duplicate in the test set would leak its training copy to KNN."],
       ["4", "Outliers", f"IQR, z-score and domain-validity limits were combined. {R['n_invalid']} impossible readings were set to missing. 30 IQR flags on dark_spots are real overripe fruit and were kept; the feature was log-transformed instead."],
       ["5", "Distributions", "Firmness and acidity are right-skewed. Brix is bimodal because it follows a sigmoid. Size variables are multimodal because of variety. The ripeness attributes shift strongly by class."],
       ["6", "Correlation", "One block of strongly correlated ripeness attributes (firmness–acidity 0.94, Brix–aroma 0.89) and a separate size block (weight–width 0.93). Temperature is independent of both."],
       ["7", "Class balance", f"Unripe {R['class_counts']['Unripe']}, Ripe {R['class_counts']['Ripe']}, Overripe {R['class_counts']['Overripe']} (imbalance ratio 1.34). Stratified splits and macro metrics were used."],
       ["8", "Representative samples", "Real records and class profiles show an ordered pattern from Unripe to Ripe to Overripe on every ripeness attribute. Size and temperature profiles are flat across classes."]],
      [0.7 * cm, 3.3 * cm, W - 4 * cm])
fig("eda_boxplots_by_class", "Figure 1 – Feature distributions by class. Ripeness attributes separate the classes; size and temperature do not. Ripe overlaps both of its neighbours.")
fig("eda_correlation", "Figure 2 – Pearson and Spearman correlation matrices.")
p("Additional tabular analysis: one-way ANOVA and Kruskal–Wallis tests both put p &lt; 10<super>−50</super> for the seven ripeness attributes. Weight, length, width and storage temperature are not significant (p &gt; 0.05).")

# ------------------------------------------------------------------ 3 preprocessing
p("3. Phase II – Preprocessing and Feature Engineering", H1)
table([["Operation", "Why", "Parameters"],
       ["Remove duplicates", "Avoid double weighting and leakage between training and test sets", "drop_duplicates(keep='first')"],
       ["Impossible values → NaN", "Data-entry errors distort scaling and distances", "domain limits (e.g. Brix 4–30, hue 30–140°)"],
       ["Stratified split first", "No information from the test set reaches any fitted step", "75/25, stratify=y, random_state=42"],
       ["Median imputation", "Distance-based methods cannot use NaN; the median is robust to skew", "SimpleImputer(median), fitted on training rows"],
       ["One-hot encoding of variety", "Turns categories into numbers without implying an order", "3 binary columns"],
       ["Feature engineering", "Ratios and aggregates capture ripening physiology", "7 new features (below)"],
       ["Feature selection", "Uninformative or redundant dimensions add noise to distances", "MI ≥ 0.10, ANOVA p &lt; 0.05, Spearman |ρ| ≤ 0.95"],
       ["Standardisation", "Stops features with large units (grams) from dominating distances", "StandardScaler inside the pipeline, fitted on training data only"]],
      [3.6 * cm, 6.8 * cm, W - 10.4 * cm])
p("<b>Engineered features.</b> brix_acid_ratio (sugar–acid maturity index); sweet_soft_index (Brix / firmness); "
  "firmness_per_100g (firmness adjusted for size); thermal_time = days × (T − 10 °C) (accumulated degree-days, an aggregated feature); "
  "volume_cm3 (ellipsoid volume); density_g_cm3; log_dark_spots (removes skew). Each one is meant to capture a process that changes "
  "monotonically with ripening. For example, the mean sugar–acid ratio rises from 12 to 38 to 88 across the three classes.")
p(f"<b>Selection.</b> Size, density, volume, the variety dummies and storage temperature carry no class information (MI ≈ 0 or ANOVA p &gt; 0.05). "
  "Three engineered or raw features were dropped as redundant because they are near-monotonic copies of a kept feature: brix_acid_ratio (ρ = 0.985 with acidity), "
  "sweet_soft_index (ρ = 0.993 with firmness) and dark_spots (ρ = 1.0 with log_dark_spots). "
  f"<b>Final 9 features:</b> {', '.join(R['selected_features'])}. They cover texture, chemistry, appearance and aroma, and heat history.")
fig("fe_mutual_info", "Figure 3 – Mutual information of each candidate feature with the class, computed on the training set.", W * 0.8)

# ------------------------------------------------------------------ 4 KNN
p("4. Phase III – KNN Classification", H1)
p("<b>Method.</b> KNN stores the scaled training points. To classify a mango it finds the K nearest training points by the chosen distance and "
  "predicts the class they vote for. I tried K ∈ {1, 3, 5, 7, 9, 11, 15, 21}, three distance metrics (Euclidean, Manhattan, Chebyshev) and "
  "two voting schemes (uniform and distance-weighted). The scaler sits inside a Pipeline, so it is fitted on training folds only. "
  "Configurations were compared by <b>5-fold stratified CV on the training set</b>. The test set (83 mangoes) was used only to evaluate the final model.")
ab = R["ablation"]
grid = {(g["metric"], g["weights"], g["K"]): g for g in R["knn_grid"]}
rows = [["K", "Euclidean CV acc", "Manhattan CV acc", "Euclidean test acc", "Manhattan test acc"]]
for k in [1, 3, 5, 7, 9]:
    e, m = grid[("euclidean", "uniform", k)], grid[("manhattan", "uniform", k)]
    rows.append([k, f3(e["cv_acc_mean"]), f3(m["cv_acc_mean"]), f3(e["test_acc"]), f3(m["test_acc"])])
table(rows, [1.5 * cm] + [(W - 1.5 * cm) / 4] * 4)
fig("knn_k_vs_accuracy", "Figure 4 – Cross-validated accuracy against K for each metric and voting scheme. Dotted lines show training accuracy.")
p(f"<b>Best configuration: K = {K['K']}, {K['metric']} distance, {K['weights']} voting.</b> It has the highest CV accuracy ({K['cv_acc']:.3f}) and macro-F1, "
  "a low spread across folds, and training accuracy close to CV accuracy. K = 1 overfits: training accuracy is 100 % and CV accuracy is lowest. Large K blurs the boundary "
  "between adjacent stages. Manhattan distance is less affected by one noisy feature than Euclidean or Chebyshev. "
  "Distance weighting gave the same CV score, so I kept the simpler uniform vote. "
  "<b>Ablation</b> (same K and metric, CV accuracy): " + "; ".join(f"{k}: {v['cv_acc_mean']:.3f}" for k, v in ab.items()) +
  ". Adding uninformative dimensions hurts KNN, which supports the feature-selection step.")
pc = K["per_class"]
table([["Class", "Precision", "Recall", "F1", "Support"]] +
      [[c, f3(pc[c]["precision"]), f3(pc[c]["recall"]), f3(pc[c]["f1"]), int(pc[c]["support"])] for c in ["Unripe", "Ripe", "Overripe", "macro avg", "weighted avg"]],
      [3.5 * cm] + [(W - 3.5 * cm) / 4] * 4)
fig("knn_confusion_matrix", f"Figure 5 – Test-set confusion matrix. Accuracy = {K['test_acc']:.3f}, macro-F1 = {K['macro_f1']:.3f}.", W * 0.85)
p("<b>Error analysis.</b> 10 of 83 test mangoes are misclassified: 4 Ripe→Unripe, 3 Ripe→Overripe, 2 Unripe→Ripe and 1 Overripe→Ripe. "
  "<b>No Unripe mango was predicted Overripe, and no Overripe mango was predicted Unripe.</b> The misclassified fruit have measurements that sit between two class profiles. "
  "For example, MG218 is labelled Ripe but is firm (29 N), green (97.5°) and low in sugar (13.4 °Brix). MG113 is labelled Ripe but already has spots, about 22 °Brix and 7 N firmness. "
  "On average only 23 % of a misclassified mango's 3 neighbours share its true class, against 97 % for correctly classified mangoes. "
  "<b>Ripe is the hardest class (recall 0.74)</b> because it borders two classes. Other causes are the continuous nature of ripening, "
  "noise in the grader's labels near thresholds, instrument noise, and different sugar and colour baselines between varieties (5 of the 10 errors are Anwar Ratol). "
  "With only 83 test mangoes, one error changes accuracy by 1.2 %. The CV estimate (0.931 ± 0.016) and the test estimate (0.880) therefore suggest a true accuracy of roughly 88–93 %.")
fig("knn_misclassified_pca", "Figure 6 – Test mangoes in PCA space, with misclassified mangoes circled. They lie on the borders between classes.", W * 0.6)

# ------------------------------------------------------------------ 5 KMeans
p("5. Phase IV – K-Means Clustering", H1)
p("<b>Method.</b> K-Means minimises the within-cluster sum of squares (WCSS). It repeatedly assigns each point to its nearest centroid and then moves each centroid to the mean of its points. "
  "Settings: init = k-means++, n_init = 20, max_iter = 300, random_state = 42. The algorithm sees only the 9 standardised features of all 330 mangoes. "
  "Labels are used afterwards for evaluation only.")
rows = [["K", "WCSS", "Silhouette", "ARI", "NMI"]] + [[t["K"], f"{t['WCSS']:.1f}", f3(t["silhouette"]), f3(t["ARI"]), f3(t["NMI"])] for t in M["table"]]
table(rows, [1.5 * cm] + [(W - 1.5 * cm) / 4] * 4)
fig("kmeans_elbow_silhouette", "Figure 7 – Elbow method (left) and Silhouette, ARI and NMI (right) for K = 2–10.")
p("<b>Choice of K = 3.</b> WCSS drops most between K = 2 and K = 3 (39 %), compared with 24 %, 14 % and below 12 % for later steps. The elbow is at 3. "
  "Silhouette is highest at K = 2 (0.511), but K = 2 merges two ripeness stages. K = 3 still has a moderate silhouette (0.442) and matches the three ripeness stages.")
fig("kmeans_clusters_pca", "Figure 8 – K-Means clusters (left) compared with the true classes (centre) in PCA space (PC1 + PC2 = 90.6 % of variance), with the silhouette plot (right).")
table([["Cluster", "n", "Profile (means)", "Unripe / Ripe / Overripe", "Dominant (purity)"],
       ["2", "80", "firmness 43.8 N, Brix 10.1, hue 100.7°, 3 days", "80 / 0 / 0", "Unripe (1.000)"],
       ["1", "124", "firmness 20.1 N, Brix 14.6, hue 84.9°, 7.5 days", "47 / 77 / 0", "Ripe (0.621)"],
       ["0", "126", "firmness 8.7 N, Brix 20.9, hue 66.9°, spots, 14 days", "0 / 31 / 95", "Overripe (0.754)"]],
      [1.4 * cm, 1 * cm, 7.2 * cm, 3.6 * cm, W - 13.2 * cm])
p(f"<b>Interpretation.</b> Without labels, K-Means found an ordered ripening gradient: green and hard, then turning, then soft, sweet and spotted. "
  f"Silhouette = {M['silhouette']:.3f}, ARI = {M['ARI']:.3f}, NMI = {M['NMI']:.3f}. "
  "78 mangoes (24 %) fall in an unexpected cluster: 47 late-stage Unripe fruit in the middle cluster and 31 late-stage Ripe fruit in the soft cluster. Unripe and Overripe fruit never share a cluster. "
  "K-Means looks for compact groups. The hardest green fruit form a tight group of their own, so the cluster boundaries shift towards the ripe end and do not line up exactly with the grader's thresholds. "
  "The clusters are meaningful as ripeness segments, but they do not reproduce the labels exactly.")

# ------------------------------------------------------------------ 6 Agglomerative
p("6. Phase V – Agglomerative Hierarchical Clustering", H1)
p("<b>Method.</b> Agglomerative clustering works bottom-up. It starts with every mango as its own cluster and repeatedly merges the two closest clusters. "
  "The linkage method defines the distance between clusters. Single uses the minimum pairwise distance, complete the maximum, average the mean, "
  "and Ward the increase in WCSS caused by a merge. Flat assignments come from sklearn AgglomerativeClustering with Euclidean distance; "
  "the dendrograms come from scipy.cluster.hierarchy.linkage.")
rows = [["Linkage", "k", "Silhouette", "ARI", "NMI", "Cluster sizes"]] + \
       [[t["linkage"], t["n_clusters"], f3(t["silhouette"]), f3(t["ARI"]), f3(t["NMI"]), t["cluster_sizes"]] for t in A["table"] if t["n_clusters"] in (2, 3, 4)]
table(rows, [2 * cm, 0.8 * cm, 2 * cm, 1.8 * cm, 1.8 * cm, W - 8.4 * cm])
fig("agg_linkage_pca", "Figure 9 – Three-cluster solutions for each linkage, shown in PCA space.")
p("<b>Linkage comparison.</b> Single linkage chains every point into one giant cluster and leaves only singletons outside it (ARI ≈ 0). Its high silhouette is misleading, because it comes from isolating one outlier. "
  "Average linkage also isolates an outlier at k = 2 but finds real structure from k = 3. Complete linkage gives a large mixed cluster. "
  "Ward gives balanced, compact clusters at every k. <b>The choice of linkage changes the grouping far more than the number of clusters does.</b> "
  "For Ward and complete linkage, silhouette falls as k grows, while ARI and NMI peak around k = 4–5.")
fig("agg_dendrograms_all", "Figure 10 – Dendrograms (last 40 merges) for the four linkages. The red line marks the 3-cluster cut.")
p("<b>Merge stages (Ward).</b> The first merges join nearly identical fruit of the same class at distances of about 0.3. Examples: MG051 + MG082 (Ripe Anwar Ratol, 17 °Brix), "
  "MG169 + MG313 (Overripe Chaunsa) and MG062 + MG083 (Unripe Sindhri, about 45 N). Middle stages (around step 160, distance about 0.9) build small neighbourhoods of 2–3 fruit. "
  "In the last merges the distance jumps from 16.5 to 30.8 to 59.9. The final merge joins a 132-fruit cluster that holds every Overripe mango plus 37 Ripe ones with a 198-fruit "
  "cluster of Unripe and Ripe fruit. The 60 hardest green mangoes (100 % Unripe) join at height 30.8. These large gaps show that 2–3 groups is the natural level of the hierarchy.")
table([["Ward cluster", "n", "Profile (means)", "Unripe / Ripe / Overripe", "Dominant (purity)"],
       ["2", "60", "firmness 46.4 N, Brix 10.0, hue 102.9°", "60 / 0 / 0", "Unripe (1.000)"],
       ["0", "138", "firmness 22.7 N, Brix 13.9, hue 86.6°", "67 / 71 / 0", "Ripe (0.514)"],
       ["1", "132", "firmness 8.8 N, Brix 20.8, hue 67.4°, spots", "0 / 37 / 95", "Overripe (0.720)"]],
      [2 * cm, 1 * cm, 6.6 * cm, 3.6 * cm, W - 13.2 * cm])
pa = A["pairwise_ari"]
p(f"Ward (3 clusters): Silhouette = {A['silhouette']:.3f}, ARI = {A['ARI']:.3f}, NMI = {A['NMI']:.3f}. Pairwise agreement between partitions (ARI): "
  f"Ward vs K-Means {pa['ward (3)']['K-Means (3)']:.2f}, Ward vs average {pa['ward (3)']['average (3)']:.2f}, "
  f"complete vs average {pa['complete (3)']['average (3)']:.2f}, single vs every other method ≤ 0.01. Ward and K-Means agree closely because both minimise within-cluster variance.")

# ------------------------------------------------------------------ 7 comparison
p("7. Phase VI – Comparative Analysis", H1)
p("Classification and clustering are different tasks, so each is judged with its own metrics. KNN uses accuracy, precision, recall and F1 on held-out labels. "
  "The clustering methods use Silhouette, an internal measure that needs no labels, and ARI and NMI, external measures that ignore the arbitrary numbering of clusters. "
  "Cluster IDs were never compared directly with class labels. For an illustration only, each cluster was mapped to its most frequent actual class "
  "and I counted how many mangoes share the mapped class. This uses the labels after fitting, so it is not a prediction accuracy and cannot be compared with KNN's accuracy.")
table([["Criterion", "KNN", "K-Means", "Agglomerative (Ward)"],
       ["Learning type", "Supervised", "Unsupervised", "Unsupervised"],
       ["Labels used during fitting", "Yes", "No", "No"],
       ["Accuracy on held-out labels", f"{K['test_acc']:.3f} (macro-F1 {K['macro_f1']:.3f}; CV {K['cv_acc']:.3f})", "Not directly applicable", "Not directly applicable"],
       ["Silhouette Score", "Not applicable", f3(M["silhouette"]), f3(A["silhouette"])],
       ["ARI / NMI", "Not applicable", f"{M['ARI']:.3f} / {M['NMI']:.3f}", f"{A['ARI']:.3f} / {A['NMI']:.3f}"],
       ["Majority-mapped agreement (illustrative)", "—", f3(M["mapped_agreement"]), f3(A["mapped_agreement"])],
       ["Main observations", "High accuracy. Every error is between adjacent stages. Ripe is the hardest class. Feature selection added about 3 points of CV accuracy.",
        "Recovers an ordered ripening gradient. Best ARI/NMI of the clustering methods. Contains a pure Unripe cluster. Elbow at K = 3.",
        "Same ordered structure with balanced clusters. The dendrogram shows natural 2–3 groups. Results depend strongly on the linkage, and single linkage fails."],
       ["Limitations", "Needs labels. Slow at prediction time on large data. Sensitive to scaling, irrelevant features and label noise. The test set is small.",
        "K must be chosen in advance. Assumes spherical clusters of similar size. Sensitive to initialisation and outliers. Cuts a continuum at arbitrary points.",
        "O(n²) time and memory. Greedy merges cannot be undone. Sensitive to the linkage choice. Ward requires Euclidean distance."]],
      [3.4 * cm, (W - 3.4 * cm) / 3, (W - 3.4 * cm) / 3, (W - 3.4 * cm) / 3])
fig("comparison_clustering", "Figure 11 – Clustering quality at 3 clusters for each method.", W * 0.7)
bullets([
    "<b>Supervised vs unsupervised.</b> With labels, KNN reaches about 88–93 %. Without labels, clustering still finds groups strongly related to ripeness (NMI 0.53–0.58). The classes therefore reflect real structure in the measurements.",
    "<b>Why agreement with the labels is only moderate.</b> The classes are thresholds on a continuous process. Clustering places its boundaries where the groups are most compact, not where the grader's thresholds are. Labels are needed to put the boundaries in the grader's places.",
    "<b>Internal and external metrics disagree.</b> Silhouette prefers K = 2 and even rewards single linkage for isolating an outlier. ARI and NMI prefer 3–4 clusters. Both kinds of metric are needed, together with domain knowledge.",
    "<b>Pattern common to all three methods.</b> Unripe and Overripe are never confused. Every error or mixed cluster sits at the boundary between adjacent stages and mostly involves Ripe fruit.",
    "<b>Recommendation.</b> Use KNN to grade individual fruit when labels are available. Use K-Means (K = 3) to segment an unlabelled batch. Use the Ward dendrogram to decide how many natural groups exist.",
])

# ------------------------------------------------------------------ 8 conclusion
p("8. Assumptions, Limitations and Conclusions", H1)
p("<b>Assumptions.</b> The data come from my documented model, with coefficients approximated from general mango physiology. "
  "The grader's label is treated as ground truth, even though it is noisy near the boundaries. Values are missing completely at random. "
  "Readings outside physical limits are entry errors. After standardisation, every feature contributes equally to the distance.")
p("<b>Limitations.</b> Generated data are more regular than a real orchard: there is no instrument drift and no disease, and only three varieties are included. "
  "The dataset is small (330 fruit; 83 in the test set), so the accuracy estimate is uncertain by about ±3–4 %. "
  "The clustering metrics describe the clusters found on this data; they do not measure prediction on new fruit.")
p(f"<b>Conclusions.</b> The full pipeline was implemented and is reproducible, with fixed seeds and every fitted step fitted on training data only. "
  f"KNN (K = {K['K']}, {K['metric']}) classified ripeness with {K['test_acc']:.1%} test accuracy, and all its errors were between adjacent stages. "
  f"K-Means and Ward clustering found the ripening gradient without labels (ARI {M['ARI']:.2f} / {A['ARI']:.2f}). "
  "Feature engineering and selection improved KNN by about 3 points over the raw attributes.")
p("<b>Reproduce:</b> pip install -r requirements.txt → python data/generate_dataset.py → run Assignment2_Mango_Ripeness.ipynb top to bottom → python report/build_report.py.")


def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 8); c.setFillColor(colors.grey)
    c.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"Page {d.page}")
    c.drawString(1.8 * cm, 1.1 * cm, "Advanced AI – Integrated Practical Assignment – Mango Ripeness")
    c.restoreState()


SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.8 * cm,
                  title="Mango Ripeness – Advanced AI Assignment Report").build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)

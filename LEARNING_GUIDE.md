# Learning Guide: Mango Ripeness Assignment

This guide follows the notebook from start to finish. For each step it explains what we did, why we did it, the ML concept behind it, and a question you may be asked in the viva.

---

## Big picture first

**Problem:** given a mango's measurements, decide whether it is **Unripe, Ripe or Overripe**.

The assignment solves this in two ways:

| Way | Algorithm | Are labels used? | Question it answers |
|---|---|---|---|
| Supervised | KNN | Yes | "Given labelled examples, can we predict the label of a new mango?" |
| Unsupervised | K-Means, Agglomerative | No | "Without labels, do natural groups exist, and do they match ripeness?" |

The full pipeline:
```
Data → EDA → Clean → Feature Engineering → Scale → Model → Evaluate → Compare
```
Almost every ML project follows this same pipeline.

---

## Step 1: Dataset generation

**What we did:** the script `data/generate_dataset.py` gives each mango a hidden "ripening stage" r between 0 and 1. Every measurement is then calculated from r, with some noise added:
- Firmness goes down (the fruit softens).
- Sugar (Brix) goes up (starch turns into sugar).
- Acidity goes down.
- Peel hue goes down (from green at about 110° to orange at about 60°).
- Aroma goes up, and dark spots appear late in ripening.

The label comes from cutting r at two thresholds:

- r below 0.38 means Unripe.
- r from 0.38 to 0.70 means Ripe.
- r above 0.70 means Overripe.

A small amount of "grader noise" is added near the thresholds. We also planted problems on purpose: missing values, typing errors and duplicate rows.

**Concepts:**
- **Latent variable:** a hidden cause (here r) that drives everything you can observe.
- **Label noise:** real labels are imperfect, especially near the class boundaries.
- **Reproducibility:** a fixed random seed (2025) produces the same data every time.
- **Data dictionary:** a document listing every column, its unit and its meaning.

**Viva question:** *"Why add errors on purpose?"* → So that the cleaning steps have real work to do, and because real data always contains these problems.

---

## Step 2: Phase I, Exploratory Data Analysis (EDA)

**Goal:** understand the data before building any model. Never feed a model data you have not looked at.

| Analysis | What we found | Concept |
|---|---|---|
| Shape and types | 338 rows × 14 columns. `describe()` showed impossible values (weight 0.18 g, Brix 48) | Data profiling |
| Missing values | 38 cells (about 0.8%), scattered randomly | **MCAR** (Missing Completely At Random), so filling them in is safe |
| Duplicates | 8 rows entered twice; we removed them | Duplicates cause **leakage**: the same row can end up in both train and test |
| Outliers | IQR, z-score and domain limits found 6 impossible values. High dark-spot counts were **real** and kept | An outlier is not always an error. Domain knowledge decides |
| Distributions | Firmness and acidity are skewed; Brix is bimodal | **Skewness**, histograms, box plots |
| Correlation | Firmness and acidity have r = 0.94; size is unrelated to ripeness | **Pearson** (linear) vs **Spearman** (monotonic) |
| Class balance | 127 / 108 / 95, a ratio of 1.34, so only mild imbalance | Imbalance calls for stratified splits and macro metrics |
| Statistical tests | ANOVA and Kruskal-Wallis: ripeness features p < 10⁻⁵⁰; size features not significant | Hypothesis testing to judge feature relevance |

**Key formulas:**
- IQR rule: a value is an outlier if `x < Q1 − 1.5·IQR` or `x > Q3 + 1.5·IQR`.
- Z-score: `z = (x − μ)/σ`. Flag the value if |z| > 3.

**Viva question:** *"Why not delete all outliers?"* → The 30 high dark-spot values are real overripe mangoes. Deleting them would throw away information about the class.

---

## Step 3: Phase II, Preprocessing and Feature Engineering

### A. Preprocessing (the order matters)
1. Remove duplicates.
2. Set impossible values to missing (NaN).
3. **Split into train and test first** (75/25, stratified).
4. Fill missing values with the median, computed **from the training data only**.
5. Standardise each feature: `z = (x − mean)/std`.

**The most important concept here is data leakage.** If the test set influences any fitted step (imputation, scaling or feature selection), the test score will look better than it really is. The rule: **fit on the training data, then only transform the test data.**

**Why scale?** KNN and clustering use distances. Without scaling, weight (hundreds of grams) dominates acidity (around 0.5), and the model effectively ignores acidity.

**Why the median instead of the mean?** The median is not pulled around by skew or outliers.

### B. Feature engineering
Feature engineering means creating new features that carry the signal better than the raw columns.

| Feature | Formula | Idea |
|---|---|---|
| brix_acid_ratio | sugar / acid | The maturity index used in the fruit industry |
| thermal_time | days × (temp − 10) | Heat accumulated over time is what drives ripening |
| firmness_per_100g | firmness / weight | Removes the effect of fruit size |
| log_dark_spots | ln(1 + spots) | Reduces the skew |
| volume, density, sweet_soft_index, one-hot variety | | Also created |

**One-hot encoding:** the variety column becomes three 0/1 columns. Coding varieties as 1, 2, 3 would invent an order that does not exist.

### C. Feature selection
- **Mutual Information:** how much a feature tells you about the class. It also catches non-linear relationships.
- **ANOVA F:** compares how far apart the class means are with the spread inside each class.
- **Redundancy:** if two features have Spearman |ρ| > 0.95, drop one of them, because it is almost a copy.

Result: 21 candidate features were reduced to **9 selected features**.

**Evidence that selection helped (ablation):**

| Feature set | Cross-validation accuracy |
|---|---|
| Raw features | 0.903 |
| All 21 features | 0.887 |
| **Selected 9 features** | **0.931** |

More features are not always better. This is the **curse of dimensionality**: noisy dimensions make distances less meaningful.

---

## Step 4: Phase III, KNN (supervised)

**How KNN works:**
1. Store all the training points. That is the whole of "training", which is why KNN is called a **lazy learner**.
2. For a new mango, measure its distance to every training point.
3. Take the K nearest points.
4. The majority vote among them decides the class.

**Distance metrics:**
- Euclidean: `√Σ(xᵢ−yᵢ)²`, the straight-line distance.
- Manhattan: `Σ|xᵢ−yᵢ|`, the city-block distance. It is less affected by one noisy feature. **This metric won.**
- Chebyshev: `max|xᵢ−yᵢ|`, where only the largest difference counts.

**Choosing K (the bias-variance tradeoff):**
- K = 1 memorises noise: training accuracy is 100%, but cross-validation accuracy is the worst. This is **overfitting**.
- A large K is too smooth and blurs the boundaries between classes. This is **underfitting**.
- K = 3 was best for this data.

**How the model was chosen:** **5-fold stratified cross-validation** on the training set. The training data is split into 5 parts. The model trains on 4 parts and is tested on the fifth, and this rotates 5 times. The 5 scores are averaged. The test set is used **only once**, at the very end. Choosing a model by its test score is cheating, because it makes the final score look better than it is.

**Pipeline:** `Pipeline([StandardScaler, KNN])` refits the scaler inside each fold automatically, so no information leaks between folds.

**Evaluation metrics:**
- **Accuracy:** correct predictions divided by all predictions. Ours was 0.880.
- **Precision:** of the mangoes predicted Ripe, how many really were Ripe.
- **Recall:** of the mangoes that really were Ripe, how many we found. Ripe had the lowest recall (0.74).
- **F1:** the harmonic mean of precision and recall.
- **Macro average:** every class counts equally, whatever its size.
- **Confusion matrix:** shows which classes get mixed up. Every error was between neighbouring stages. Unripe and Overripe were never confused with each other.

**Viva question:** *"Why is Ripe the hardest class?"* → Ripe borders two classes, ripening is continuous, and labels are noisy near the thresholds.

---

## Step 5: Phase IV, K-Means (unsupervised)

**How it works:**
1. Pick K starting centres. The **k-means++** method spreads them out.
2. Assign each point to its nearest centre.
3. Move each centre to the mean of the points assigned to it.
4. Repeat steps 2 and 3 until nothing changes.

**Objective:** minimise **WCSS (inertia)**, the sum of squared distances from each point to its centre.

**Settings:** `n_init=20` runs the algorithm 20 times and keeps the best result. This avoids getting stuck in a poor local minimum.

**Choosing K:**
- **Elbow method:** plot WCSS against K and look for the bend. The biggest drop came at K = 3 (39%).
- **Silhouette:** `s = (b − a)/max(a, b)`, where a is the average distance to points in your own cluster and b is the average distance to the nearest other cluster. It ranges from −1 to 1, and higher is better.

**Evaluating against the labels afterwards:**
- **ARI (Adjusted Rand Index):** checks whether pairs of mangoes are grouped together or apart in the same way by the clusters and by the classes. It is corrected for chance: 0 means random, 1 means perfect agreement.
- **NMI (Normalized Mutual Information):** how much information the clusters and the classes share, on a scale from 0 to 1.
- Cluster IDs are **arbitrary**. "Cluster 0" does not mean "Unripe". Never compare the raw ID numbers with the labels.

**Finding:** without seeing any labels, K-Means found three groups: green and hard, turning, and soft with spots. One cluster was 100% Unripe. Its boundaries do not sit exactly where the grader's thresholds are, because K-Means looks for compact groups, not human-defined cut-offs.

**PCA:** compresses the 9 features into 2 so they can be plotted. It is used for visualisation only. The first component (PC1) alone explained 84% of the variance, because all the ripeness features move together.

---

## Step 6: Phase V, Agglomerative clustering

**How it works (bottom-up):** start with 330 clusters, one mango each. Repeatedly merge the two closest clusters until only one cluster is left. The record of these merges is drawn as a **dendrogram**.

**Linkage methods (how the distance between two clusters is measured):**

| Linkage | Distance between clusters | What happened in our data |
|---|---|---|
| Single | The closest pair of points | **Chaining**: one giant cluster plus single points. It failed (ARI ≈ 0) |
| Complete | The farthest pair of points | Produced one large mixed cluster |
| Average | The mean over all pairs | Reasonable results from k = 3 |
| Ward | The increase in WCSS caused by merging | Balanced, compact clusters. It has the same goal as K-Means |

**Reading a dendrogram:**
- The height of a join is the distance at which two clusters merged.
- Large vertical jumps show the natural number of clusters. Here the jumps were 16, then 31, then 60, which points to 2–3 groups.
- Cutting the tree with a horizontal line gives the final flat clusters.
- The first merges joined near-identical mangoes of the same class, for example MG051 and MG082.

**Lesson:** the choice of linkage changed the result more than the number of clusters did.

---

## Step 7: Phase VI, Comparison

**Key rule from the assignment PDF:** do not compare clustering and classification using accuracy as if they were the same task.
- For KNN, use accuracy, precision, recall and F1.
- For clustering, use Silhouette (internal, needs no labels) together with ARI and NMI (external, uses the labels).

**Majority mapping:** for illustration only, give each cluster the label of its most common class. You must explain that this is not a real accuracy, because the labels are used after the model has been fitted.

**Main takeaways:**
1. Labels help: KNN reaches roughly 88–93% accuracy.
2. Clustering still finds the ripeness structure (NMI about 0.55), so the classes are real patterns in the data.
3. Silhouette and ARI can disagree. Silhouette preferred K = 2, while ARI preferred 3–4.
4. No method ever confused Unripe with Overripe.

---

## ML concepts cheat sheet

| Area | Concepts |
|---|---|
| Learning types | Supervised vs unsupervised; lazy vs eager learners |
| Data quality | Missing values (MCAR), imputation, duplicates, outliers vs errors |
| Statistics | Mean, median, skew, IQR, z-score, Pearson and Spearman correlation, ANOVA, Kruskal-Wallis |
| Preprocessing | Train/test split, stratification, scaling, one-hot encoding, **data leakage** |
| Features | Feature engineering, ratios, log transforms, mutual information, redundancy, curse of dimensionality, ablation |
| Model selection | Cross-validation, hyperparameter tuning, bias-variance tradeoff, overfitting and underfitting |
| Distances | Euclidean, Manhattan, Chebyshev |
| Classification metrics | Accuracy, precision, recall, F1, macro and weighted averages, confusion matrix |
| Clustering | K-Means, k-means++, WCSS, elbow method, silhouette, hierarchical clustering, linkage methods, dendrogram |
| External validation | ARI, NMI, majority mapping |
| Dimensionality reduction | PCA (for visualisation) |

---

## Skills and tools developed

### Tools
- **Python and pandas:** loading and cleaning data, `groupby`, `crosstab`.
- **NumPy:** numerical operations and random number generation with seeds.
- **Matplotlib and Seaborn:** histograms, box plots, heatmaps, pair plots, scatter plots.
- **SciPy:** statistical tests, hierarchical linkage, dendrograms.
- **scikit-learn:** `Pipeline`, `StandardScaler`, `SimpleImputer`, `KNeighborsClassifier`, `KMeans`, `AgglomerativeClustering`, `PCA`, `cross_val_score`, and the metrics functions.
- **Jupyter:** a reproducible analysis notebook.
- **Git and GitHub:** version control and submission.
- **ReportLab:** generating the PDF report from code.

### Skills
- **Data design:** turning a real-world problem into a structured, documented dataset.
- **Data cleaning judgement:** telling genuine errors apart from real but extreme values.
- **Feature thinking:** using domain knowledge to build better inputs.
- **Experimental rigour:** avoiding leakage, cross-validation, fixed seeds, ablation studies.
- **Model evaluation:** choosing the right metric for each task.
- **Interpretation:** explaining *why* a model makes mistakes, not just reporting numbers.
- **Technical communication:** writing reports and telling a clear story with charts.

---

## Viva preparation: likely questions

1. *Why split the data before imputing?* → To prevent leakage. Statistics from the test set must never influence training.
2. *Why scale the features for KNN?* → KNN is based on distance. Features with large units would dominate the distance.
3. *Why use an odd K?* → It avoids tied votes between two classes.
4. *Why did Manhattan distance win?* → It is less affected by a single noisy feature.
5. *K = 1 gives 100% training accuracy. Why is that bad?* → Each training point is its own nearest neighbour, so the model has memorised the data. That is overfitting.
6. *Why choose the model by cross-validation and not by the test score?* → Choosing by test score tunes the model to the test set, so the final estimate is too optimistic.
7. *Silhouette was best at K = 2, so why choose K = 3?* → The elbow is at 3, K = 3 matches the three ripeness stages, and K = 2 merges two stages together.
8. *Why can't you compare a cluster ID directly with a label?* → Cluster IDs are arbitrary numbers. Use ARI or NMI instead.
9. *Why did single linkage fail?* → Chaining: in continuous data, a chain of close neighbours links every point together.
10. *How do Ward and K-Means compare?* → Both minimise WCSS, so their results are similar (ARI of 0.80 between them).
11. *What is PCA used for here?* → Only for the 2-D plots. The models use all 9 features.
12. *What are the limitations?* → The data is generated rather than collected, the test set is small (83 mangoes, so about ±3–4% uncertainty), and real data would be messier.

---

## Suggested study order

1. Open the notebook and read only the markdown (text) cells, from top to bottom.
2. Rerun the notebook one cell at a time and look at every plot.
3. For each figure, say out loud in one sentence what it shows.
4. Change K or the distance metric, rerun, and watch how the numbers change.
5. Answer the 12 viva questions above without looking at your notes.

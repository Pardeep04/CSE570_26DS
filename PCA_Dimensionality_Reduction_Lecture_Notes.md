# Lecture Notes: Curse of Dimensionality and Dimensionality Reduction Using PCA

**Subject:** Machine Learning  
**Topics:** Curse of Dimensionality, Dimensionality Reduction, Principal Component Analysis (PCA), PC1/PC2/PC3, Explained Variance, and Python Implementation

## 1. Curse of Dimensionality

The **curse of dimensionality** describes difficulties that can arise as the number of features (dimensions) in a dataset becomes very large. In a high-dimensional space, a fixed number of data points tends to become sparse. As a result, many algorithms need more data to learn useful patterns reliably.

**Example:** We initially predict student performance using five features: study hours, attendance, assignment marks, previous marks, and class participation. If we add 95 more features, the dataset has 100 features. Some new features might be relevant, but others could be redundant or noisy. With insufficient training data, the model may overfit or become harder to train.

### Common problems

1. **Data sparsity:** Samples occupy a tiny fraction of a large feature space.
2. **Overfitting:** Models may learn noise rather than patterns that generalize.
3. **Computational cost:** Additional features can increase storage, memory use, and computation.
4. **Less informative distances:** In some high-dimensional datasets, distances between observations become less distinguishable, affecting algorithms such as KNN and K-Means.
5. **Greater sample requirements:** More training examples may be needed to cover relevant parts of the feature space.

**One way to address these problems:** Dimensionality reduction.

## 2. What Is Dimensionality Reduction?

**Dimensionality reduction** means representing a dataset using fewer features while trying to retain useful information.

For example, a dataset with 100 input features might be represented using 10 new features. Whether this works well depends on how much useful structure can be compressed.

### Why use it?

- Reduce the number of features and potentially lower computational requirements.
- Remove redundant variation.
- Make high-dimensional data easier to analyze.
- Sometimes reduce noise and overfitting.
- Enable 2D or 3D visualization (though visualization is **not** included in the code in these notes).

### Two main approaches

| Feature selection | Feature extraction |
|---|---|
| Keeps a subset of original columns | Creates new columns from existing ones |
| Original feature meaning is retained | New feature meaning may be harder to interpret |
| Example: Retain study hours and attendance | Example: Create PC1 and PC2 |
| Examples: Correlation filtering, ANOVA, RFE | Examples: PCA, LDA, autoencoders |

**PCA is a feature extraction method.**

## 3. What Is Principal Component Analysis (PCA)?

**Principal Component Analysis (PCA)** is an **unsupervised, linear dimensionality reduction technique**. It transforms numerical features into new, mutually perpendicular directions called **principal components**. These components are ordered by the amount of variance they capture.

PCA does **not** simply choose two existing columns. Instead, it calculates new weighted combinations of the original columns.

### Example dataset

| Student | Study hours | Attendance | Assignment marks | Test marks |
|---|---:|---:|---:|---:|
| A | 2 | 50 | 40 | 45 |
| B | 4 | 65 | 55 | 60 |
| C | 6 | 75 | 70 | 72 |
| D | 8 | 90 | 85 | 88 |
| E | 10 | 95 | 95 | 96 |

The features tend to increase together, so they contain some overlapping information. PCA may represent much of their variation with fewer new columns.

## 4. PC1 and PC2 Explained for Students

Start with just **two features**: Maths and Science marks.

| Student | Maths | Science |
|---|---:|---:|
| A | 20 | 30 |
| B | 40 | 50 |
| C | 60 | 70 |
| D | 80 | 90 |

Both subjects increase together. The most obvious pattern is that students with high Maths marks also have high Science marks.

### 4.1 PC1: The main pattern

**PC1 (first principal component)** is the new direction that captures the **largest possible variance**.

In our marks example, PC1 can be thought of as an **overall performance pattern** because the subjects rise together. As an analogy, average marks also summarize overall performance:

| Student | Maths | Science | Average |
|---|---:|---:|---:|
| A | 20 | 30 | 25 |
| B | 40 | 50 | 45 |
| C | 60 | 70 | 65 |
| D | 80 | 90 | 85 |

**Important:** PC1 is not literally the average in general. PCA determines mathematical weights from the data.

### 4.2 PC2: The remaining pattern

**PC2 (second principal component)** captures the **largest variance left after PC1**, in a direction perpendicular to PC1.

Consider two more students:

| Student | Maths | Science | Average |
|---|---:|---:|---:|
| E | 90 | 50 | 70 |
| F | 50 | 90 | 70 |

Both have the same average, but E is stronger in Maths while F is stronger in Science. An overall-performance measure alone cannot distinguish this difference. PC2 can represent such a **Maths-versus-Science contrast**.

In simple terms:

- **PC1:** What is the biggest pattern in the data? (Here: overall performance.)
- **PC2:** What other independent pattern remains? (Here: differences between subject strengths.)

These interpretations apply to this example, **not automatically to every dataset**.

### 4.3 Why a plus sign in PC1 and a minus sign in PC2?

For two **standardized** features \(X_1\) (Maths) and \(X_2\) (Science) with a positive correlation and equal variance, the principal component directions can be written as:

\[
PC1 = 0.707X_1 + 0.707X_2
\]

\[
PC2 = 0.707X_1 - 0.707X_2
\]

- **The plus sign in PC1:** Captures the *common movement* of Maths and Science. When both rise, the PC1 score rises in this orientation.
- **The minus sign in PC2:** Captures their *contrast*. A student relatively stronger in Maths gets a different PC2 score from a student relatively stronger in Science.

**Intuitive demonstration (not the actual standardized PCA score):**

- E: Maths − Science = 90 − 50 = **+40**
- F: Maths − Science = 50 − 90 = **−40**

The different signs distinguish the students even though they have equal average marks.

**Two important cautions:**

1. **PC1 does not always contain plus signs, and PC2 does not always contain minus signs.** PCA computes coefficients from the actual data and correlation structure.
2. The signs of an *entire component* can be reversed without changing its direction or explained variance. For example, \(0.707X_1-0.707X_2\) and \(-0.707X_1+0.707X_2\) define the same PC axis with opposite orientation.

### 4.4 What is PC3?

**PC3 (third principal component)** captures the greatest remaining variance not captured by PC1 and PC2. It is perpendicular to both. We consider PC3 when two components do not retain enough variance for our chosen goal. A dataset must have at least three independent feature directions for a nonzero PC3.

| Component | Meaning |
|---|---|
| PC1 | Maximum possible variance |
| PC2 | Maximum remaining variance perpendicular to PC1 |
| PC3 | Maximum remaining variance perpendicular to PC1 and PC2 |

## 5. What Is Variance and Why Does PCA Use It?

**Variance** measures how spread out values are around their mean.

| Student | Feature A | Feature B |
|---|---:|---:|
| 1 | 50 | 10 |
| 2 | 50 | 30 |
| 3 | 50 | 60 |
| 4 | 50 | 90 |

- **Feature A** has variance 0 because all values are identical.
- **Feature B** has positive variance because its values differ substantially.

PCA searches for *directions (combinations of features)* along which the observations are spread out the most. Keeping those directions retains as much of the original overall variation as possible for a given number of linear components.

**Caution:** Greater variance does **not** automatically mean greater usefulness for predicting a target. PCA is unsupervised and does not examine target labels.

### 5.1 Explained variance ratio

The **explained variance ratio** reports the share of total variance captured by a component.

Suppose the values are:

| Component | Explained variance | Cumulative variance |
|---|---:|---:|
| PC1 | 60% | 60% |
| PC2 | 25% | 85% |
| PC3 | 10% | 95% |
| PC4 | 5% | 100% |

If our goal is **at least 95%**, we select **PC1 + PC2 + PC3**. PC1 and PC2 alone retain only 85%.

**Rule:** Add the components in order until the cumulative explained variance reaches the required threshold. A 95% threshold is common in teaching examples, but not mandatory for every application.

## 6. How PCA Works: Step by Step

1. **Load the numerical features.** For example, the Iris dataset has four features: sepal length, sepal width, petal length, and petal width.
2. **Standardize features** when appropriate so different numerical scales do not dominate the analysis. The common formula is:

   \[
   Z = \frac{X-\mu}{\sigma}
   \]

   Here \(X\) is an original value, \(\mu\) is the feature mean, and \(\sigma\) is its standard deviation. If \(X=80\), \(\mu=70\), and \(\sigma=10\), the standardized value is \(1\).
3. **Find principal component directions.** PCA analyzes the variation and covariation of features. Mathematically, the directions relate to eigenvectors of the covariance matrix; practical implementations often use singular value decomposition (SVD).
4. **Choose components.** For example, `PCA(n_components=2)` requests exactly two components, while `PCA(n_components=0.95)` chooses enough components to retain at least 95% of variance for a supported solver.
5. **Transform the data.** Represent each original row using scores on the selected components. The number of rows stays the same; the number of columns decreases.
6. **Evaluate explained variance.** Inspect the percentages retained to understand the trade-off between compression and information loss.

## 7. Python Implementation: PCA Without Visualization

**Aim:** Reduce the Iris dataset from **4 features to 2 principal components** using Python. The program includes **no graphs or visualization libraries**.

```python
# =====================================================
# DIMENSIONALITY REDUCTION USING PCA
# Dataset: Iris
# No visualization
# =====================================================

# Step 1: Import required libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 2: Load the Iris dataset
iris = load_iris()

# X contains four numerical features
X = iris.data

# Create a DataFrame to display the original features
df = pd.DataFrame(X, columns=iris.feature_names)

print("Original Dataset:")
print(df.head())
print("\nOriginal Dataset Shape:", X.shape)

# Step 3: Standardize numerical features
# Each feature will have mean approximately 0
# and standard deviation approximately 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFirst Five Standardized Records:")
print(X_scaled[:5])

# Step 4: Apply PCA and select two components
# PCA learns directions of maximum variance
pca = PCA(n_components=2)

# Fit PCA and transform the standardized dataset
X_pca = pca.fit_transform(X_scaled)

print("\nShape After PCA:", X_pca.shape)

# Step 5: Create a reduced dataset with PC1 and PC2
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])

print("\nReduced Dataset:")
print(df_pca.head())

# Step 6: Check the variance explained by each component
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

# Add explained variance percentages for selected components
total_variance = pca.explained_variance_ratio_.sum() * 100
print("\nTotal Variance Retained (%):")
print(round(total_variance, 2))

# Step 7: Examine component weights
# Row 1 has the weights for PC1
# Row 2 has the weights for PC2
# Columns correspond to the original Iris features
print("\nPrincipal Component Weights:")
print(pca.components_)

# Step 8: Compare the number of features
print("\nOriginal Features:", X.shape[1])
print("Reduced Features:", X_pca.shape[1])
```

### 7.1 Expected main output

```text
Original Dataset Shape: (150, 4)
Shape After PCA: (150, 2)
Explained Variance Ratio:
[0.7296 0.2285]
Total Variance Retained (%):
95.81
Original Features: 4
Reduced Features: 2
```

Small differences in displayed decimal precision can occur.

### 7.2 Explanation of important code

| Code | Explanation |
|---|---|
| `load_iris()` | Loads a dataset with 150 samples and 4 numerical input features |
| `StandardScaler()` | Standardizes features to comparable scales |
| `fit_transform(X)` on the scaler | Learns means and standard deviations, then scales data |
| `PCA(n_components=2)` | Requests two new principal components |
| `pca.fit_transform(X_scaled)` | Learns the PCA directions and transforms the input rows |
| `pca.explained_variance_ratio_` | Fraction of total variance captured by PC1 and PC2 |
| `pca.components_` | Weights that describe the new component directions |
| `X_pca.shape` | Confirms the resulting number of samples and components |

**Interpretation:** Before PCA the dataset has 150 rows and 4 features. After PCA it has 150 rows and 2 components. For standardized Iris data, PC1 and PC2 together retain approximately **95.81% of the total variance**.

## 8. Automatic Component Selection (No Visualization)

Use this version if you want PCA to choose the minimum number of components that retain at least 95% variance.

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load numerical input features
X = load_iris().data

# Standardize all features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Select enough components to retain at least 95% variance
pca = PCA(n_components=0.95)

# Fit PCA and create the reduced dataset
X_reduced = pca.fit_transform(X_scaled)

# Print the results
print("Original Features:", X.shape[1])
print("Selected Components:", pca.n_components_)
print("Reduced Dataset Shape:", X_reduced.shape)
print("Variance Retained (%):",
      round(pca.explained_variance_ratio_.sum() * 100, 2))
```

**Expected output for the standardized Iris dataset:**

```text
Original Features: 4
Selected Components: 2
Reduced Dataset Shape: (150, 2)
Variance Retained (%): 95.81
```

### Important note about data leakage

For a supervised ML project, **split the dataset into training and testing subsets before fitting the scaler or PCA**. Fit both transformations on the training subset only, then use the fitted transformations on the test subset. A scikit-learn `Pipeline` is a convenient way to do this. The examples above only demonstrate PCA on a dataset; they are not a predictive model evaluation.

## 9. Advantages and Limitations

| Advantages | Limitations |
|---|---|
| Fewer input columns | Some variance is lost if components are removed |
| Can reduce memory and computation | Components may be difficult to interpret |
| Can remove redundant variation | Sensitive to scaling and outliers |
| Useful for compression and analysis | Captures linear combinations only |
| Does not need class labels | Retaining variance does not guarantee better predictions |

## 10. PCA Versus Feature Selection

| PCA | Feature selection |
|---|---|
| Creates new components | Retains selected original features |
| Uses weighted combinations of inputs | Keeps original column meanings |
| Usually unsupervised | Can be supervised or unsupervised |
| Output names include PC1 and PC2 | Output names are original feature names |

## 11. Questions for Students

1. What is the curse of dimensionality?
2. Define dimensionality reduction and state two benefits.
3. Differentiate feature selection and feature extraction.
4. What is PCA? Is it supervised or unsupervised?
5. Why is standardization often performed before PCA?
6. Explain PC1 using Maths and Science marks.
7. Explain PC2 using two students with identical totals but different subject strengths.
8. Why might PC1 use a plus sign and PC2 a minus sign in a two-feature example?
9. Are the plus and minus signs fixed for every PCA dataset? Explain.
10. What is variance, and why does PCA try to preserve it?
11. What is explained variance ratio?
12. If PC1 = 60%, PC2 = 25%, and PC3 = 10%, how many components retain at least 95% variance?
13. What does `n_components=2` mean?
14. Does PCA reduce the number of rows in a dataset?
15. What is stored in `pca.components_`?
16. Why might PCA fail to improve classification accuracy?

## 12. Quick Revision Summary

- **Curse of dimensionality:** Difficulties caused by working with many features and limited data.
- **Dimensionality reduction:** Represent data using fewer features.
- **PCA:** Creates new uncorrelated, linear component directions that preserve as much variance as possible.
- **PC1:** Captures the largest variance.
- **PC2:** Captures the largest remaining variance, perpendicular to PC1.
- **PC3:** Captures the largest remaining variance, perpendicular to PC1 and PC2.
- **Plus/minus example:** With two positively correlated, standardized features, a sum can capture common movement while a difference captures contrast. The signs are **not universal rules**.
- **Explained variance:** Measures how much variance each component retains.
- **Iris result:** Four original features become two principal components while approximately 95.81% of total standardized variance is retained.

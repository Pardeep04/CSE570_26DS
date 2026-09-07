# CSE570 – Machine Learning with Python
## Unit II: Exploratory Data Analysis and Feature Engineering

> **Topics:** EDA, visualization, correlation analysis, outlier detection, feature selection, feature transformation, data imbalance handling, and preprocessing pipelines.

---

# 1. Exploratory Data Analysis (EDA)

## What is EDA?

**Exploratory Data Analysis (EDA)** is the process of understanding a dataset before applying a machine-learning algorithm.

EDA helps us answer:

- What type of data is available?
- Are values missing?
- Are duplicate records present?
- Are there unusual or extreme values?
- How are variables distributed?
- Which variables are related?
- Is the target class balanced?

<div style="background:#E8F4FD;padding:10px;border-left:5px solid #1976D2;">
<b>Key Idea:</b> Machine-learning models learn from the data we provide. If the data contains errors, noise, missing values, or misleading patterns, the model will also learn from them.
</div>

### Basic inspection

```python
# Import pandas for reading and working with tabular data
import pandas as pd

# Load the customer churn dataset into a DataFrame named 'df'
df = pd.read_csv("CSE570_Unit2_Customer_Churn_Dataset.csv")

# Display the first 5 rows to understand the structure of the dataset
print(df.head())

# Display the number of rows and columns in the dataset
print(df.shape)

# Show column names, data types, non-null counts, and memory usage
df.info()

# Generate summary statistics for numerical columns
df.describe()
```

- `head()` → first few rows
- `shape` → number of rows and columns
- `info()` → data types and missing values
- `describe()` → statistical summary

---

# 2. Missing Values

A **missing value** means information is unavailable for a particular observation.

Example:

| Age | Income |
|---|---|
| 25 | 50000 |
| NaN | 65000 |

### Check missing values

```python
# Check the number of missing (NaN) values in each column
df.isnull().sum()
```

### Common treatments

- Numerical feature → mean or median
- Categorical feature → mode
- Large amount of missing data → consider removing the feature
- Missingness has meaning → create a separate category/indicator

```python
# Replace missing Age values with the median Age
# Median is preferred because it is less affected by extreme values/outliers
df["Age"] = df["Age"].fillna(df["Age"].median())
```

<div style="background:#FFF8E1;padding:10px;border-left:5px solid #FFB300;">
<b>Why median?</b> Median is less affected by extreme values than the mean.
</div>

---

# 3. Duplicate Records

Duplicate records may occur because of repeated data entry or merging datasets.

```python
# Count how many rows are exact duplicates
df.duplicated().sum()
```

Remove exact duplicates:

```python
# Remove rows that are exact duplicates
# The cleaned DataFrame is stored back in 'df'
df = df.drop_duplicates()
```

**Important:** Do not remove duplicates blindly. In transaction data, two identical records may represent two valid transactions.

---


# 4. Data Visualization

**Data visualization** converts raw data into graphs so that patterns, trends, relationships, class differences, skewness, and unusual observations become easier to understand.

Visualization is an important part of EDA because statistical tables alone may hide important behavior in the data.

## Why do we visualize data?

Visualization helps us answer questions such as:

- Is a numerical variable normally distributed or skewed?
- Are extreme observations present?
- Which category occurs most frequently?
- Does a feature differ between churned and retained customers?
- Do two numerical features move together?
- Is the target class balanced?
- Are there clusters or unusual groups in the data?

<div style="background:#E8F4FD;padding:10px;border-left:5px solid #1976D2;">
<b>Professor's Rule:</b> Choose the graph according to the type of variable and the question you want to answer. A graph should reveal information, not merely decorate the analysis.
</div>

## Choosing the Correct Plot

| Question | Variable Type | Recommended Plot |
|---|---|---|
| What is the distribution? | One numerical | Histogram |
| Are there possible outliers? | One numerical | Boxplot |
| How frequent is each category? | One categorical | Bar plot |
| How do two numerical variables relate? | Numerical + Numerical | Scatter plot |
| How does a numerical feature differ across groups? | Numerical + Categorical | Boxplot |
| How do categories relate to the target? | Categorical + Target | Grouped/stacked bar |
| How are many numerical variables related? | Multiple numerical | Correlation heatmap |
| Is the target balanced? | Target category | Count/Bar plot |

---

## 4.1 Histogram — Understanding Distribution

A **histogram** divides numerical values into intervals called **bins** and displays how many observations fall into each interval.

It helps identify:

- Symmetric distributions
- Positive or negative skewness
- Multiple peaks
- Long tails
- Possible extreme values

```python
# Import matplotlib for creating graphs
import matplotlib.pyplot as plt

# Create a histogram of the Age column
# bins=20 divides the Age range into 20 intervals
df["Age"].hist(bins=20)

# Label the x-axis
plt.xlabel("Age")

# Label the y-axis
plt.ylabel("Frequency")

# Add a descriptive title
plt.title("Distribution of Customer Age")

# Display the graph
plt.show()
```

### How to interpret it

If most values occur on the left and a long tail extends toward larger values, the distribution is **positively skewed**.

For example, income variables are often positively skewed because most people have moderate income while a few observations have very high income.

### Compare several numerical features

```python
# Select multiple numerical features whose distributions we want to compare
df[
    ["Age", "TenureMonths", "MonthlyCharges",
     "InternetUsageGB", "AnnualIncome"]

# Create a separate histogram for each selected feature
# figsize controls the overall figure size and bins controls the number of intervals
].hist(figsize=(12, 8), bins=20)

# Adjust spacing automatically so labels do not overlap
plt.tight_layout()

# Display all histograms
plt.show()
```

This allows us to quickly compare the shape of several features.

---

## 4.2 Boxplot — Distribution and Outliers

A **boxplot** summarizes a numerical feature using quartiles.

It displays:

- Minimum non-outlier value
- Q1
- Median
- Q3
- Maximum non-outlier value
- Potential outliers

```python
# Create a figure of width 8 inches and height 3 inches
plt.figure(figsize=(8, 3))

# Draw a boxplot for MonthlyCharges
# dropna() removes missing values because the plot should use valid numerical values
# vert=False makes the boxplot horizontal
plt.boxplot(
    df["MonthlyCharges"].dropna(),
    vert=False
)

# Label the horizontal axis
plt.xlabel("Monthly Charges")

# Add a title to explain what the graph represents
plt.title("Boxplot of Monthly Charges")

# Display the boxplot
plt.show()
```

### Interpretation

Points lying far beyond the whiskers are potential outliers.

However:

> A boxplot detects unusual values; it does not prove that those values are incorrect.

---

## 4.3 Bar Plot — Categorical Variables

A **bar plot** compares frequencies or values across categories.

```python
# Count customers in each contract category
# plot(kind="bar") converts those counts into a bar chart
df["ContractType"].value_counts().plot(kind="bar")

# Label the x-axis with the categorical feature name
plt.xlabel("Contract Type")

# Label the y-axis with the frequency being displayed
plt.ylabel("Number of Customers")

# Add a suitable graph title
plt.title("Customers by Contract Type")

# Rotate category labels slightly so they are easier to read
plt.xticks(rotation=20)

# Display the bar chart
plt.show()
```

### What can we learn?

We can quickly identify:

- Most common category
- Least common category
- Imbalanced categorical features

---

## 4.4 Target-Class Visualization

Before classification, always inspect the target distribution.

```python
# Count observations in each Churn class
# sort_index() keeps class labels such as 0 and 1 in logical order
df["Churn"].value_counts().sort_index().plot(kind="bar")

# Label the target classes on the x-axis
plt.xlabel("Churn")

# Show how many customers belong to each class
plt.ylabel("Number of Customers")

# Add a descriptive title
plt.title("Churn Class Distribution")

# Keep class labels horizontal
plt.xticks(rotation=0)

# Display the class-distribution plot
plt.show()
```

If class `0` has many more observations than class `1`, the target is imbalanced.

---

## 4.5 Grouped Bar Plot — Category vs Target

We may want to know whether churn differs by contract type.

```python
# Build a frequency table between ContractType and Churn
# Rows represent contract types and columns represent churn classes
pd.crosstab(
    df["ContractType"],
    df["Churn"]

# Plot the cross-tabulation as grouped bars
).plot(kind="bar")

# Label the y-axis
plt.ylabel("Number of Customers")

# Add a title to describe the comparison
plt.title("Contract Type vs Churn")

# Rotate contract-type labels for readability
plt.xticks(rotation=20)

# Display the grouped bar chart
plt.show()
```

### Interpretation

If the proportion of churned customers is much larger for one contract type, that feature may contain useful predictive information.

A percentage table is even more informative:

```python
# Create a cross-tabulation between ContractType and Churn
contract_churn = pd.crosstab(
    df["ContractType"],
    df["Churn"],

    # normalize="index" converts counts within each contract type into proportions
    normalize="index"

# Multiply by 100 to convert proportions into percentages
) * 100

# Display the churn percentage for each contract type
print(contract_churn)
```

---

## 4.6 Scatter Plot — Relationship Between Numerical Features

A **scatter plot** represents each observation as one point.

```python
# Create a scatter plot to study the relationship between two numerical features
plt.scatter(
    # Values on the x-axis
    df["TenureMonths"],

    # Values on the y-axis
    df["TotalCharges"],

    # Make points slightly transparent so overlapping observations are visible
    alpha=0.5
)

# Label the axes
plt.xlabel("Tenure Months")
plt.ylabel("Total Charges")

# Add a title
plt.title("Tenure vs Total Charges")

# Display the scatter plot
plt.show()
```

Scatter plots can reveal:

- Positive relationships
- Negative relationships
- No clear relationship
- Clusters
- Non-linear behavior
- Outliers

### Add the target class

```python
# Loop through each unique churn class, for example 0 and 1
for churn_class in sorted(df["Churn"].unique()):

    # Keep only the rows belonging to the current churn class
    subset = df[df["Churn"] == churn_class]

    # Plot the current class on the same scatter plot
    plt.scatter(
        subset["TenureMonths"],
        subset["TotalCharges"],

        # Transparency helps when many points overlap
        alpha=0.5,

        # This label will appear in the legend
        label=f"Churn = {churn_class}"
    )

# Label the axes
plt.xlabel("Tenure Months")
plt.ylabel("Total Charges")

# Add a graph title
plt.title("Tenure vs Total Charges by Churn")

# Display a legend so we can identify each churn class
plt.legend()

# Show the final plot
plt.show()
```

This helps determine whether the two target classes occupy different regions of the feature space.

---

## 4.7 Numerical Feature vs Categorical Group

Boxplots can also compare a numerical feature across groups.

Example: monthly charges for churned and retained customers.

```python
# Extract MonthlyCharges for customers who did not churn
# dropna() removes missing MonthlyCharges values
group0 = df[df["Churn"] == 0]["MonthlyCharges"].dropna()

# Extract MonthlyCharges for customers who churned
group1 = df[df["Churn"] == 1]["MonthlyCharges"].dropna()

# Draw two boxplots side by side for comparison
plt.boxplot(
    [group0, group1],

    # Give meaningful names to the two groups
    tick_labels=["Retained", "Churned"]
)

# Label the y-axis
plt.ylabel("Monthly Charges")

# Add a title
plt.title("Monthly Charges by Churn Status")

# Display the comparison
plt.show()
```

### Interpretation

Compare:

- Median
- Spread
- Outliers
- Difference between groups

If the distributions differ strongly, the feature may be useful for prediction.

---

## 4.8 Correlation Heatmap

A **correlation heatmap** shows the correlation between several numerical features in one visual matrix.

Correlation values range from **-1 to +1**:

| Correlation Value | Meaning |
|---|---|
| `+1` | Perfect positive relationship |
| Close to `+1` | Strong positive relationship |
| Close to `0` | Little or no linear relationship |
| Close to `-1` | Strong negative relationship |
| `-1` | Perfect negative relationship |

The **color** of each cell gives a quick visual idea of the relationship, while the **number written inside the cell** gives the exact correlation value.

### Easy Example

Suppose we have three features:

- `TenureMonths`
- `MonthlyCharges`
- `TotalCharges`

A small correlation matrix may look like this:

| Feature | TenureMonths | MonthlyCharges | TotalCharges |
|---|---:|---:|---:|
| TenureMonths | 1.00 | 0.20 | 0.82 |
| MonthlyCharges | 0.20 | 1.00 | 0.55 |
| TotalCharges | 0.82 | 0.55 | 1.00 |

How do we read this?

- `TenureMonths` and `TotalCharges = 0.82`  
  This is a **strong positive correlation**. Customers who stay longer generally accumulate higher total charges.

- `MonthlyCharges` and `TotalCharges = 0.55`  
  This is a **moderate positive correlation**. Higher monthly charges may contribute to higher total charges.

- `TenureMonths` and `MonthlyCharges = 0.20`  
  This is a **weak positive correlation**. The relationship is present but not very strong.

- A value such as `-0.75` would indicate a **strong negative correlation**.  
  For example, if `Discount` increases while `FinalPrice` decreases, their correlation may be strongly negative.

- A value such as `0.02` would indicate **almost no linear relationship**.

> **Important:** A high correlation means that two variables move together, but it does not prove that one variable causes the other.

```python
# Import NumPy for numerical operations
import numpy as np

# Select only numerical columns and calculate their Pearson correlation matrix
corr = df.select_dtypes(include="number").corr()

# Create a figure large enough for all feature names
plt.figure(figsize=(10, 7))

# Display the correlation matrix as an image
# aspect="auto" lets matplotlib adjust the cell proportions
plt.imshow(corr, aspect="auto")

# Add a color scale showing correlation strength
plt.colorbar(label="Correlation")

# Put feature names on the x-axis
plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=45,
    ha="right"
)

# Put feature names on the y-axis
plt.yticks(
    range(len(corr.index)),
    corr.index
)

# Display correlation values inside each cell
for i in range(len(corr.index)):
    for j in range(len(corr.columns)):
        plt.text(
            j,
            i,
            f"{corr.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

# Add a title
plt.title("Correlation Matrix")

# Adjust spacing so labels fit properly
plt.tight_layout()

# Display the heatmap
plt.show()
```

### Understanding the Added Code

The following part writes the correlation number inside every heatmap cell:

```python
for i in range(len(corr.index)):
    for j in range(len(corr.columns)):
        plt.text(
            j,
            i,
            f"{corr.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )
```

Let us understand it step by step:

- `for i in range(len(corr.index)):`  
  Moves through each **row** of the correlation matrix.

- `for j in range(len(corr.columns)):`  
  Moves through each **column** of the correlation matrix.

- `corr.iloc[i, j]`  
  Reads the correlation value present at row `i` and column `j`.

- `f"{corr.iloc[i, j]:.2f}"`  
  Displays the correlation value up to **2 decimal places**.

  Example:

  ```text
  Original value = 0.823764
  Displayed value = 0.82
  ```

- `ha="center"`  
  Places the number at the horizontal center of the cell.

- `va="center"`  
  Places the number at the vertical center of the cell.

### How to Interpret the Final Heatmap

Use both the **color** and the **number**.

For example:

```text
0.91  → Strong positive correlation
0.67  → Moderate positive correlation
0.25  → Weak positive correlation
0.03  → Almost no linear correlation
-0.40 → Moderate negative correlation
-0.88 → Strong negative correlation
1.00  → A feature correlated with itself
```

The diagonal of the correlation matrix is always `1.00` because every feature is perfectly correlated with itself.

### Why is this useful in Machine Learning?

A correlation heatmap can help us:

- identify strongly related numerical features,
- detect possible redundant features,
- understand relationships before feature selection,
- identify features that may carry similar information,
- decide whether further investigation is required.

However, do **not** automatically delete a feature just because two features have high correlation. Also consider:

- domain meaning,
- model type,
- interpretability,
- validation performance.

Use this heatmap to identify strongly related numerical features before feature selection.

---

## 4.9 Visualizing Missing Values

Missing-value percentages can also be visualized.

```python
# Calculate the percentage of missing values in every column
missing_percentage = (
    df.isnull().mean() * 100

# Sort columns from highest to lowest missing percentage
).sort_values(ascending=False)

# Keep only columns that actually contain missing values
missing_percentage[
    missing_percentage > 0

# Plot their missing-value percentages as a bar chart
].plot(kind="bar")

# Label the y-axis
plt.ylabel("Missing Values (%)")

# Add a title
plt.title("Missing Values by Feature")

# Display the graph
plt.show()
```

This immediately identifies the columns requiring attention.

---

## 4.10 Visualization Checklist

Before moving to modeling, inspect:

1. Numerical distributions
2. Categorical frequencies
3. Target distribution
4. Potential outliers
5. Numerical relationships
6. Feature differences across target classes
7. Correlation among numerical features
8. Missing-value patterns

<div style="background:#E8F5E9;padding:10px;border-left:5px solid #2E7D32;">
<b>Key Takeaway:</b> Good visualization should lead to a preprocessing or modeling decision. For example, a skewed histogram may suggest transformation, a boxplot may trigger outlier investigation, and a class-count plot may indicate imbalance handling.
</div>

---


# 5. Correlation Analysis

**Correlation** measures the strength and direction of the relationship between two variables.

Correlation ranges from **-1 to +1**.

| Value | Meaning |
|---|---|
| +1 | Strong positive relationship |
| 0 | No linear relationship |
| -1 | Strong negative relationship |

Example:

If customer tenure increases and total charges also increase, the correlation is positive.

```python
# Select numerical columns and compute their Pearson correlation matrix
corr = df.select_dtypes(include="number").corr()

# Print the matrix so we can inspect relationships between numerical features
print(corr)
```

### Pearson Correlation

Used mainly for **linear relationships between numerical variables**.

**Linear relationship:** When one variable changes, the other changes at a roughly constant rate, forming a straight-line pattern.

Example:

```text
X: 1, 2, 3, 4
Y: 2, 4, 6, 8
```

Here, every increase of 1 in `X` increases `Y` by 2.

### Spearman Correlation

Used when the relationship is monotonic, ordinal, non-normal, or affected by outliers.

**Rank-based monotonic relationship:** It checks whether the order of values consistently increases or decreases, even if the rate of change is not constant.

Example:

```text
X: 1, 2, 3, 4
Y: 2, 5, 20, 100
```

Here, `Y` increases unevenly, but it always increases as `X` increases.

> **In short:** Pearson asks, “Is the relationship approximately a straight line?” Spearman asks, “As X increases, does Y generally keep increasing or decreasing?”

```python
# Compute Spearman correlation for numerical columns
# Spearman correlation is useful for monotonic or non-linear ranked relationships
df.select_dtypes(include="number").corr(method="spearman")
```

<div style="background:#FFEBEE;padding:10px;border-left:5px solid #D32F2F;">
<b>Remember:</b> Correlation does not prove causation.
</div>

---

# 6. Outlier Detection

An **outlier** is an observation that is unusually far from most other observations.

Examples:

- Age = 250
- Monthly bill = ₹5,00,000 when normal bills are around ₹1,000

Outliers may be:

- Data-entry errors
- Measurement errors
- Rare but genuine observations
- Fraud/anomalies

Therefore, an outlier should be **investigated before deletion**.

---

## IQR Method

The Interquartile Range is:

**IQR = Q3 − Q1**

Outlier limits:

- Lower = Q1 − 1.5 × IQR
- Upper = Q3 + 1.5 × IQR

```python
# Calculate the first quartile (25th percentile)
q1 = df["MonthlyCharges"].quantile(0.25)

# Calculate the third quartile (75th percentile)
q3 = df["MonthlyCharges"].quantile(0.75)

# Interquartile Range (IQR) measures the spread of the middle 50% of values
iqr = q3 - q1

# Calculate the lower boundary for possible outliers
lower = q1 - 1.5 * iqr

# Calculate the upper boundary for possible outliers
upper = q3 + 1.5 * iqr

# Select rows where MonthlyCharges lies outside the IQR boundaries
outliers = df[
    (df["MonthlyCharges"] < lower) |
    (df["MonthlyCharges"] > upper)
]

# Display the observations identified as potential outliers
print(outliers)
```

Possible treatments:

- Correct the value
- Remove invalid observations
- Cap extreme values
- Apply transformation

---

# 7. Feature Selection

**Feature selection** means keeping useful input features and removing irrelevant or redundant ones. It can reduce training time, noise, and overfitting.

**Main approaches:**
- **Filter:** statistical score, e.g. ANOVA/correlation
- **Wrapper:** tests feature subsets using a model, e.g. RFE
- **Embedded:** selection happens during training, e.g. L1 or tree importance

```python
# Select numerical predictors only for this short ANOVA example
from sklearn.feature_selection import SelectKBest, f_classif

X_fs = df.select_dtypes(include="number").drop(columns=["Churn"]).copy()
X_fs = X_fs.fillna(X_fs.median())   # ANOVA cannot use missing values
y = df["Churn"]

# Keep the 4 numerical features with the highest ANOVA F-scores
selector = SelectKBest(score_func=f_classif, k=4)
selector.fit(X_fs, y)

selected = X_fs.columns[selector.get_support()]
print("Selected features:", list(selected))
```

> **Idea:** A selected feature should finally be judged by validation performance, not only by its score.

---

# 8. Feature Transformation

**Feature transformation** changes the representation of a feature so that it is easier for a model to learn from it.

Common transformations include **scaling, log transformation, and categorical encoding**.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

# Standardize numerical features: approximately mean 0 and standard deviation 1
scaler = StandardScaler()
scaled_values = scaler.fit_transform(
    df[["Age", "MonthlyCharges"]].fillna(df[["Age", "MonthlyCharges"]].median())
)

# Log transformation can reduce strong positive skewness
log_income = np.log1p(df["AnnualIncome"].fillna(df["AnnualIncome"].median()))

# One-hot encode a nominal categorical feature
contract_encoded = pd.get_dummies(df["ContractType"], prefix="Contract")
```

> **Idea:** Scaling is useful for models such as KNN, SVM, Logistic Regression, PCA, and neural networks.

---

# 9. Data Imbalance Handling

A dataset is **imbalanced** when one target class has many more records than another.

For example:

```text
Churn = 0 → 900 customers
Churn = 1 → 100 customers
```

Here, 90% of the records belong to one class and only 10% belong to the other class.

### Why is it a problem?

A model may learn to predict the majority class most of the time.

For example, if it predicts every customer as `Churn = 0`, it may still get **90% accuracy**, but it completely fails to identify customers who actually churn.

So, for imbalanced datasets, **accuracy alone can be misleading**.

```python
# Count records in each target class
print(df["Churn"].value_counts())

# Check the proportion of each class
print(df["Churn"].value_counts(normalize=True))
```

### Common Solutions

- **Oversampling** → increase minority-class records.
- **Undersampling** → reduce majority-class records.
- **SMOTE** → create synthetic samples for the minority class.
- **Class weighting** → give more importance to the minority class during training.

```python
from sklearn.linear_model import LogisticRegression

# Give more importance to the minority class during model training
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)
```

`class_weight="balanced"` automatically gives more importance to the less frequent class.

```text
Imbalanced Data
      ↓
Model favors majority class
      ↓
Balance / give more importance to minority class
      ↓
Better learning for both classes
```

> **Important:** If SMOTE is used, apply it only to the training data after the train-test split.

Also consider **Precision, Recall, F1-score, and Confusion Matrix**, not only accuracy.

---

# 10. Data Preprocessing Pipeline

A **preprocessing pipeline** applies the same cleaning and transformation steps consistently to training and test data.

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Separate predictors and target; CustomerID is only an identifier
X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"]

# Split before learning preprocessing parameters to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_features = X.select_dtypes(include="number").columns
categorical_features = X.select_dtypes(exclude="number").columns

# Numerical data: fill missing values, then standardize
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical data: fill missing values, then one-hot encode
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Apply the correct pipeline to each column type
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features)
])

# Learn preprocessing only from training data
X_train_ready = preprocessor.fit_transform(X_train)
X_test_ready = preprocessor.transform(X_test)
```

> **Flow:** Missing-value treatment → Encoding → Scaling → Model


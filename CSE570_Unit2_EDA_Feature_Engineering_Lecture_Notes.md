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
import pandas as pd

df = pd.read_csv("CSE570_Unit2_Customer_Churn_Dataset.csv")

print(df.head())
print(df.shape)
df.info()
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
df.isnull().sum()
```

### Common treatments

- Numerical feature → mean or median
- Categorical feature → mode
- Large amount of missing data → consider removing the feature
- Missingness has meaning → create a separate category/indicator

```python
df["Age"] = df["Age"].fillna(df["Age"].median())
```

<div style="background:#FFF8E1;padding:10px;border-left:5px solid #FFB300;">
<b>Why median?</b> Median is less affected by extreme values than the mean.
</div>

---

# 3. Duplicate Records

Duplicate records may occur because of repeated data entry or merging datasets.

```python
df.duplicated().sum()
```

Remove exact duplicates:

```python
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
import matplotlib.pyplot as plt

df["Age"].hist(bins=20)

plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Distribution of Customer Age")
plt.show()
```

### How to interpret it

If most values occur on the left and a long tail extends toward larger values, the distribution is **positively skewed**.

For example, income variables are often positively skewed because most people have moderate income while a few observations have very high income.

### Compare several numerical features

```python
df[
    ["Age", "TenureMonths", "MonthlyCharges",
     "InternetUsageGB", "AnnualIncome"]
].hist(figsize=(12, 8), bins=20)

plt.tight_layout()
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
plt.figure(figsize=(8, 3))

plt.boxplot(
    df["MonthlyCharges"].dropna(),
    vert=False
)

plt.xlabel("Monthly Charges")
plt.title("Boxplot of Monthly Charges")
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
df["ContractType"].value_counts().plot(kind="bar")

plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.title("Customers by Contract Type")
plt.xticks(rotation=20)
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
df["Churn"].value_counts().sort_index().plot(kind="bar")

plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.title("Churn Class Distribution")
plt.xticks(rotation=0)
plt.show()
```

If class `0` has many more observations than class `1`, the target is imbalanced.

---

## 4.5 Grouped Bar Plot — Category vs Target

We may want to know whether churn differs by contract type.

```python
pd.crosstab(
    df["ContractType"],
    df["Churn"]
).plot(kind="bar")

plt.ylabel("Number of Customers")
plt.title("Contract Type vs Churn")
plt.xticks(rotation=20)
plt.show()
```

### Interpretation

If the proportion of churned customers is much larger for one contract type, that feature may contain useful predictive information.

A percentage table is even more informative:

```python
contract_churn = pd.crosstab(
    df["ContractType"],
    df["Churn"],
    normalize="index"
) * 100

print(contract_churn)
```

---

## 4.6 Scatter Plot — Relationship Between Numerical Features

A **scatter plot** represents each observation as one point.

```python
plt.scatter(
    df["TenureMonths"],
    df["TotalCharges"],
    alpha=0.5
)

plt.xlabel("Tenure Months")
plt.ylabel("Total Charges")
plt.title("Tenure vs Total Charges")
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
for churn_class in sorted(df["Churn"].unique()):

    subset = df[df["Churn"] == churn_class]

    plt.scatter(
        subset["TenureMonths"],
        subset["TotalCharges"],
        alpha=0.5,
        label=f"Churn = {churn_class}"
    )

plt.xlabel("Tenure Months")
plt.ylabel("Total Charges")
plt.title("Tenure vs Total Charges by Churn")
plt.legend()
plt.show()
```

This helps determine whether the two target classes occupy different regions of the feature space.

---

## 4.7 Numerical Feature vs Categorical Group

Boxplots can also compare a numerical feature across groups.

Example: monthly charges for churned and retained customers.

```python
group0 = df[df["Churn"] == 0]["MonthlyCharges"].dropna()
group1 = df[df["Churn"] == 1]["MonthlyCharges"].dropna()

plt.boxplot(
    [group0, group1],
    tick_labels=["Retained", "Churned"]
)

plt.ylabel("Monthly Charges")
plt.title("Monthly Charges by Churn Status")
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

A correlation matrix may contain many values. A heatmap makes the overall pattern easier to inspect.

```python
import numpy as np

corr = df.select_dtypes(include="number").corr()

plt.figure(figsize=(10, 7))
plt.imshow(corr, aspect="auto")
plt.colorbar(label="Correlation")

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(corr.index)),
    corr.index
)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
```

Use this to identify strongly related numerical features before feature selection.

---

## 4.9 Visualizing Missing Values

Missing-value percentages can also be visualized.

```python
missing_percentage = (
    df.isnull().mean() * 100
).sort_values(ascending=False)

missing_percentage[
    missing_percentage > 0
].plot(kind="bar")

plt.ylabel("Missing Values (%)")
plt.title("Missing Values by Feature")
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
corr = df.select_dtypes(include="number").corr()
print(corr)
```

### Pearson Correlation

Used mainly for **linear relationships between numerical variables**.

### Spearman Correlation

Used when the relationship is monotonic, ordinal, non-normal, or affected by outliers.

```python
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
q1 = df["MonthlyCharges"].quantile(0.25)
q3 = df["MonthlyCharges"].quantile(0.75)

iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outliers = df[
    (df["MonthlyCharges"] < lower) |
    (df["MonthlyCharges"] > upper)
]

print(outliers)
```

Possible treatments:

- Correct the value
- Remove invalid observations
- Cap extreme values
- Apply transformation

---


# 7. Feature Selection

**Feature selection** is the process of choosing the most relevant input variables and removing features that are irrelevant, redundant, noisy, or unnecessarily complex.

Suppose a dataset contains 100 variables but only 15 provide useful information about the target. Training a model on all 100 variables may increase complexity without improving prediction.

## Why is Feature Selection Important?

Feature selection can:

- Reduce model complexity
- Reduce training time
- Improve interpretability
- Reduce noise
- Reduce overfitting
- Remove redundant information
- Help some models generalize better

<div style="background:#FFF8E1;padding:10px;border-left:5px solid #FFB300;">
<b>Important distinction:</b> Feature selection keeps or removes existing features. Feature transformation changes features into a new representation.
</div>

Example:

- **Feature Selection:** Remove `CustomerID`
- **Feature Transformation:** Convert `AnnualIncome` into `log(AnnualIncome)`

---

## 7.1 Which Features Should We Consider Removing?

### 1. Identifier Features

Identifiers usually describe the record rather than the phenomenon being predicted.

Examples:

- Customer ID
- Roll number
- Transaction ID

```python
X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"]
```

`CustomerID` is unique but normally has no meaningful relationship with churn.

---

## 7.2 Constant and Near-Constant Features

A feature containing almost the same value for every observation contains little useful information.

```python
print(df.nunique())
```

A feature with only one unique value is constant.

Example:

```text
Country = India
Country = India
Country = India
...
```

Such a feature cannot help distinguish observations.

---

## 7.3 Correlation-Based Feature Selection

Highly correlated predictors may contain similar information.

```python
numeric_df = df.select_dtypes(include="number")

corr = numeric_df.corr().abs()

print(corr)
```

Example:

`TotalCharges` may be strongly related to:

- `MonthlyCharges`
- `TenureMonths`

because total charges depend partly on monthly payment and duration.

### Important

Do not automatically remove every correlated feature.

Consider:

- Domain meaning
- Model type
- Interpretability
- Predictive performance

---

# 7.4 Three Main Feature-Selection Approaches

There are three major families:

1. **Filter Methods**
2. **Wrapper Methods**
3. **Embedded Methods**

| Method | Main Idea | Speed |
|---|---|---|
| Filter | Statistical relationship with target | Fast |
| Wrapper | Evaluate subsets using a model | Slower |
| Embedded | Selection occurs during training | Moderate |

---

# 7.5 Filter Methods

Filter methods select features using statistical properties **before or independently of the final predictive model**.

Examples:

- Correlation
- Chi-square test
- ANOVA F-test
- Mutual Information
- Variance threshold

### Advantages

- Fast
- Easy to understand
- Suitable for high-dimensional datasets

### Limitation

They may not consider interactions between features.

---

## SelectKBest with ANOVA F-test

For numerical features and a categorical target:

```python
from sklearn.feature_selection import SelectKBest, f_classif

features = [
    "Age",
    "TenureMonths",
    "MonthlyCharges",
    "SupportCalls",
    "SatisfactionScore",
    "InternetUsageGB",
    "AnnualIncome"
]

X = df[features].copy()

X = X.fillna(X.median())

y = df["Churn"]

selector = SelectKBest(
    score_func=f_classif,
    k=4
)

selector.fit(X, y)
```

See selected features:

```python
selected_features = X.columns[
    selector.get_support()
]

print(selected_features)
```

See scores:

```python
import pandas as pd

scores = pd.DataFrame({
    "Feature": X.columns,
    "Score": selector.scores_,
    "Selected": selector.get_support()
})

print(
    scores.sort_values(
        "Score",
        ascending=False
    )
)
```

### Interpretation

A larger F-score indicates a stronger difference between target classes for that feature.

However, the score alone does not guarantee that the feature will improve every model.

---

## Mutual Information

Mutual Information can capture some non-linear dependency.

```python
from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(
    X,
    y,
    random_state=42
)

mi = pd.Series(
    mi_scores,
    index=X.columns
).sort_values(ascending=False)

print(mi)
```

### Difference from simple correlation

Correlation mainly measures linear association.

Mutual information can identify broader dependencies between a feature and the target.

---

# 7.6 Wrapper Methods

Wrapper methods select features by actually training a machine-learning model with different subsets.

They ask:

> Which subset of features produces the best model performance?

Examples:

- Recursive Feature Elimination (RFE)
- Sequential Forward Selection
- Sequential Backward Selection

### Advantages

- Considers the predictive model
- Can identify useful combinations of features

### Disadvantages

- More computationally expensive
- May overfit if not validated carefully

---

## Recursive Feature Elimination (RFE)

RFE repeatedly removes less important features until the required number remains.

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000
)

rfe = RFE(
    estimator=model,
    n_features_to_select=4
)

rfe.fit(X, y)
```

Check selection:

```python
rfe_result = pd.DataFrame({
    "Feature": X.columns,
    "Selected": rfe.support_,
    "Ranking": rfe.ranking_
})

print(rfe_result)
```

### Interpretation

- `Selected = True` → feature retained
- `Ranking = 1` → selected feature
- Higher rank → removed earlier

---

# 7.7 Embedded Methods

Embedded methods perform feature selection **during model training**.

Examples:

- Lasso Regression
- Logistic Regression with L1 penalty
- Decision Trees
- Random Forest
- Gradient Boosting

They provide a balance between filter and wrapper methods.

---

## Random Forest Feature Importance

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X, y)

importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(importance)
```

Visualize importance:

```python
importance.sort_values().plot(
    kind="barh"
)

plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.show()
```

### Interpretation

Features with larger importance contributed more strongly to the tree ensemble's decisions.

### Caution

Feature importance is model-dependent.

A feature considered important by Random Forest may not have the same importance in Logistic Regression.

---

# 7.8 L1 Regularization for Feature Selection

L1 regularization can force some coefficients toward exactly zero.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    max_iter=1000
)

model.fit(X, y)
```

Inspect coefficients:

```python
coefficients = pd.Series(
    model.coef_[0],
    index=X.columns
)

print(coefficients)
```

Features with coefficients equal or close to zero contribute little under this model.

---

# 7.9 Comparing Feature-Selection Methods

A useful classroom comparison is:

```text
Correlation / ANOVA
        ↓
     Filter

RFE + Logistic Regression
        ↓
     Wrapper

Random Forest / L1
        ↓
     Embedded
```

The selected features may differ because each method asks a different question.

### Filter asks:

> Is this feature statistically related to the target?

### Wrapper asks:

> Does this feature improve this specific model when used with other features?

### Embedded asks:

> How important is this feature during model training?

---

# 7.10 Feature Selection vs Feature Extraction

These terms should not be confused.

### Feature Selection

Keeps a subset of original features.

Example:

```text
Original:
Age, Income, Tenure, Charges, SupportCalls

Selected:
Tenure, Charges, SupportCalls
```

### Feature Extraction

Creates new features from existing features.

Example:

PCA transforms many original variables into new principal components.

PCA is covered later under dimensionality reduction.

---

# 7.11 Practical Feature-Selection Workflow

A sensible sequence is:

### Step 1
Remove identifiers and obviously irrelevant columns.

### Step 2
Check constant or near-constant features.

### Step 3
Study correlation and redundancy.

### Step 4
Apply a filter technique.

### Step 5
Apply a model-based technique.

### Step 6
Compare model performance before and after selection.

<div style="background:#E8F5E9;padding:10px;border-left:5px solid #2E7D32;">
<b>Most important rule:</b> The final decision should not be based only on a feature-selection score. Verify whether the selected feature subset improves validation performance and still makes sense for the problem domain.
</div>

---

## Quick Comparison

| Technique | Category | Useful When |
|---|---|---|
| Correlation | Filter | Removing redundant numerical variables |
| ANOVA F-test | Filter | Numerical input + categorical target |
| Mutual Information | Filter | Possible non-linear dependency |
| RFE | Wrapper | Selecting features for a specific model |
| L1 Regularization | Embedded | Sparse linear/logistic models |
| Random Forest Importance | Embedded | Tree-based feature ranking |

---


# 8. Feature Transformation

**Feature transformation** changes the representation or scale of a feature to make it more suitable for machine learning.

---

## Standardization

Transforms data so that approximately:

- Mean = 0
- Standard deviation = 1

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled = scaler.fit_transform(
    df[["Age", "MonthlyCharges"]].fillna(0)
)
```

Scaling is important for:

- KNN
- SVM
- Logistic Regression
- PCA
- Neural Networks

---

## Min-Max Scaling

Converts values usually to the range **0 to 1**.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[["Age", "MonthlyCharges"]].fillna(0))
```

---

## Log Transformation

Useful for highly positively skewed data such as salary or income.

```python
import numpy as np

df["LogIncome"] = np.log1p(df["AnnualIncome"])
```

It compresses very large values and can reduce skewness.

---

## Categorical Encoding

Machine-learning algorithms usually require numerical inputs.

For nominal categories, **One-Hot Encoding** is commonly used.

```python
pd.get_dummies(df["ContractType"])
```

Avoid assigning artificial order such as:

`North=1, South=2, East=3, West=4`

because it may incorrectly suggest that one category is greater than another.

---

# 9. Data Imbalance Handling

A dataset is **imbalanced** when one target class has many more samples than another.

Example:

- Class 0 = 90%
- Class 1 = 10%

A model predicting every case as Class 0 would obtain 90% accuracy but would completely fail to detect Class 1.

### Check class balance

```python
df["Churn"].value_counts()
df["Churn"].value_counts(normalize=True)
```

For imbalanced data, also consider:

- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Balanced Accuracy

---

## Common Techniques

### 1. Oversampling

Increase minority-class observations.

### 2. Undersampling

Reduce majority-class observations.

### 3. SMOTE

SMOTE creates synthetic minority-class samples.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

### 4. Class Weighting

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)
```

<div style="background:#FFEBEE;padding:10px;border-left:5px solid #D32F2F;">
<b>Important:</b> Apply SMOTE only to training data, not before train-test splitting. Otherwise, data leakage may occur.
</div>

---

# 10. Data Preprocessing Pipeline

A **pipeline** combines multiple preprocessing steps into one workflow.

Typical flow:

**Missing-value treatment → Encoding → Scaling → Model**

Why use a pipeline?

- Prevents data leakage
- Makes code reusable
- Ensures the same preprocessing is applied to training and testing data
- Makes model deployment easier

---

## Numerical Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
```

## Categorical Pipeline

```python
from sklearn.preprocessing import OneHotEncoder

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
```

## Combine Both

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])
```

---

# 11. Data Leakage

**Data leakage** occurs when information from the test set indirectly enters model training.

### Wrong approach

```python
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
```

The scaler has already learned from the complete dataset.

### Better approach

Split first and use a pipeline.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Then:

```python
pipeline.fit(X_train, y_train)
pipeline.predict(X_test)
```

---

# 12. Complete Unit-II Workflow

<div style="background:#E8F5E9;padding:12px;border-left:5px solid #2E7D32;">
<b>Recommended Workflow</b><br><br>
Load Dataset → Understand Data → Check Missing Values → Check Duplicates → Visualize Data → Correlation Analysis → Detect Outliers → Select Features → Transform Features → Check Class Imbalance → Build Preprocessing Pipeline → Train Model
</div>

---

# 13. Quick Revision

| Topic | Purpose |
|---|---|
| EDA | Understand the dataset |
| Visualization | Discover patterns |
| Correlation | Study relationships |
| Outlier Detection | Identify unusual observations |
| Feature Selection | Keep important variables |
| Feature Transformation | Improve feature representation |
| Imbalance Handling | Improve minority-class learning |
| Pipeline | Automate safe preprocessing |

---

## Important Questions

1. What is EDA and why is it required?
2. Differentiate Pearson and Spearman correlation.
3. Explain the IQR method for outlier detection.
4. What is feature selection?
5. Differentiate filter, wrapper, and embedded methods.
6. Why is scaling required?
7. What is class imbalance?
8. Why can accuracy be misleading for imbalanced datasets?
9. What is SMOTE?
10. What is a preprocessing pipeline?
11. What is data leakage?
12. Why should preprocessing be fitted only on training data?

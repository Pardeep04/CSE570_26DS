# ⚙️ CSE570: Machine Learning with Python

## Unit I — Part 3: Encoding, Scaling, Train-Test Split, and ML Pipeline

> **Course focus:** converting cleaned data into a machine-learning-ready form.

---

## 📚 Unit I Navigation

| Part | Topic | File |
|---|---|---|
| **1** | Python Fundamentals and NumPy | [Open Part 1](CSE570_Unit_1_Part_1_Python_and_NumPy.md) |
| **2** | Pandas, Data Loading, and Missing Values | [Open Part 2](CSE570_Unit_1_Part_2_Pandas_and_Data_Cleaning.md) |
| **3** | Encoding, Scaling, Splitting, and ML Pipeline | **You are here** |

---

## 🎯 Learning Outcomes

After completing this file, students will be able to:

- distinguish numerical, nominal, ordinal, and target variables;
- encode categorical variables using suitable methods;
- standardise or normalise numerical features;
- divide data into training and testing subsets;
- explain and prevent data leakage;
- create a reusable preprocessing pipeline using Scikit-Learn.

---

<table><tr><td bgcolor="#E8F4FD">
<b>💡 Why preprocessing matters</b><br>
Machine learning algorithms generally require consistent numerical input. Encoding converts categories into numbers, scaling makes numerical ranges comparable, and train-test splitting allows us to evaluate performance on unseen data.
</td></tr></table>

# 1. Preparing a Sample Dataset

Before learning preprocessing techniques, we first create a small student-placement dataset. Each **row** represents one student, while each **column** represents a characteristic of that student. For example, `Age` and `StudyHours` are numerical features, whereas `Gender`, `City`, and `Education` are categorical features. `Placed` is the **target variable** because it is the outcome that a machine-learning model would try to predict.

This dataset intentionally contains different kinds of variables so that we can understand why a single preprocessing method cannot be applied to every column. Numerical columns may need scaling, unordered categorical columns may need one-hot encoding, and ordered categorical columns need an encoding that preserves their natural sequence.

```python
import numpy as np
import pandas as pd

student_data = {
    "Age": [20, 21, 22, 20, 23, 21],
    "StudyHours": [2.5, 4.0, 5.5, 3.0, 6.0, 4.5],
    "Gender": ["Female", "Male", "Female", "Male", "Female", "Male"],
    "City": ["Delhi", "Mumbai", "Delhi", "Chandigarh", "Mumbai", "Delhi"],
    "Education": [
        "School", "Undergraduate", "Postgraduate",
        "Undergraduate", "Postgraduate", "School"
    ],
    "Placed": ["No", "Yes", "Yes", "No", "Yes", "Yes"]
}

df = pd.DataFrame(student_data)
print(df)
```

---

# 2. Encoding Categorical Variables

Most algorithms cannot directly process words such as `Delhi`, `Female`, or `Postgraduate`. Encoding converts categories into numerical representations.

## 2.1 Understanding Category Types

A **categorical variable** stores labels or groups rather than measured quantities. Before encoding such a variable, we must decide whether its categories have a meaningful order. This decision is important because an unsuitable encoding can introduce a relationship that does not exist in the original data.

| Type | Description | Example |
|---|---|---|
| Binary nominal | Two categories without order | Yes/No |
| Multi-category nominal | Several categories without order | Delhi/Mumbai/Chandigarh |
| Ordinal | Categories with meaningful order | Low/Medium/High |
| Target class | Output to be predicted | Placed/Not Placed |

For example, `Delhi`, `Mumbai`, and `Chandigarh` are names of cities; none is naturally higher or lower than another. They are therefore **nominal** categories. In contrast, `School`, `Undergraduate`, and `Postgraduate` describe increasing education levels, so they are **ordinal** categories. The target class is treated separately because it represents the answer the model must learn to predict.

---

## 2.2 Binary Mapping

Binary values can be mapped directly to `0` and `1`.

Binary mapping is suitable when a variable has exactly two possible categories. It replaces one category with `0` and the other with `1`, producing a single numerical column. This is compact and easy for a model to process. The choice of which category receives `0` or `1` usually does not change the meaning, provided that the mapping is documented and applied consistently to future data.

```python
# Work on a copy so that the original data remains unchanged.
df_mapped = df.copy()

# Convert Gender into a numerical binary feature.
df_mapped["Gender"] = df_mapped["Gender"].map({
    "Female": 0,
    "Male": 1
})

# Convert the target class into numerical labels.
df_mapped["Placed"] = df_mapped["Placed"].map({
    "No": 0,
    "Yes": 1
})

print(df_mapped[["Gender", "Placed"]])
```

<table><tr><td bgcolor="#FFF4CE">
<b>⚠️ Interpretation</b><br>
The codes are identifiers only. Encoding Female as 0 and Male as 1 does not mean one category is better or greater.
</td></tr></table>

---

## 2.3 Label Encoding

`LabelEncoder` assigns a unique integer to each category.

During `fit_transform()`, the encoder first discovers all unique labels, stores them in `classes_`, and then replaces each label with its integer position. This method is commonly used for **target labels**, such as converting `Fail` and `Pass` into `0` and `1`. It should be used cautiously for nominal input columns because numbers can appear to imply distance or order even when none exists.

```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

df_label = df.copy()

# Learn city categories and convert them into integer codes.
df_label["CityCode"] = label_encoder.fit_transform(df_label["City"])

print(df_label[["City", "CityCode"]])
print("Learned categories:", label_encoder.classes_)
```

<table><tr><td bgcolor="#FDECEC">
<b>❌ Limitation</b><br>
For nominal input features, integer codes may create an artificial order. A model could incorrectly interpret CityCode 2 as greater than CityCode 1. One-hot encoding is generally safer for such input features.
</td></tr></table>

---

## 2.4 Ordinal Encoding

Ordinal categories have a meaningful sequence.

Ordinal encoding converts ordered categories into numbers while preserving their ranking. Here, `School → 0`, `Undergraduate → 1`, and `Postgraduate → 2` express educational progression. The numerical gaps do not necessarily mean that the difference between every pair of levels is exactly equal; the numbers primarily communicate order to the model.

```python
education_order = {
    "School": 0,
    "Undergraduate": 1,
    "Postgraduate": 2
}

df_ordinal = df.copy()

# Preserve the natural educational progression.
df_ordinal["Education"] = df_ordinal["Education"].map(
    education_order
)

print(df_ordinal[["Education"]])
```

---

## 2.5 One-Hot Encoding with Pandas

One-hot encoding creates a separate indicator column for each category.

Suppose `City` contains Delhi, Mumbai, and Chandigarh. One-hot encoding creates three new columns—one for each city. A row receives `1` in the column matching its city and `0` in the other city columns. Because the categories are represented independently, the model is not told that one city is greater than another.

When `drop_first=True` is used, one generated column is removed and treated as a **reference category**. Its value can be inferred when all remaining indicator columns contain `0`. This may reduce redundancy in some linear models, although keeping all columns is often simpler when the encoder is used inside a general preprocessing pipeline.

```python
# Create one binary column for every City category.
df_one_hot = pd.get_dummies(
    df,
    columns=["City"],
    dtype=int
)

print(df_one_hot)
```

Using `drop_first=True`:

```python
# Remove one reference category to reduce redundant columns.
df_one_hot_reduced = pd.get_dummies(
    df,
    columns=["City"],
    drop_first=True,
    dtype=int
)

print(df_one_hot_reduced)
```

---

## 2.6 One-Hot Encoding with Scikit-Learn

Scikit-Learn's `OneHotEncoder` is especially useful in a production or pipeline workflow because it learns category names from the training data and applies the same column structure to later data. The option `handle_unknown="ignore"` prevents an error if a new category appears during testing or prediction; that unknown category is represented by zeros across the learned category columns.

The encoder expects a two-dimensional input, which is why `df[["City"]]` uses double brackets. The method `get_feature_names_out()` then provides meaningful names for the newly generated columns.

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    # Ignore a category that appears later in test data.
    handle_unknown="ignore",

    # Return a regular NumPy array for easy demonstration.
    sparse_output=False
)

# The input remains two-dimensional: DataFrame with one column.
city_encoded = encoder.fit_transform(df[["City"]])

# Create meaningful output column names.
city_columns = encoder.get_feature_names_out(["City"])

city_df = pd.DataFrame(
    city_encoded,
    columns=city_columns
)

print(city_df)
```

### Choosing an encoding method

| Variable type | Recommended method |
|---|---|
| Binary nominal feature | Mapping or one-hot encoding |
| Multi-category nominal feature | One-hot encoding |
| Ordered category | Ordered mapping or `OrdinalEncoder` |
| Target label | Mapping or `LabelEncoder` |

---

# 3. Feature Scaling

Features may have very different ranges. For example, age may range from 18 to 60 while annual income may range from 200,000 to 2,000,000. Without scaling, large-valued features may dominate some algorithms.

Scaling is particularly important for:

- K-Nearest Neighbours;
- Support Vector Machines;
- K-Means clustering;
- Principal Component Analysis;
- Logistic Regression;
- regularised regression;
- neural networks.

Decision Trees and Random Forests usually do not require scaling.

Scaling changes the numerical range of a feature but does not change the order of its observations. It becomes important for algorithms that calculate **distances**, **gradients**, or **weighted combinations**. For example, if income is measured in lakhs while age is measured in years, an unscaled distance calculation may be influenced mainly by income simply because its numbers are larger. Tree-based algorithms instead make threshold decisions such as `Age < 25`, so they are generally unaffected by differences in measurement scale.

---

## 3.1 Standardisation

Standardisation centres a feature around zero and scales it using its standard deviation.

For each value, the feature mean is subtracted and the result is divided by the feature's standard deviation. A standardised value of `0` is at the mean, a value near `+1` is approximately one standard deviation above the mean, and a value near `-1` is approximately one standard deviation below it. Standardisation does not force values into a fixed interval, so values may still be greater than `1` or less than `-1`.

```text
z = (x - mean) / standard_deviation
```

```python
from sklearn.preprocessing import StandardScaler

sample = pd.DataFrame({
    "Age": [20, 25, 30, 35, 40],
    "Income": [250000, 350000, 500000, 700000, 900000]
})

standard_scaler = StandardScaler()

# Learn the mean and standard deviation, then transform the values.
scaled_values = standard_scaler.fit_transform(sample)

# Convert the NumPy result back into a labelled DataFrame.
standardised_df = pd.DataFrame(
    scaled_values,
    columns=sample.columns
)

print(standardised_df.round(3))
```

<table><tr><td bgcolor="#EAF7EA">
<b>✅ Result</b><br>
After standardisation, each feature has a mean close to 0 and a standard deviation close to 1.
</td></tr></table>

---

## 3.2 Min-Max Normalisation

Min-Max scaling usually converts values to the range `[0, 1]`.

The minimum observed training value becomes `0`, the maximum becomes `1`, and other values are placed proportionally between them. For instance, if the minimum is 20, the maximum is 40, and a value is 30, its scaled value is `(30−20)/(40−20) = 0.5`. This method is useful when features need a bounded range, but an extreme minimum or maximum can compress most ordinary observations into a small part of that range.

```text
x_scaled = (x - minimum) / (maximum - minimum)
```

```python
from sklearn.preprocessing import MinMaxScaler

minmax_scaler = MinMaxScaler()

# Learn each column's minimum and maximum, then scale it.
normalised_values = minmax_scaler.fit_transform(sample)

normalised_df = pd.DataFrame(
    normalised_values,
    columns=sample.columns
)

print(normalised_df)
```

---

## 3.3 Standardisation versus Normalisation

Neither method is universally best. `StandardScaler` is a strong general choice for many linear, distance-based, and gradient-based models. `MinMaxScaler` is useful when a known bounded range is desirable. The selected scaler must be learned from the training data only and then reused without refitting on validation, test, or future data.

| StandardScaler | MinMaxScaler |
|---|---|
| Mean approximately 0 | Values usually between 0 and 1 |
| Standard deviation approximately 1 | Based on minimum and maximum |
| Common general-purpose choice | Useful when a fixed range is preferred |
| Affected by extreme values | Also strongly affected by extreme values |

---

# 4. Train-Test Split

A machine learning model should be trained on one portion of the data and evaluated on a separate unseen portion.

- **Training set:** used to learn model patterns.
- **Testing set:** used to estimate performance on unseen data.

The purpose of the split is to imitate a real application: the model learns from past examples and is then asked to make predictions for examples it has never seen. If the same rows are used for both training and testing, the resulting score may measure memorisation rather than the model's ability to generalise.

A common split is 80% training and 20% testing, but the appropriate ratio depends on dataset size and the evaluation plan. The test set must be kept untouched until the final evaluation. In more advanced experiments, a separate validation set or cross-validation is also used for model selection and hyperparameter tuning.

## 4.1 Separating Features and Target

In supervised learning, `X` conventionally represents the **feature matrix** and `y` represents the **target vector**. `X` contains the information available to the model, while `y` contains the correct answers. The target must be removed from `X`; otherwise, the model would receive the answer as an input, causing severe target leakage.

```python
# X contains input features.
X = df.drop(columns=["Placed"])

# y contains the output that the model should predict.
y = df["Placed"]

print("Feature shape:", X.shape)
print("Target shape:", y.shape)
```

---

## 4.2 Performing the Split

`train_test_split()` shuffles and divides corresponding rows of `X` and `y` together, ensuring that every feature row remains paired with its correct target. `test_size=0.20` assigns 20% of the observations to testing. `random_state=42` fixes the random selection so that the same experiment can be reproduced, while `stratify=y` aims to preserve the target-class proportions.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,

    # Reserve 20% of rows for testing.
    test_size=0.20,

    # Make the split reproducible.
    random_state=42,

    # Preserve the target-class proportion in both subsets.
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
```

<table><tr><td bgcolor="#E8F4FD">
<b>💡 Why random_state is used</b><br>
The same value produces the same split every time, making classroom demonstrations and experiments reproducible.
</td></tr></table>

---

## 4.3 Why Stratification Matters

Suppose a dataset contains 90 `Yes` cases and 10 `No` cases. A random split might place too few `No` cases in the test set. `stratify=y` helps preserve the original class ratio.

Without sufficient examples of every class, evaluation can be misleading. A test set containing only majority-class observations cannot properly reveal whether the model recognises the minority class. Stratification is therefore particularly valuable for imbalanced classification datasets. However, each class must contain enough observations to be represented in the selected subsets.

```python
print("Original distribution:")
print(y.value_counts(normalize=True))

print("Training distribution:")
print(y_train.value_counts(normalize=True))

print("Testing distribution:")
print(y_test.value_counts(normalize=True))
```

---

# 5. Data Leakage

**Data leakage** occurs when information from the test data influences training. It leads to unrealistically high performance.

Leakage does not always mean that the target column was copied directly into the inputs. It can happen indirectly when preprocessing statistics—such as a mean, median, minimum, maximum, standard deviation, or category list—are calculated using the complete dataset. Even though a scaler does not read the target, fitting it on the test rows allows information about the test distribution to influence training.

The difference between the main transformation methods is:

- `fit()` learns parameters, such as means, medians, scales, or categories;
- `transform()` applies already learned parameters to data;
- `fit_transform()` learns parameters and immediately applies them to the same data.

Therefore, training data may use `fit_transform()`, but test data must use only `transform()`.

### Incorrect approach

The following short comparison focuses only on the order of splitting and scaling. It assumes that `X` is already numerical; in the mixed student dataset, categorical columns would also need encoding, as demonstrated later through `ColumnTransformer`.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Incorrect: the scaler learns from the complete dataset,
# including information from rows intended for testing.
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42
)
```

### Correct approach

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

# Learn scaling values from training data only.
X_train_scaled = scaler.fit_transform(X_train)

# Apply the already learned transformation to test data.
X_test_scaled = scaler.transform(X_test)
```

<table><tr><td bgcolor="#FDECEC">
<b>❌ Golden rule</b><br>
Never use <code>fit()</code> or <code>fit_transform()</code> on the test set. The test set must remain unseen during learning.
</td></tr></table>

---

# 6. Complete Preprocessing Pipeline

A pipeline combines multiple preprocessing steps into one reusable workflow.

Think of a pipeline as an assembly line. Raw data enters at one end and passes through a fixed sequence of operations. Numerical columns may be imputed and scaled, categorical columns may be imputed and encoded, and the prepared features may finally be given to a model. The same learned assembly line is then reused during prediction, ensuring consistent treatment of every new record.

## 6.1 Why Use a Pipeline?

A pipeline:

- applies transformations in the correct order;
- prevents accidental leakage;
- keeps training and prediction consistent;
- simplifies cross-validation and model deployment;
- makes code easier to maintain.

---

## 6.2 Creating a Dataset with Missing Values

This example includes missing numerical and categorical entries to represent a realistic raw dataset. Missing values cannot simply pass through most machine-learning algorithms, so they are handled inside the pipeline. Keeping imputation inside the pipeline ensures that replacement values are learned only from the training subset.

```python
import numpy as np
import pandas as pd

raw_data = {
    "Age": [20, 21, np.nan, 22, 23, 20, 24, 21],
    "StudyHours": [2.5, 4.0, 5.5, np.nan, 6.0, 3.0, 7.0, 4.5],
    "Gender": ["Female", "Male", "Female", "Male", None, "Male", "Female", "Male"],
    "City": ["Delhi", "Mumbai", "Delhi", "Chandigarh", "Mumbai", "Delhi", None, "Mumbai"],
    "Placed": [0, 1, 1, 0, 1, 0, 1, 1]
}

df = pd.DataFrame(raw_data)

X = df.drop(columns=["Placed"])
y = df["Placed"]
```

---

## 6.3 Separate Numerical and Categorical Columns

Different data types require different transformations. The two numerical columns can be imputed with a numerical statistic and then standardised. The categorical columns instead require a categorical imputation strategy followed by encoding. Explicit column lists make this routing clear and prevent an inappropriate transformation from being applied to the wrong data type.

```python
numeric_features = ["Age", "StudyHours"]
categorical_features = ["Gender", "City"]
```

---

## 6.4 Numerical Pipeline

The numerical pipeline performs its steps in order. First, the median learned from the training data replaces each missing numerical value. The median is often preferred when a feature may contain extreme values because it is less influenced by outliers than the mean. Second, the completed numerical columns are standardised so that their scales are comparable.

```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

numeric_pipeline = Pipeline(steps=[
    # Step 1: Replace missing numerical values with the median.
    ("imputer", SimpleImputer(strategy="median")),

    # Step 2: Standardise the completed numerical values.
    ("scaler", StandardScaler())
])
```

---

## 6.5 Categorical Pipeline

The categorical pipeline first replaces a missing category with the most frequently occurring training category. It then one-hot encodes the completed categories. `handle_unknown="ignore"` allows the pipeline to process a category not encountered during training without changing the learned output structure or stopping with an error.

```python
from sklearn.preprocessing import OneHotEncoder

categorical_pipeline = Pipeline(steps=[
    # Step 1: Replace missing categories with the most frequent value.
    ("imputer", SimpleImputer(strategy="most_frequent")),

    # Step 2: Convert categories into one-hot columns.
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
```

---

## 6.6 Combining Pipelines with ColumnTransformer

`ColumnTransformer` acts as a dispatcher. It sends `Age` and `StudyHours` through the numerical pipeline and sends `Gender` and `City` through the categorical pipeline. The transformed results are then joined side by side into one machine-learning-ready feature matrix. This makes it possible to preprocess a mixed-type DataFrame through one coordinated object.

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers=[
    # Apply the numerical pipeline only to numerical columns.
    ("numeric", numeric_pipeline, numeric_features),

    # Apply the categorical pipeline only to categorical columns.
    ("categorical", categorical_pipeline, categorical_features)
])
```

---

## 6.7 Splitting and Transforming Data

The data is split before the preprocessor is fitted. Calling `preprocessor.fit_transform(X_train)` learns all required training-only information: numerical medians, scaling statistics, frequent categories, and one-hot category names. Calling `preprocessor.transform(X_test)` reuses those stored values. The number of output columns may be larger than the number of original columns because one categorical feature can expand into several one-hot columns.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Learn imputation, scaling, and encoding from training data.
X_train_ready = preprocessor.fit_transform(X_train)

# Apply the learned transformations to test data.
X_test_ready = preprocessor.transform(X_test)

print("Prepared training shape:", X_train_ready.shape)
print("Prepared testing shape:", X_test_ready.shape)
```

---

## 6.8 Adding a Machine Learning Model

Although model training is covered in later units, this example shows how preprocessing and a model can be connected.

The outer pipeline joins preprocessing and classification into a single estimator. When `fit()` is called, the training data is first transformed and the classifier is then trained on the transformed output. When `predict()` is called, the stored preprocessing steps are automatically applied before the classifier generates predictions. This reduces manual code and prevents the training and prediction workflows from becoming inconsistent.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

complete_model = Pipeline(steps=[
    # First prepare the raw columns.
    ("preprocessing", preprocessor),

    # Then train Logistic Regression on prepared features.
    ("classifier", LogisticRegression(max_iter=1000))
])

# The pipeline receives the original, unprocessed DataFrame.
complete_model.fit(X_train, y_train)

# The same preprocessing is automatically applied before prediction.
predictions = complete_model.predict(X_test)

print("Predictions:", predictions)
print("Actual values:", y_test.to_numpy())
```

<table><tr><td bgcolor="#EAF7EA">
<b>✅ Important benefit</b><br>
During prediction, the pipeline automatically uses the same median values, scaling parameters, and category columns learned from the training data.
</td></tr></table>

---

# 7. Common Mistakes and Best Practices

| Common mistake | Better practice |
|---|---|
| Encoding categories before splitting without care | Use an encoder inside a pipeline |
| Scaling the complete dataset | Fit the scaler on training data only |
| Calling `fit_transform()` on test data | Call only `transform()` on test data |
| Using LabelEncoder for nominal input features | Prefer one-hot encoding |
| Dropping all incomplete rows automatically | Examine missingness and choose a justified method |
| Modifying the original dataset repeatedly | Use `.copy()` during exploration |
| Ignoring unseen categories | Use `handle_unknown="ignore"` |
| Omitting `random_state` | Set it for reproducible experiments |
| Ignoring class imbalance during splitting | Use `stratify=y` for classification |

---

# 8. End-to-End Student Example

This example brings all previous ideas together. Follow it as a flow: create raw data, separate `X` and `y`, split the observations, define feature groups, build type-specific pipelines, combine them, attach a classifier, and finally train and predict. Notice that the model receives the original DataFrame; all preprocessing occurs safely inside the combined pipeline.

```python
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------
# Step 1: Create a small raw dataset.
# ---------------------------------------------------------
data = {
    "Age": [20, 22, 21, np.nan, 24, 23, 20, 25, 22, 21],
    "StudyHours": [2, 5, 3, 4, 7, 6, np.nan, 8, 5, 3],
    "City": [
        "Delhi", "Mumbai", "Delhi", "Chandigarh", "Mumbai",
        "Delhi", "Chandigarh", "Mumbai", None, "Delhi"
    ],
    "Placed": [0, 1, 0, 1, 1, 1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

# ---------------------------------------------------------
# Step 2: Separate input features and target.
# ---------------------------------------------------------
X = df.drop(columns=["Placed"])
y = df["Placed"]

# ---------------------------------------------------------
# Step 3: Split before learning preprocessing parameters.
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------------
# Step 4: Define column groups.
# ---------------------------------------------------------
numeric_features = ["Age", "StudyHours"]
categorical_features = ["City"]

# ---------------------------------------------------------
# Step 5: Build preprocessing for each column type.
# ---------------------------------------------------------
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# ---------------------------------------------------------
# Step 6: Connect preprocessing and classification.
# ---------------------------------------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# ---------------------------------------------------------
# Step 7: Train and predict.
# ---------------------------------------------------------
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Predicted classes:", predictions)
print("Actual classes:", y_test.to_numpy())
```

---

# 9. Practice Exercises

1. Create a dataset containing age, salary, department, experience level, and promotion status.
2. Encode promotion status using binary mapping.
3. Encode department using one-hot encoding.
4. Encode experience level using the order `Beginner < Intermediate < Advanced`.
5. Standardise age and salary.
6. Apply Min-Max scaling and compare the resulting values.
7. Split the dataset into 75% training and 25% testing data.
8. Use stratification when splitting a classification target.
9. Demonstrate one incorrect leakage-prone approach and correct it.
10. Build a `ColumnTransformer` for numerical and categorical columns.
11. Create a complete pipeline using imputation, scaling, encoding, and Logistic Regression.
12. Explain why the test set must not be used during `fit()`.

---

# 10. Unit I Summary

| Stage | Purpose | Common tool |
|---|---|---|
| Python fundamentals | Write data-processing logic | Core Python |
| Numerical arrays | Efficient numerical operations | NumPy |
| Tabular data | Inspect and manipulate datasets | Pandas |
| Missing-value handling | Complete or remove incomplete data | Pandas / `SimpleImputer` |
| Encoding | Convert categories into numbers | `OneHotEncoder` / mapping |
| Scaling | Make feature ranges comparable | `StandardScaler` / `MinMaxScaler` |
| Train-test split | Evaluate on unseen data | `train_test_split` |
| Combined workflow | Apply safe, repeatable preprocessing | `Pipeline` / `ColumnTransformer` |

---

## ✅ Unit I Completion Checklist

- [ ] I can write basic Python programs.
- [ ] I can create and manipulate NumPy arrays.
- [ ] I can create and inspect Pandas DataFrames.
- [ ] I can load CSV, Excel, and JSON files.
- [ ] I can detect and handle missing values.
- [ ] I can select an appropriate encoding technique.
- [ ] I can standardise and normalise numerical features.
- [ ] I can create a stratified train-test split.
- [ ] I understand data leakage.
- [ ] I can build a preprocessing pipeline.

---

## ⬅️ Previous

Return to [Part 2: Pandas, Data Loading, and Missing Values](CSE570_Unit_1_Part_2_Pandas_and_Data_Cleaning.md).

# CSE570: Machine Learning with Python

## Unit I: Python Fundamentals and Data Preparation for Machine Learning

### Part 3: Encoding, Scaling, Splitting, and ML Pipeline

---

## Unit I File Structure

1. [Part 1: Python Fundamentals and NumPy](CSE570_Unit_1_Part_1_Python_and_NumPy.md)
2. [Part 2: Pandas, Data Loading, and Missing Values](CSE570_Unit_1_Part_2_Pandas_and_Data_Cleaning.md)
3. [Part 3: Encoding, Scaling, Splitting, and ML Pipeline](CSE570_Unit_1_Part_3_Preprocessing_and_Pipeline.md)

---

## Contents

- [10. Encoding Categorical Variables](#10-encoding-categorical-variables)
- [11. Feature Scaling](#11-feature-scaling)
- [12. Train-Test Split](#12-traintest-split)
- [13. Complete Data-Preparation Pipeline](#13-complete-datapreparation-pipeline)
- [14. Common Mistakes and Best Practices](#14-common-mistakes-and-best-practices)
- [15. Practice Exercises](#15-practice-exercises)
- [16. Summary](#16-summary)

---

## 10. Encoding Categorical Variables

Most machine learning algorithms require numerical inputs. Categorical values such as `Male`, `Female`, `Delhi`, and `Mumbai` must therefore be encoded.

### 10.1 Binary Mapping

Binary categories can be mapped directly.

```python
df_encoded = df.copy()

df_encoded["Gender"] = df_encoded["Gender"].map({
    "Female": 0,
    "Male": 1
})

df_encoded["Placed"] = df_encoded["Placed"].map({
    "No": 0,
    "Yes": 1
})

print(df_encoded)
```

### 10.2 Label Encoding

Label encoding assigns an integer to each category.

```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df_encoded["CityLabel"] = label_encoder.fit_transform(df_encoded["City"])

print(df_encoded[["City", "CityLabel"]])
print(label_encoder.classes_)
```

Potential issue: label encoding may introduce an artificial order. For example, the values `Delhi = 0`, `Mumbai = 2`, and `Chandigarh = 1` do not mean that one city is greater than another.

Label encoding is generally more suitable for:

- Binary variables
- Ordinal categories
- Target labels

### 10.3 Ordinal Encoding

Ordinal categories have a meaningful order.

```python
education_level = pd.Series([
    "School",
    "Undergraduate",
    "Postgraduate",
    "Undergraduate"
])

education_mapping = {
    "School": 0,
    "Undergraduate": 1,
    "Postgraduate": 2
}

encoded_education = education_level.map(education_mapping)
print(encoded_education)
```

### 10.4 One-Hot Encoding with Pandas

One-hot encoding creates a separate binary column for each category.

```python
df_one_hot = pd.get_dummies(
    df,
    columns=["City"],
    dtype=int
)

print(df_one_hot)
```

Using `drop_first=True`:

```python
df_one_hot = pd.get_dummies(
    df,
    columns=["City"],
    drop_first=True,
    dtype=int
)

print(df_one_hot)
```

Dropping one category may be useful for linear models because it avoids redundant dummy variables.

### 10.5 One-Hot Encoding with Scikit-Learn

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

city_encoded = encoder.fit_transform(df[["City"]])

city_columns = encoder.get_feature_names_out(["City"])
city_df = pd.DataFrame(city_encoded, columns=city_columns)

print(city_df)
```

`handle_unknown="ignore"` prevents an error when the test set contains a category that did not appear in the training set.

### 10.6 Choosing an Encoding Method

| Variable Type | Recommended Method |
|---|---|
| Binary nominal | Mapping or one-hot encoding |
| Multi-category nominal | One-hot encoding |
| Ordinal | Ordered mapping or ordinal encoding |
| Target class | Label encoding or mapping |

---

## 11. Feature Scaling

Features may have different numerical ranges. For example, age may range from 18 to 60, while annual income may range from 200,000 to 2,000,000. Features with larger values can dominate distance-based or gradient-based algorithms.

Feature scaling is especially important for:

- K-Nearest Neighbours
- Support Vector Machines
- K-Means clustering
- Principal Component Analysis
- Logistic Regression
- Regularised regression models
- Neural networks

Tree-based algorithms such as Decision Trees and Random Forests usually do not require feature scaling.

### 11.1 Standardisation

Standardisation transforms data so that it has approximately:

- Mean = 0
- Standard deviation = 1

Formula:

```text
z = (x - mean) / standard_deviation
```

Implementation:

```python
from sklearn.preprocessing import StandardScaler

sample = pd.DataFrame({
    "Age": [20, 25, 30, 35, 40],
    "Income": [250000, 350000, 500000, 700000, 900000]
})

standard_scaler = StandardScaler()
scaled_values = standard_scaler.fit_transform(sample)

standardised_df = pd.DataFrame(
    scaled_values,
    columns=sample.columns
)

print(standardised_df)
```

### 11.2 Min-Max Normalisation

Min-Max scaling usually transforms values to the range `[0, 1]`.

Formula:

```text
x_scaled = (x - minimum) / (maximum - minimum)
```

Implementation:

```python
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
normalised_values = min_max_scaler.fit_transform(sample)

normalised_df = pd.DataFrame(
    normalised_values,
    columns=sample.columns
)

print(normalised_df)
```

### 11.3 StandardScaler versus MinMaxScaler

| Aspect | StandardScaler | MinMaxScaler |
|---|---|---|
| Typical output | Mean 0 and standard deviation 1 | Usually between 0 and 1 |
| Sensitive to outliers | Yes | Yes |
| Common use | SVM, Logistic Regression, PCA | Neural networks, bounded inputs |
| Preserves relative distribution shape | Generally yes | Generally yes |

### 11.4 Fitting and Transforming Correctly

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Do not call `fit_transform()` separately on the test set.

Incorrect:

```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
```

Correct:

```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The second approach ensures that both datasets use the mean and standard deviation learned from the training set.

---

## 12. Train-Test Split

A machine learning model should be evaluated on data that was not used during training.

The dataset is commonly divided into:

- **Training set:** used to learn model parameters
- **Testing set:** used to evaluate the trained model

### 12.1 Separating Features and Target

```python
X = df.drop(columns=["Placed"])
y = df["Placed"]

print(X.head())
print(y.head())
```

By convention:

- `X` represents input features.
- `y` represents the target variable.

### 12.2 Basic Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
```

### 12.3 Meaning of Important Parameters

| Parameter | Meaning |
|---|---|
| `test_size=0.20` | Uses 20% of the data for testing |
| `random_state=42` | Makes the split reproducible |
| `shuffle=True` | Randomly shuffles observations before splitting |
| `stratify=y` | Preserves target-class proportions |

### 12.4 Stratified Split for Classification

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

Stratification is useful when the target classes are imbalanced or when the dataset is small.

### 12.5 Checking Class Distribution

```python
print("Complete dataset:")
print(y.value_counts(normalize=True))

print("Training set:")
print(y_train.value_counts(normalize=True))

print("Testing set:")
print(y_test.value_counts(normalize=True))
```

---

## 13. Complete Data-Preparation Pipeline

This section combines missing-value handling, categorical encoding, scaling, and train-test splitting.

### 13.1 Create the Dataset

```python
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


data = {
    "Age": [20, 21, np.nan, 22, 20, 23, 21, 24, 22, 20],
    "Gender": [
        "Male", "Female", "Female", "Male", "Female",
        "Male", "Male", "Female", "Male", "Female"
    ],
    "City": [
        "Delhi", "Mumbai", "Delhi", "Chandigarh", np.nan,
        "Mumbai", "Delhi", "Chandigarh", "Mumbai", "Delhi"
    ],
    "StudyHours": [4.5, 5.2, 3.8, np.nan, 6.1, 2.9, 4.0, 5.5, 4.8, 6.0],
    "Attendance": [78, 85, 72, 90, 88, 65, 75, 92, 81, 89],
    "Placed": ["Yes", "Yes", "No", "Yes", "Yes", "No", "No", "Yes", "Yes", "Yes"]
}

df = pd.DataFrame(data)
print(df)
```

### 13.2 Separate Features and Target

```python
X = df.drop(columns=["Placed"])
y = df["Placed"].map({"No": 0, "Yes": 1})
```

### 13.3 Split Before Fitting Preprocessing Steps

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### 13.4 Identify Numerical and Categorical Columns

```python
numeric_features = ["Age", "StudyHours", "Attendance"]
categorical_features = ["Gender", "City"]
```

### 13.5 Create Numerical Preprocessing Steps

```python
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)
```

The numerical pipeline:

1. Replaces missing numerical values with the training-set median.
2. Standardises numerical features.

### 13.6 Create Categorical Preprocessing Steps

```python
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        )
    ]
)
```

The categorical pipeline:

1. Replaces missing categories with the most frequent training-set category.
2. Converts categorical columns into one-hot encoded columns.
3. Ignores unseen test-set categories instead of raising an error.

### 13.7 Combine the Pipelines

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)
```

### 13.8 Fit on Training Data and Transform Both Sets

```python
X_train_prepared = preprocessor.fit_transform(X_train)
X_test_prepared = preprocessor.transform(X_test)

print("Prepared training shape:", X_train_prepared.shape)
print("Prepared testing shape:", X_test_prepared.shape)
```

### 13.9 Retrieve Transformed Feature Names

```python
feature_names = preprocessor.get_feature_names_out()

X_train_prepared_df = pd.DataFrame(
    X_train_prepared,
    columns=feature_names,
    index=X_train.index
)

X_test_prepared_df = pd.DataFrame(
    X_test_prepared,
    columns=feature_names,
    index=X_test.index
)

print(X_train_prepared_df)
print(X_test_prepared_df)
```

### 13.10 Why This Workflow Is Recommended

This workflow:

- Keeps numerical and categorical preprocessing separate.
- Prevents preprocessing leakage from the test set.
- Applies identical transformations to training and testing data.
- Handles missing values systematically.
- Handles unseen categorical values.
- Can later be combined directly with a machine learning model.

---

## 14. Common Mistakes and Best Practices

### 14.1 Fitting Preprocessing on the Complete Dataset

**Mistake:** Computing means, medians, category mappings, or scaling parameters before the train-test split.

**Best practice:** Split first and fit preprocessing steps on the training set only.

### 14.2 Scaling the Target Variable Accidentally

For ordinary classification tasks, scale only the input features unless the method specifically requires target transformation.

### 14.3 Applying Label Encoding to Nominal Features

Cities, colours, and departments do not have a natural order. One-hot encoding is generally safer for such variables.

### 14.4 Ignoring Data Types

A numerical column may be loaded as text because of invalid entries.

```python
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
```

Invalid values become missing values and can then be imputed.

### 14.5 Modifying the Original Data Unintentionally

Use a copy when experimenting.

```python
clean_df = df.copy()
```

### 14.6 Ignoring Duplicate Records

```python
print(df.duplicated().sum())
df = df.drop_duplicates()
```

Duplicates may bias a model if repeated observations are not genuine.

### 14.7 Using Different Columns During Training and Testing

A preprocessing pipeline ensures consistent feature order and transformation rules.

### 14.8 Failing to Set `random_state`

Without a fixed random state, train-test results may change each time the code is executed.

```python
random_state=42
```

### 14.9 Ignoring Unknown Categories

Use:

```python
OneHotEncoder(handle_unknown="ignore")
```

This makes the preprocessing pipeline more robust.

---

## 15. Practice Exercises

### Exercise 1: Python Basics

Create a list containing the marks of ten students. Perform the following tasks:

1. Calculate the highest and lowest marks.
2. Calculate the average marks.
3. Create a new list containing only marks greater than or equal to 60.
4. Count how many students passed when the passing marks are 40.

### Exercise 2: NumPy Array Operations

Create a NumPy array containing the monthly sales values:

```text
[120, 150, 170, 130, 200, 220]
```

Perform the following operations:

1. Calculate total sales.
2. Calculate average sales.
3. Find the highest and lowest sales values.
4. Increase every sales value by 10%.
5. Select sales values greater than 160.

### Exercise 3: DataFrame Creation

Create a DataFrame with the columns:

- EmployeeID
- Name
- Department
- Experience
- Salary

Add at least eight records and perform the following tasks:

1. Display the first five rows.
2. Display the DataFrame shape and data types.
3. Select employees with more than three years of experience.
4. Sort employees by salary in descending order.
5. Create a new column named `Senior`.

### Exercise 4: Missing Values

Create or load a dataset containing missing values. Perform the following tasks:

1. Count missing values in each column.
2. Calculate the missing-value percentage.
3. Fill numerical missing values with the median.
4. Fill categorical missing values with the mode.
5. Verify that no missing values remain.

### Exercise 5: Categorical Encoding

Create a DataFrame containing:

- Gender
- City
- Qualification
- Selected

Perform the following tasks:

1. Map `Selected` from Yes/No to 1/0.
2. Map `Qualification` using an ordinal order.
3. One-hot encode `City`.
4. Display the transformed dataset.

### Exercise 6: Feature Scaling

Create a dataset containing `Age`, `MonthlyIncome`, and `Experience`. Apply:

1. StandardScaler
2. MinMaxScaler

Compare the transformed values and explain the difference.

### Exercise 7: Train-Test Split

Using a classification dataset:

1. Separate input features and the target variable.
2. Create an 80:20 train-test split.
3. Set `random_state=42`.
4. Use stratification.
5. Compare class proportions in the original, training, and testing targets.

### Exercise 8: Complete Preprocessing Pipeline

Create a preprocessing pipeline for a dataset containing:

- Two numerical columns with missing values
- Two categorical columns with missing values
- One binary target column

The pipeline should:

1. Impute numerical values using the median.
2. Standardise numerical variables.
3. Impute categorical variables using the most frequent value.
4. Apply one-hot encoding.
5. Fit only on the training set.
6. Transform both training and testing sets.

---

## 16. Summary

In this unit, we covered the foundations required to prepare data for machine learning:

- Python variables, data types, collections, conditions, loops, and functions
- NumPy arrays, indexing, vectorisation, reshaping, and statistical operations
- Pandas Series and DataFrames
- Loading data from CSV, Excel, and JSON files
- Inspecting dataset structure, data types, distributions, and duplicates
- Detecting, deleting, and imputing missing values
- Encoding binary, nominal, and ordinal categorical variables
- Scaling numerical features using StandardScaler and MinMaxScaler
- Separating input features from the target variable
- Creating reproducible and stratified train-test splits
- Building a leakage-safe preprocessing pipeline using scikit-learn

A well-designed preprocessing workflow improves the reliability, reproducibility, and maintainability of a machine learning project.

---

## Repository Suggestions

A simple Git repository structure for this unit can be:

```text
CSE570-Machine-Learning-with-Python/
│
├── README.md
├── requirements.txt
├── Unit-01-Python-and-Data-Preparation/
│   ├── README.md
│   ├── data/
│   │   └── student_ml_data.csv
│   ├── notebooks/
│   │   └── unit_01_practice.ipynb
│   └── scripts/
│       └── preprocessing_pipeline.py
```

Suggested `requirements.txt`:

```text
numpy
pandas
matplotlib
scikit-learn
openpyxl
```

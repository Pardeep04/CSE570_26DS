# CSE570: Machine Learning with Python

## Unit I: Python Fundamentals and Data Preparation for Machine Learning

### Part 2: Pandas, Data Loading, and Missing Values

---

## Unit I File Structure

1. [Part 1: Python Fundamentals and NumPy](CSE570_Unit_1_Part_1_Python_and_NumPy.md)
2. [Part 2: Pandas, Data Loading, and Missing Values](CSE570_Unit_1_Part_2_Pandas_and_Data_Cleaning.md)
3. [Part 3: Encoding, Scaling, Splitting, and ML Pipeline](CSE570_Unit_1_Part_3_Preprocessing_and_Pipeline.md)

---

## Contents

- [6. Pandas DataFrames](#6-pandas-dataframes)
- [7. Loading Data](#7-loading-data)
- [8. Understanding a Dataset](#8-understanding-a-dataset)
- [9. Handling Missing Values](#9-handling-missing-values)

---

## 6. Pandas DataFrames

Pandas is used to organise, clean, transform, and analyse structured data.

### 6.1 Pandas Series

A Series is a one-dimensional labelled data structure.

```python
import pandas as pd

marks = pd.Series([72, 85, 68, 90], name="Marks")
print(marks)
```

Custom index labels can also be used.

```python
marks = pd.Series(
    [72, 85, 68],
    index=["Aman", "Riya", "Karan"]
)

print(marks)
```

### 6.2 Creating a DataFrame from a Dictionary

A DataFrame is a two-dimensional table consisting of rows and columns.

```python
data = {
    "StudentID": [101, 102, 103, 104],
    "Name": ["Aman", "Riya", "Karan", "Simran"],
    "Age": [20, 21, 20, 22],
    "Marks": [78, 85, 69, 91]
}

df = pd.DataFrame(data)
print(df)
```

### 6.3 Selecting Columns

```python
print(df["Name"])
print(df[["Name", "Marks"]])
```

Selecting a single column returns a Series. Selecting multiple columns returns a DataFrame.

### 6.4 Selecting Rows with `loc` and `iloc`

`loc` selects rows and columns by labels.

```python
print(df.loc[0])
print(df.loc[0:2, ["Name", "Marks"]])
```

`iloc` selects rows and columns by numerical position.

```python
print(df.iloc[0])
print(df.iloc[0:3, 1:4])
```

### 6.5 Filtering Rows

```python
high_scorers = df[df["Marks"] >= 80]
print(high_scorers)
```

Using multiple conditions:

```python
filtered = df[(df["Marks"] >= 75) & (df["Age"] <= 21)]
print(filtered)
```

Use `&` for AND, `|` for OR, and place each condition inside parentheses.

### 6.6 Adding a New Column

```python
df["Passed"] = df["Marks"] >= 40
print(df)
```

A calculated column can also be created.

```python
df["BonusMarks"] = df["Marks"] + 5
print(df)
```

### 6.7 Updating Values

```python
df.loc[df["StudentID"] == 103, "Marks"] = 74
print(df)
```

### 6.8 Removing Rows and Columns

```python
df_without_bonus = df.drop(columns=["BonusMarks"])
print(df_without_bonus)
```

To remove a row:

```python
df_without_first_row = df.drop(index=0)
print(df_without_first_row)
```

By default, `drop()` returns a new DataFrame. The original DataFrame remains unchanged unless `inplace=True` is used.

### 6.9 Sorting Data

```python
sorted_df = df.sort_values(by="Marks", ascending=False)
print(sorted_df)
```

Sorting by multiple columns:

```python
sorted_df = df.sort_values(
    by=["Age", "Marks"],
    ascending=[True, False]
)

print(sorted_df)
```

### 6.10 Renaming Columns

```python
renamed_df = df.rename(columns={"Marks": "FinalMarks"})
print(renamed_df)
```

---

## 7. Loading Data

Machine learning datasets are commonly stored in CSV, Excel, JSON, or database formats.

### 7.1 Loading a CSV File

```python
import pandas as pd

df = pd.read_csv("students.csv")
print(df.head())
```

### 7.2 Loading an Excel File

```python
df = pd.read_excel("students.xlsx")
print(df.head())
```

To load a particular sheet:

```python
df = pd.read_excel("students.xlsx", sheet_name="Sheet1")
```

### 7.3 Loading a JSON File

```python
df = pd.read_json("students.json")
print(df.head())
```

### 7.4 Common File-Loading Parameters

```python
df = pd.read_csv(
    "students.csv",
    usecols=["StudentID", "Age", "Marks"],
    na_values=["NA", "?", "Missing", "-"],
    encoding="utf-8"
)
```

Important parameters include:

| Parameter | Purpose |
|---|---|
| `usecols` | Loads selected columns only |
| `nrows` | Loads a specified number of rows |
| `na_values` | Treats specified values as missing |
| `encoding` | Defines the file's text encoding |
| `delimiter` | Specifies the field separator |
| `header` | Specifies the row containing column names |

### 7.5 Creating a Sample Dataset

The following dataset will be used in later examples.

```python
import numpy as np
import pandas as pd

data = {
    "StudentID": [101, 102, 103, 104, 105, 106, 107, 108],
    "Age": [20, 21, np.nan, 22, 20, 23, 21, np.nan],
    "Gender": ["Male", "Female", "Female", "Male", "Female", "Male", "Male", "Female"],
    "City": ["Delhi", "Mumbai", "Delhi", "Chandigarh", np.nan, "Mumbai", "Delhi", "Chandigarh"],
    "StudyHours": [4.5, 5.2, 3.8, np.nan, 6.1, 2.9, 4.0, 5.5],
    "Marks": [78, 85, 69, 91, 88, 62, 74, 89],
    "Placed": ["Yes", "Yes", "No", "Yes", "Yes", "No", "No", "Yes"]
}

df = pd.DataFrame(data)
print(df)
```

Saving it as a CSV file:

```python
df.to_csv("student_ml_data.csv", index=False)
```

---

## 8. Understanding a Dataset

A dataset should be examined before preprocessing or model training.

### 8.1 Viewing Initial and Final Rows

```python
print(df.head())
print(df.tail())
```

A custom number of rows can be displayed.

```python
print(df.head(3))
```

### 8.2 Dataset Shape

```python
print(df.shape)
```

The result is a tuple:

```text
(number_of_rows, number_of_columns)
```

### 8.3 Column Names

```python
print(df.columns)
```

### 8.4 Data Types and Non-Null Counts

```python
print(df.info())
```

`info()` helps identify:

- Column names
- Number of non-null values
- Data types
- Approximate memory usage

### 8.5 Statistical Summary

```python
print(df.describe())
```

For categorical columns:

```python
print(df.describe(include="object"))
```

For all columns:

```python
print(df.describe(include="all"))
```

### 8.6 Unique Values

```python
print(df["Gender"].unique())
print(df["City"].nunique())
print(df["City"].value_counts(dropna=False))
```

### 8.7 Checking Duplicate Rows

```python
print(df.duplicated())
print("Duplicate count:", df.duplicated().sum())
```

Removing duplicate rows:

```python
df = df.drop_duplicates()
```

---

## 9. Handling Missing Values

Missing values may occur because of incomplete forms, failed sensors, data-entry errors, unavailable information, or problems during data integration.

### 9.1 Detecting Missing Values

```python
print(df.isnull())
```

Count missing values in every column:

```python
print(df.isnull().sum())
```

Calculate missing-value percentages:

```python
missing_percentage = df.isnull().mean() * 100
print(missing_percentage)
```

### 9.2 Removing Rows with Missing Values

```python
complete_rows = df.dropna()
print(complete_rows)
```

This method is simple but may remove too much information, especially in a small dataset.

### 9.3 Removing Columns with Missing Values

```python
df_without_missing_columns = df.dropna(axis=1)
print(df_without_missing_columns)
```

A threshold can be used to retain columns containing a minimum number of non-missing values.

```python
df_threshold = df.dropna(axis=1, thresh=6)
```

### 9.4 Filling Numerical Missing Values with the Mean

```python
df_mean = df.copy()
df_mean["Age"] = df_mean["Age"].fillna(df_mean["Age"].mean())

print(df_mean)
```

The mean is sensitive to extreme values.

### 9.5 Filling Numerical Missing Values with the Median

```python
df_median = df.copy()
df_median["Age"] = df_median["Age"].fillna(df_median["Age"].median())
df_median["StudyHours"] = df_median["StudyHours"].fillna(
    df_median["StudyHours"].median()
)

print(df_median)
```

The median is often preferred when a numerical feature contains outliers or has a skewed distribution.

### 9.6 Filling Categorical Missing Values with the Mode

```python
df_mode = df.copy()
city_mode = df_mode["City"].mode()[0]
df_mode["City"] = df_mode["City"].fillna(city_mode)

print(df_mode)
```

### 9.7 Filling Missing Values with a Constant

```python
df_constant = df.copy()
df_constant["City"] = df_constant["City"].fillna("Unknown")

print(df_constant)
```

### 9.8 Using `SimpleImputer`

Scikit-learn provides `SimpleImputer` for systematic missing-value handling.

```python
from sklearn.impute import SimpleImputer

numeric_imputer = SimpleImputer(strategy="median")

df[["Age", "StudyHours"]] = numeric_imputer.fit_transform(
    df[["Age", "StudyHours"]]
)

categorical_imputer = SimpleImputer(strategy="most_frequent")

df[["City"]] = categorical_imputer.fit_transform(df[["City"]])

print(df)
```

Common imputation strategies are:

| Strategy | Suitable For |
|---|---|
| `mean` | Numerical data without strong outliers |
| `median` | Skewed numerical data or data with outliers |
| `most_frequent` | Categorical data |
| `constant` | Numerical or categorical data when a fixed replacement is appropriate |

### 9.9 Important Leakage Warning

Imputation values must be learned from the training set only.

Incorrect approach:

```python
# Avoid fitting preprocessing on the complete dataset before splitting.
df["Age"] = df["Age"].fillna(df["Age"].median())
```

Preferred approach:

1. Split the data into training and testing sets.
2. Fit the imputer on the training data.
3. Transform both training and testing data using the fitted imputer.

This prevents information from the test set from influencing model development.

---

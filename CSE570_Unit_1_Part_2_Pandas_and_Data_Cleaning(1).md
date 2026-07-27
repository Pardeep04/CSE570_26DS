# 🐼 CSE570: Machine Learning with Python

## Unit I — Part 2: Pandas, Data Loading, and Missing Values

> **Course focus:** Pandas DataFrames, loading data, dataset inspection, and missing-value handling.

---

## 📚 Unit I Navigation

| Part | Topic | File |
|---|---|---|
| **1** | Python Fundamentals and NumPy | [Open Part 1](CSE570_Unit_1_Part_1_Python_and_NumPy.md) |
| **2** | Pandas, Data Loading, and Missing Values | **You are here** |
| **3** | Encoding, Scaling, Splitting, and ML Pipeline | [Open Part 3](CSE570_Unit_1_Part_3_Preprocessing_and_Pipeline.md) |

---

## 🎯 Learning Outcomes

After completing this file, students will be able to:

- create Pandas Series and DataFrames;
- select, filter, update, sort, rename, and remove data;
- load CSV, Excel, and JSON files;
- inspect the structure and quality of a dataset;
- identify missing values;
- remove or impute missing data using suitable strategies.

---

<table><tr><td bgcolor="#E8F4FD">
<b>💡 Why Pandas matters</b><br>
Most machine learning datasets are organised as rows and columns. Pandas provides the DataFrame structure used to inspect, clean, transform, and prepare such data.
</td></tr></table>

# 1. Pandas Series and DataFrames

## 1.1 Importing Pandas

```python
# Import Pandas with its standard alias.
import pandas as pd

print("Pandas version:", pd.__version__)
```

---

## 1.2 Pandas Series

A **Series** is a one-dimensional labelled collection. It is similar to one column of a table.

```python
import pandas as pd

# Create a Series and assign it a meaningful name.
marks = pd.Series([72, 85, 68, 90], name="Marks")

print(marks)
```

### Series with custom labels

```python
marks = pd.Series(
    [72, 85, 68],
    index=["Aman", "Riya", "Karan"],
    name="Marks"
)

# Access a value through its custom label.
print("Riya's marks:", marks["Riya"])
```

---

## 1.3 Creating a DataFrame

A **DataFrame** is a two-dimensional table containing rows and columns.

```python
student_data = {
    "StudentID": [101, 102, 103, 104],
    "Name": ["Aman", "Riya", "Karan", "Simran"],
    "Age": [20, 21, 20, 22],
    "Marks": [78, 85, 69, 91]
}

# Convert the dictionary into a DataFrame.
df = pd.DataFrame(student_data)

print(df)
```

**Expected structure:**

| index | StudentID | Name | Age | Marks |
|---:|---:|---|---:|---:|
| 0 | 101 | Aman | 20 | 78 |
| 1 | 102 | Riya | 21 | 85 |
| 2 | 103 | Karan | 20 | 69 |
| 3 | 104 | Simran | 22 | 91 |

---

## 1.4 Selecting Columns

```python
# Select one column. The result is a Series.
name_series = df["Name"]
print(name_series)

# Select multiple columns. The result is a DataFrame.
selected_columns = df[["Name", "Marks"]]
print(selected_columns)
```

<table><tr><td bgcolor="#FFF4CE">
<b>⚠️ Remember</b><br>
Use one pair of brackets for a single column and a list inside double brackets for multiple columns.
</td></tr></table>

---

## 1.5 Selecting Rows with `loc` and `iloc`

- `loc` uses row and column **labels**.
- `iloc` uses numerical **positions**.

```python
# Select the row whose label is 0.
print(df.loc[0])

# Select rows 0 through 2 and only the Name and Marks columns.
# With loc, the ending row label is included.
print(df.loc[0:2, ["Name", "Marks"]])
```

```python
# Select the first row by numerical position.
print(df.iloc[0])

# Select the first three rows and columns at positions 1 to 3.
# With iloc slicing, the ending position is excluded.
print(df.iloc[0:3, 1:4])
```

---

## 1.6 Filtering Rows

```python
# Keep rows where Marks is at least 80.
high_scorers = df[df["Marks"] >= 80]
print(high_scorers)
```

### Filtering with multiple conditions

```python
# Each condition must be enclosed in parentheses.
# '&' represents AND for Pandas conditions.
filtered_students = df[
    (df["Marks"] >= 75) &
    (df["Age"] <= 21)
]

print(filtered_students)
```

| Symbol | Meaning |
|---|---|
| `&` | AND |
| `|` | OR |
| `~` | NOT |

---

## 1.7 Adding and Updating Columns

```python
# Create a Boolean column based on a condition.
df["Passed"] = df["Marks"] >= 40

# Create a calculated column.
df["BonusMarks"] = df["Marks"] + 5

print(df)
```

### Updating a particular value

```python
# Locate StudentID 103 and update only the Marks column.
df.loc[df["StudentID"] == 103, "Marks"] = 74

print(df)
```

---

## 1.8 Removing Rows and Columns

```python
# Create a new DataFrame without BonusMarks.
df_without_bonus = df.drop(columns=["BonusMarks"])
print(df_without_bonus)
```

```python
# Remove the row whose index label is 0.
df_without_first_row = df.drop(index=0)
print(df_without_first_row)
```

<table><tr><td bgcolor="#EAF7EA">
<b>✅ Safe practice</b><br>
Prefer assigning the result to a new variable while learning. This preserves the original DataFrame and makes errors easier to correct.
</td></tr></table>

---

## 1.9 Sorting and Renaming

```python
# Sort from highest to lowest marks.
sorted_df = df.sort_values(by="Marks", ascending=False)
print(sorted_df)
```

```python
# Sort by Age in ascending order and Marks in descending order.
sorted_df = df.sort_values(
    by=["Age", "Marks"],
    ascending=[True, False]
)

print(sorted_df)
```

```python
# Rename Marks to FinalMarks without modifying the original DataFrame.
renamed_df = df.rename(columns={"Marks": "FinalMarks"})
print(renamed_df)
```

---

# 2. Loading Data from Files

Machine learning data is commonly stored in CSV, Excel, or JSON files.

## 2.1 Loading a CSV File

```python
import pandas as pd

# Read the complete CSV file into a DataFrame.
df = pd.read_csv("students.csv")

# Show the first five rows.
print(df.head())
```

## 2.2 Loading an Excel File

```python
# openpyxl may be required for .xlsx files.
df = pd.read_excel("students.xlsx")
print(df.head())
```

```python
# Load a specific worksheet by its name.
df = pd.read_excel(
    "students.xlsx",
    sheet_name="Sheet1"
)
```

## 2.3 Loading a JSON File

```python
# Read structured JSON records into a DataFrame.
df = pd.read_json("students.json")
print(df.head())
```

---

## 2.4 Useful File-Loading Parameters

```python
df = pd.read_csv(
    "students.csv",

    # Read only the required columns.
    usecols=["StudentID", "Age", "Marks"],

    # Treat these text values as missing data.
    na_values=["NA", "?", "Missing", "-"],

    # Specify the character encoding.
    encoding="utf-8"
)
```

| Parameter | Purpose |
|---|---|
| `usecols` | Loads selected columns only |
| `nrows` | Loads a limited number of rows |
| `skiprows` | Skips unwanted rows |
| `na_values` | Defines additional missing-value symbols |
| `dtype` | Sets required column data types |
| `encoding` | Specifies text encoding |
| `sheet_name` | Selects an Excel worksheet |

---

## 2.5 Handling File Errors

```python
import pandas as pd

try:
    # Attempt to read the requested file.
    df = pd.read_csv("students.csv")
    print("File loaded successfully.")

except FileNotFoundError:
    # This block runs when the file path is incorrect.
    print("File not found. Check the file name and path.")

except pd.errors.EmptyDataError:
    # This block runs when the file contains no data.
    print("The file is empty.")
```

---

# 3. Understanding a Dataset

Before cleaning data, always inspect its rows, columns, data types, summary statistics, and missing values.

## 3.1 Previewing Data

```python
# Display the first five rows.
print(df.head())

# Display the first ten rows.
print(df.head(10))

# Display the final five rows.
print(df.tail())

# Select five random rows.
print(df.sample(5, random_state=42))
```

---

## 3.2 Shape, Columns, and Data Types

```python
# shape returns (number_of_rows, number_of_columns).
print("Shape:", df.shape)

# Display all column names.
print("Columns:", df.columns.tolist())

# Display the data type of each column.
print(df.dtypes)
```

---

## 3.3 Dataset Information

```python
# info() displays column names, non-null counts, and data types.
df.info()
```

<table><tr><td bgcolor="#E8F4FD">
<b>💡 Interpretation</b><br>
When a column has fewer non-null values than the total number of rows, that column contains missing data.
</td></tr></table>

---

## 3.4 Descriptive Statistics

```python
# Summarise numerical columns.
print(df.describe())

# Include both numerical and categorical columns.
print(df.describe(include="all"))
```

For numerical data, `describe()` commonly returns count, mean, standard deviation, minimum, quartiles, and maximum.

---

## 3.5 Unique Values and Frequencies

```python
# Count the number of unique values in every column.
print(df.nunique())

# Display unique values in a categorical column.
print(df["City"].unique())

# Count how often each category occurs.
print(df["City"].value_counts())

# Include missing values in the frequency table.
print(df["City"].value_counts(dropna=False))
```

---

## 3.6 Detecting Duplicate Rows

```python
# Return True for rows that duplicate an earlier row.
duplicate_mask = df.duplicated()
print(duplicate_mask)

# Count duplicate rows.
print("Duplicate rows:", df.duplicated().sum())

# Remove duplicates and create a cleaned copy.
df_no_duplicates = df.drop_duplicates()
```

---

# 4. Handling Missing Values

A missing value means that information is unavailable, unrecorded, or invalid. Pandas generally displays missing values as `NaN` or `None`.

## 4.1 Creating a Sample Dataset

```python
import numpy as np
import pandas as pd

student_data = {
    "StudentID": [101, 102, 103, 104, 105],
    "Age": [20, 21, np.nan, 22, 20],
    "City": ["Delhi", "Mumbai", "Delhi", None, "Chandigarh"],
    "Marks": [78, np.nan, 69, 91, 85]
}

df = pd.DataFrame(student_data)
print(df)
```

---

## 4.2 Detecting Missing Values

```python
# True indicates that the corresponding value is missing.
print(df.isna())

# Count missing values column by column.
missing_count = df.isna().sum()
print(missing_count)
```

### Missing-value percentage

```python
# Divide missing counts by total rows and multiply by 100.
missing_percentage = (df.isna().sum() / len(df)) * 100

missing_report = pd.DataFrame({
    "Missing_Count": df.isna().sum(),
    "Missing_Percentage": missing_percentage.round(2)
})

print(missing_report)
```

---

## 4.3 Removing Missing Values

### Remove rows containing any missing value

```python
# A row is removed when at least one column is missing.
complete_rows = df.dropna()
print(complete_rows)
```

### Remove rows only when selected columns are missing

```python
# Remove rows only when Marks is missing.
df_marks_available = df.dropna(subset=["Marks"])
print(df_marks_available)
```

### Remove columns containing missing values

```python
# axis=1 means operate on columns.
df_complete_columns = df.dropna(axis=1)
print(df_complete_columns)
```

<table><tr><td bgcolor="#FDECEC">
<b>❌ Caution</b><br>
Dropping data can reduce the sample size and remove useful information. Use it only when the loss is acceptable and justified.
</td></tr></table>

---

## 4.4 Filling Numerical Missing Values

### Mean imputation

```python
# Create a copy to preserve the original dataset.
df_mean = df.copy()

# Calculate the mean using available values.
marks_mean = df_mean["Marks"].mean()

# Replace missing Marks with the mean.
df_mean["Marks"] = df_mean["Marks"].fillna(marks_mean)

print(df_mean)
```

### Median imputation

```python
df_median = df.copy()

# Median is less affected by extreme values than the mean.
age_median = df_median["Age"].median()
df_median["Age"] = df_median["Age"].fillna(age_median)

print(df_median)
```

---

## 4.5 Filling Categorical Missing Values

```python
df_mode = df.copy()

# mode() may return multiple values, so [0] selects the first mode.
city_mode = df_mode["City"].mode()[0]

# Replace missing City values with the most frequent city.
df_mode["City"] = df_mode["City"].fillna(city_mode)

print(df_mode)
```

An explicit category can also be used:

```python
df_unknown = df.copy()

# Preserve the fact that the original value was unavailable.
df_unknown["City"] = df_unknown["City"].fillna("Unknown")
```

---

## 4.6 Forward Fill and Backward Fill

These techniques are mainly suitable for ordered or time-series data.

```python
# Forward fill uses the previous available value.
df_forward = df.ffill()

# Backward fill uses the next available value.
df_backward = df.bfill()
```

<table><tr><td bgcolor="#FFF4CE">
<b>⚠️ Important</b><br>
Do not use forward fill merely because it is convenient. The row order must have a meaningful sequence.
</td></tr></table>

---

## 4.7 Missing-Value Indicators

Sometimes the fact that a value is missing may itself be informative.

```python
df_indicator = df.copy()

# Create 1 when Marks is missing and 0 otherwise.
df_indicator["Marks_Was_Missing"] = (
    df_indicator["Marks"].isna().astype(int)
)

# Fill the original Marks column after preserving missingness information.
df_indicator["Marks"] = df_indicator["Marks"].fillna(
    df_indicator["Marks"].median()
)

print(df_indicator)
```

---

## 4.8 Imputation Using Scikit-Learn

```python
from sklearn.impute import SimpleImputer

# Create a median imputer for numerical columns.
numeric_imputer = SimpleImputer(strategy="median")

# fit_transform() learns the median and fills missing values.
df[["Age", "Marks"]] = numeric_imputer.fit_transform(
    df[["Age", "Marks"]]
)

# Create a most-frequent imputer for categorical columns.
categorical_imputer = SimpleImputer(strategy="most_frequent")

df[["City"]] = categorical_imputer.fit_transform(df[["City"]])

print(df)
```

---

## 4.9 Choosing a Missing-Value Strategy

| Situation | Possible strategy |
|---|---|
| Very few incomplete rows | Remove rows carefully |
| Numerical data without major outliers | Mean imputation |
| Numerical data with skewness or outliers | Median imputation |
| Categorical data | Mode or `Unknown` category |
| Ordered time-series data | Forward or backward fill, when justified |
| ML workflow | `SimpleImputer` inside a pipeline |

---

# 5. Complete Cleaning Example

```python
import numpy as np
import pandas as pd

raw_data = {
    "StudentID": [101, 102, 102, 103, 104],
    "Name": [" Aman ", "Riya", "Riya", "Karan", "Simran"],
    "Age": [20, 21, 21, np.nan, 22],
    "City": ["Delhi", "Mumbai", "Mumbai", None, "Delhi"],
    "Marks": [78, 85, 85, 69, np.nan]
}

df = pd.DataFrame(raw_data)

# Step 1: Remove exact duplicate rows.
df = df.drop_duplicates()

# Step 2: Remove unwanted spaces from student names.
df["Name"] = df["Name"].str.strip()

# Step 3: Fill numerical missing values with the median.
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Marks"] = df["Marks"].fillna(df["Marks"].median())

# Step 4: Fill categorical missing values with the mode.
df["City"] = df["City"].fillna(df["City"].mode()[0])

# Step 5: Reset row labels after rows have been removed.
df = df.reset_index(drop=True)

print(df)
```

---

# 6. Practice Exercises

1. Create a DataFrame containing student ID, name, course, attendance, and marks.
2. Select only the name and marks columns.
3. Filter students whose attendance is at least 75 and marks are at least 40.
4. Add a `Result` column containing `Pass` or `Fail`.
5. Load a CSV file and display its shape, columns, data types, and first ten rows.
6. Create a missing-value report containing count and percentage.
7. Fill missing numerical values using the median.
8. Fill missing categorical values using the mode.
9. Remove duplicate rows and reset the index.
10. Explain why dropping every incomplete row may be harmful.

---

# 7. Quick Revision

| Operation | Pandas command |
|---|---|
| First rows | `df.head()` |
| Dataset size | `df.shape` |
| Column types | `df.dtypes` |
| Full structure | `df.info()` |
| Summary statistics | `df.describe()` |
| Missing count | `df.isna().sum()` |
| Remove missing rows | `df.dropna()` |
| Fill missing values | `df.fillna()` |
| Remove duplicates | `df.drop_duplicates()` |
| Sort values | `df.sort_values()` |

---

## ⬅️ Previous / ➡️ Next

[Part 1: Python Fundamentals and NumPy](CSE570_Unit_1_Part_1_Python_and_NumPy.md) · [Part 3: Encoding, Scaling, Splitting, and ML Pipeline](CSE570_Unit_1_Part_3_Preprocessing_and_Pipeline.md)

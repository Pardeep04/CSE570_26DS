# CSE570: Machine Learning with Python

# Part 1 — Exploratory Data Analysis and Visualisation


## Unit II Structure

- [Part 1 — EDA and Visualisation](CSE570_Unit_2_Part_1_EDA_and_Visualisation.md)
- [Part 2 — Correlation, Outliers, and Feature Engineering](CSE570_Unit_2_Part_2_Correlation_Outliers_and_Feature_Engineering.md)
- [Part 3 — Imbalance and Preprocessing Pipelines](CSE570_Unit_2_Part_3_Imbalance_and_Preprocessing_Pipelines.md)


<div style="background-color:#E8F4FD; border-left:6px solid #2196F3; padding:14px;">
<b>Purpose:</b> This part introduces students to Exploratory Data Analysis and basic visualisation before model building.
</div>

## Learning Outcomes

Students will be able to inspect a dataset, identify missing values and duplicates, calculate descriptive statistics, and create basic plots.

## Dataset

Use `student_performance_eda.csv`. It contains student-related numerical and categorical features, missing values, and a placement target.

## 1. Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

- `pandas` handles tables.
- `numpy` supports numerical work.
- `matplotlib` creates charts.

## 2. Load Data

```python
df = pd.read_csv("student_performance_eda.csv")
df.head()
```

`head()` displays the first five rows and confirms that loading was successful.

## 3. Basic Inspection

```python
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
df.info()
```

- `shape` gives rows and columns.
- `columns` lists variable names.
- `info()` shows data types and non-null counts.

## 4. Summary Statistics

```python
df.describe().round(2)
```

This shows count, mean, standard deviation, quartiles, minimum, and maximum.

## 5. Missing Values

```python
missing_report = pd.DataFrame({
    "Missing_Count": df.isnull().sum(),
    "Missing_Percentage": df.isnull().mean().mul(100).round(2)
})
missing_report
```

<div style="background-color:#FFF8E1; border-left:6px solid #FFC107; padding:14px;">
Do not delete missing rows immediately. First study how much data is missing and whether imputation is suitable.
</div>

## 6. Duplicate Records

```python
print("Duplicate rows:", df.duplicated().sum())
print("Repeated Student IDs:", df["Student_ID"].duplicated().sum())
```

## 7. Categorical Frequencies

```python
df["Study_Mode"].value_counts()
```

Percentages:

```python
df["Study_Mode"].value_counts(normalize=True).mul(100).round(2)
```

## 8. Histogram

```python
plt.figure(figsize=(8, 5))
plt.hist(df["Final_Score"].dropna(), bins=12, edgecolor="black")
plt.title("Distribution of Final Scores")
plt.xlabel("Final Score")
plt.ylabel("Number of Students")
plt.show()
```

A histogram helps identify centre, spread, skewness, and unusual values.

## 9. Bar Chart

```python
counts = df["Study_Mode"].value_counts()
plt.figure(figsize=(7, 4))
plt.bar(counts.index, counts.values)
plt.title("Students by Study Mode")
plt.xlabel("Study Mode")
plt.ylabel("Count")
plt.show()
```

## 10. Box Plot

```python
plt.figure(figsize=(7, 4))
plt.boxplot(df["Study_Hours_Per_Day"].dropna(), vert=False)
plt.title("Study Hours per Day")
plt.xlabel("Hours")
plt.show()
```

A box plot is useful for spotting potential outliers.

## 11. Scatter Plot

```python
plot_data = df.dropna(subset=["Study_Hours_Per_Day", "Final_Score"])

plt.figure(figsize=(8, 5))
plt.scatter(plot_data["Study_Hours_Per_Day"], plot_data["Final_Score"], alpha=0.7)
plt.title("Study Hours and Final Score")
plt.xlabel("Study Hours per Day")
plt.ylabel("Final Score")
plt.show()
```

## Practice Tasks

1. Display the last five rows.
2. Find the number of numerical and categorical columns.
3. Create a missing-value report.
4. Plot a histogram of `Previous_Score`.
5. Plot a bar chart of `Placed`.
6. Explain one pattern visible in the scatter plot.

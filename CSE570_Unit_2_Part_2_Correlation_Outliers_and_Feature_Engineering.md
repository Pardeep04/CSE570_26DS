# CSE570: Machine Learning with Python

# Part 2 — Correlation, Outliers, and Feature Engineering


## Unit II Structure

- [Part 1 — EDA and Visualisation](CSE570_Unit_2_Part_1_EDA_and_Visualisation.md)
- [Part 2 — Correlation, Outliers, and Feature Engineering](CSE570_Unit_2_Part_2_Correlation_Outliers_and_Feature_Engineering.md)
- [Part 3 — Imbalance and Preprocessing Pipelines](CSE570_Unit_2_Part_3_Imbalance_and_Preprocessing_Pipelines.md)


<div style="background-color:#E8F5E9; border-left:6px solid #34A853; padding:14px;">
<b>Purpose:</b> This part explains relationships, unusual values, feature transformation, and basic feature selection.
</div>

## 1. Correlation Analysis

Correlation measures linear association.

- Near `+1`: strong positive relationship
- Near `-1`: strong negative relationship
- Near `0`: weak linear relationship

```python
numeric_df = df.select_dtypes(include="number")
correlation_matrix = numeric_df.corr()
correlation_matrix.round(2)
```

<div style="background-color:#FFF8E1; border-left:6px solid #FFC107; padding:14px;">
Correlation does not prove causation.
</div>

## 2. Correlation Heatmap

```python
fig, ax = plt.subplots(figsize=(10, 7))
image = ax.imshow(correlation_matrix, aspect="auto")

ax.set_xticks(range(len(correlation_matrix.columns)))
ax.set_yticks(range(len(correlation_matrix.columns)))
ax.set_xticklabels(correlation_matrix.columns, rotation=90)
ax.set_yticklabels(correlation_matrix.columns)

for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        ax.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=8)

fig.colorbar(image)
plt.tight_layout()
plt.show()
```

## 3. Detect Highly Correlated Features

```python
absolute_corr = correlation_matrix.abs()

upper_triangle = absolute_corr.where(
    np.triu(np.ones(absolute_corr.shape), k=1).astype(bool)
)

high_corr = [
    column for column in upper_triangle.columns
    if any(upper_triangle[column] > 0.85)
]

print(high_corr)
```

## 4. Outlier Detection Using IQR

```python
def find_iqr_outliers(dataframe, column):
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = dataframe[
        (dataframe[column] < lower) |
        (dataframe[column] > upper)
    ]

    return lower, upper, outliers
```

Example:

```python
lower, upper, outliers = find_iqr_outliers(df, "Study_Hours_Per_Day")
print(lower, upper)
outliers
```

## 5. Outlier Capping

```python
df_capped = df.copy()

df_capped["Study_Hours_Per_Day"] = df_capped[
    "Study_Hours_Per_Day"
].clip(lower=lower, upper=upper)
```

Capping keeps the row but restricts the extreme value.

## 6. Log Transformation

```python
df["Study_Hours_Log"] = np.log1p(df["Study_Hours_Per_Day"])
```

`log1p(x)` calculates `log(1+x)` and safely handles zero.

## 7. Square-Root Transformation

```python
df["Assignments_Sqrt"] = np.sqrt(df["Assignments_Completed"])
```

## 8. Binning

```python
df["Attendance_Level"] = pd.cut(
    df["Attendance_Percentage"],
    bins=[0, 60, 75, 90, 100],
    labels=["Low", "Moderate", "Good", "Excellent"],
    include_lowest=True
)
```

## 9. Interaction Feature

```python
df["Attendance_Study_Interaction"] = (
    df["Attendance_Percentage"] *
    df["Study_Hours_Per_Day"]
)
```

## 10. Basic Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif

X_numeric = df.select_dtypes(include="number").drop(columns=["Student_ID"])
X_numeric = X_numeric.fillna(X_numeric.median())
y = df["Placed"].map({"No": 0, "Yes": 1})

selector = SelectKBest(score_func=f_classif, k=4)
selector.fit(X_numeric, y)

scores = pd.Series(
    selector.scores_,
    index=X_numeric.columns
).sort_values(ascending=False)

scores
```

## Practice Tasks

1. Find the strongest positive correlation with `Final_Score`.
2. Detect outliers in `Attendance_Percentage`.
3. Create a capped version of `Previous_Score`.
4. Create `High_Assignment_Completion`.
5. Compare original and log-transformed study hours.

# CSE570: Machine Learning with Python

# Part 3 — Imbalanced Data and Preprocessing Pipelines


## Unit II Structure

- [Part 1 — EDA and Visualisation](CSE570_Unit_2_Part_1_EDA_and_Visualisation.md)
- [Part 2 — Correlation, Outliers, and Feature Engineering](CSE570_Unit_2_Part_2_Correlation_Outliers_and_Feature_Engineering.md)
- [Part 3 — Imbalance and Preprocessing Pipelines](CSE570_Unit_2_Part_3_Imbalance_and_Preprocessing_Pipelines.md)


<div style="background-color:#F3E8FD; border-left:6px solid #8E24AA; padding:14px;">
<b>Purpose:</b> This part explains class imbalance and shows how to build a complete leakage-safe preprocessing pipeline.
</div>

## 1. Check Class Distribution

```python
df["Placed"].value_counts()
```

Percentages:

```python
df["Placed"].value_counts(normalize=True).mul(100).round(2)
```

Accuracy alone can be misleading when one class is much larger.

## 2. Install imbalanced-learn in Colab

```python
!pip -q install imbalanced-learn
```

## 3. Prepare Numerical Data for Resampling

```python
X_resample = df.select_dtypes(include="number").drop(columns=["Student_ID"])
X_resample = X_resample.fillna(X_resample.median())
y_resample = df["Placed"].map({"No": 0, "Yes": 1})
```

## 4. Random Oversampling

```python
from imblearn.over_sampling import RandomOverSampler

sampler = RandomOverSampler(random_state=42)
X_over, y_over = sampler.fit_resample(X_resample, y_resample)
y_over.value_counts()
```

## 5. Random Undersampling

```python
from imblearn.under_sampling import RandomUnderSampler

sampler = RandomUnderSampler(random_state=42)
X_under, y_under = sampler.fit_resample(X_resample, y_resample)
y_under.value_counts()
```

## 6. SMOTE

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_resample, y_resample)
y_smote.value_counts()
```

<div style="background-color:#FCE8E6; border-left:6px solid #D93025; padding:14px;">
Never resample the full dataset before splitting. Resampling must be applied only to training data.
</div>

## 7. Train-Test Split

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["Student_ID", "Placed"])
y = df["Placed"].map({"No": 0, "Yes": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

## 8. Identify Column Types

```python
numerical_columns = X_train.select_dtypes(include="number").columns.tolist()
categorical_columns = X_train.select_dtypes(exclude="number").columns.tolist()
```

## 9. Numerical Pipeline

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
```

## 10. Categorical Pipeline

```python
from sklearn.preprocessing import OneHotEncoder

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
```

## 11. ColumnTransformer

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ("numeric", numerical_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])
```

## 12. Complete Pipeline

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression

model_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("feature_selection", SelectKBest(score_func=f_classif, k=8)),
    ("model", LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    ))
])
```

## 13. Train and Evaluate

```python
model_pipeline.fit(X_train, y_train)
predictions = model_pipeline.predict(X_test)
```

```python
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=0))
```

## Practice Tasks

1. Compare class counts before and after each sampling method.
2. Explain oversampling versus undersampling.
3. Replace Logistic Regression with a Decision Tree.
4. Explain why `stratify=y` is used.
5. Explain why `handle_unknown="ignore"` is helpful.

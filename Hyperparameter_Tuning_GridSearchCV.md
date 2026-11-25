# 🟦 Hyperparameter Tuning and GridSearchCV

---

## 🟣 1. Hyperparameter Tuning

🔹 **Definition:** Hyperparameter tuning is the process of finding the best values of a machine learning model's hyperparameters to improve its performance on unseen data.

Hyperparameters are settings specified before model training. Unlike model parameters, they are not learned automatically during training.

| Algorithm | Examples of hyperparameters |
|---|---|
| KNN | `n_neighbors`, `weights` |
| Decision Tree | `max_depth`, `criterion` |
| Random Forest | `n_estimators`, `max_depth` |
| SVM | `C`, `kernel`, `gamma` |

### 🧩 Example: KNN

| Number of neighbors (K) | Validation accuracy |
|---:|---:|
| 3 | 90% |
| 5 | 94% |
| 7 | **98%** 🏆 |
| 9 | 92% |

K = 7 is selected because it provides the highest validation accuracy in this illustrative example.

---

## 🔵 2. Grid Search Cross-Validation (GridSearchCV)

🔹 **Definition:** GridSearchCV is a hyperparameter tuning technique that tests all specified combinations of hyperparameters using cross-validation and selects the combination with the best average validation score.

It is available in the `sklearn.model_selection` module.

### 🧩 Example

```python
param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance']
}
```

| Combination | n_neighbors | weights | Mean CV accuracy |
|---:|---:|---|---:|
| 1 | 3 | uniform | 92% |
| 2 | 3 | distance | 94% |
| 3 | 5 | uniform | 95% |
| 4 | 5 | distance | **98%** 🏆 |
| 5 | 7 | uniform | 93% |
| 6 | 7 | distance | 96% |

🏆 **Best combination:** `n_neighbors=5`, `weights='distance'` (illustrative scores).

- Total combinations = 3 × 2 = 6
- With `cv=5`, total CV trainings = 6 × 5 = 30
- By default, one additional training refits the best model on the complete training dataset.

---

## 🟠 3. Why is Cross-Validation Required in GridSearchCV?

Cross-validation helps select the best hyperparameters based on performance across multiple subsets of data rather than a single validation split. This reduces the risk of selecting a combination that performs well only on one particular split.

### 🔴 Without cross-validation

| K | Validation accuracy |
|---:|---:|
| 3 | **98%** 🏆 |
| 5 | 96% |
| 7 | 92% |

K = 3 is selected based on a single validation split.

### 🟢 With 5-fold cross-validation

| K | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 98% | 80% | 85% | 90% | 87% | 88% |
| 5 | 96% | 95% | 97% | 96% | 96% | **96%** 🏆 |
| 7 | 92% | 90% | 94% | 93% | 91% | 92% |

K = 5 is selected because it has the highest average validation accuracy and more consistent performance. All scores above are hypothetical.

### ✅ Benefits of cross-validation

- Reduces dependency on a single validation split.
- Provides a more reliable estimate of model performance.
- Helps select hyperparameters that generalize better to unseen data.

---

## 🟢 4. Python Code: GridSearchCV using KNN

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_iris(return_X_y=True)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create model with feature scaling
model = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# Define hyperparameter grid
param_grid = {
    'knn__n_neighbors': [3, 5, 7],
    'knn__weights': ['uniform', 'distance']
}

# Apply GridSearchCV
grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

# Train model
grid.fit(X_train, y_train)

# Display best hyperparameters and CV accuracy
print('Best Parameters:', grid.best_params_)
print('Best CV Accuracy:', grid.best_score_)

# Evaluate best model on unseen test data
y_pred = grid.predict(X_test)
print('Test Accuracy:', accuracy_score(y_test, y_pred))
```

💡 **Note:** The pipeline performs feature scaling separately within each training fold, preventing data leakage. The `knn__` prefix identifies the KNN step in the pipeline. The test set is kept separate from hyperparameter tuning.

### 📌 Important arguments and attributes

| Keyword | Description |
|---|---|
| `param_grid` | Hyperparameters and their possible values |
| `cv=5` | Five-fold cross-validation (stratified by default for this classification task) |
| `scoring='accuracy'` | Evaluation metric |
| `best_params_` | Best hyperparameter combination |
| `best_score_` | Best mean cross-validation score |
| `best_estimator_` | Best refitted model |

---

## 🟡 Quick Summary

- 🟣 **Hyperparameter tuning:** Finds the best model settings.
- 🔵 **Grid Search:** Generates all specified hyperparameter combinations.
- 🟠 **Cross-validation:** Evaluates each combination on different subsets of data.
- 🟢 **GridSearchCV:** Selects the combination with the best mean validation score.

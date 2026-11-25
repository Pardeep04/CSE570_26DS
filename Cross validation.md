# 📘 Cross-Validation in Python
## 🔵 K-Fold and Stratified K-Fold Cross-Validation

## 🔵 1. Introduction

Cross-validation is a machine learning technique used to estimate the performance of a model on unseen data. The dataset is divided into subsets called **folds**. A fresh model is trained and validated multiple times using different combinations of folds. Finally, the average performance is calculated.

> 💡 **Real-world example:** A teacher conducts five tests rather than one and calculates the average marks to obtain a more reliable estimate of a student's knowledge.

### 🔹 Why cross-validation is required

- To evaluate model performance on unseen data.
- To identify potential overfitting by comparing training and validation performance.
- To reduce dependency on a single train-test split.
- To compare machine learning algorithms and support model selection.

## 🔵 2. Train-Test Split vs Cross-Validation

For a dataset of 100 records, a traditional 80:20 split trains on 80 records and tests on 20 records once. Five-fold cross-validation trains on 80 records and validates on 20 records in each of five iterations, rotating the validation subset.

| Parameter | Train-Test Split | Cross-Validation |
|---|---|---|
| Model training | Once | Multiple times |
| Evaluation | Once | Multiple times |
| Reliability | Depends on a single split | Generally more reliable |
| Computational time | Less | More |
| Data utilization | One fixed split | Different training and validation subsets |

The two techniques covered in these notes are K-Fold and Stratified K-Fold.

---

## 🔵 3. K-Fold Cross-Validation

K-Fold divides the dataset into K equal or approximately equal subsets. In each iteration, K−1 folds are used for training and one fold is used for validation. This process is repeated K times.

### 🔹 3.1 Working of K-Fold

Suppose there are 100 records and K = 5. Each fold contains 20 records.

```text
              F1   F2   F3   F4   F5
Iteration 1   V    T    T    T    T
Iteration 2   T    V    T    T    T
Iteration 3   T    T    V    T    T
Iteration 4   T    T    T    V    T
Iteration 5   T    T    T    T    V

T = Training fold     V = Validation fold
Each iteration: 80 training records, 20 validation records
```

**Color-coded fold diagram**

| Iteration | F1 | F2 | F3 | F4 | F5 |
|---|:---:|:---:|:---:|:---:|:---:|
| 1 | 🟧 | 🟦 | 🟦 | 🟦 | 🟦 |
| 2 | 🟦 | 🟧 | 🟦 | 🟦 | 🟦 |
| 3 | 🟦 | 🟦 | 🟧 | 🟦 | 🟦 |
| 4 | 🟦 | 🟦 | 🟦 | 🟧 | 🟦 |
| 5 | 🟦 | 🟦 | 🟦 | 🟦 | 🟧 |

🟦 Training &nbsp; · &nbsp; 🟧 Validation


| Iteration | Training folds | Validation fold | Illustrative accuracy |
|---|---|---|---:|
| 1 | F2, F3, F4, F5 | F1 | 90% |
| 2 | F1, F3, F4, F5 | F2 | 85% |
| 3 | F1, F2, F4, F5 | F3 | 95% |
| 4 | F1, F2, F3, F5 | F4 | 80% |
| 5 | F1, F2, F3, F4 | F5 | 90% |

Average Accuracy = Sum of fold accuracies / K

> 🟢 Average Accuracy = (90 + 85 + 95 + 80 + 90) / 5 = **88%**

| K | Total records | Training per iteration | Validation per iteration | Iterations |
|---:|---:|---:|---:|---:|
| 2 | 100 | 50 | 50 | 2 |
| 4 | 100 | 75 | 25 | 4 |
| 5 | 100 | 80 | 20 | 5 |
| 10 | 100 | 90 | 10 | 10 |

### 🔹 3.2 Python Syntax

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
```

| Parameter | Explanation |
|---|---|
| `n_splits` | Number of folds |
| `shuffle=True` | Randomly shuffles records before splitting |
| `random_state=42` | Makes random splitting reproducible |

### 🔹 3.3 Example 1: Understanding K-Fold using a simple dataset

```python
import numpy as np
from sklearn.model_selection import KFold

# Create a dataset containing 10 records
data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# Divide the dataset into 5 folds
kf = KFold(n_splits=5)

fold = 1

# Generate training and validation indices
for train_index, test_index in kf.split(data):
    print("Fold:", fold)
    print("Training Data:", data[train_index])
    print("Validation Data:", data[test_index])
    print("------------------------")
    fold = fold + 1
```

#### 🟠 Output

```text
Fold: 1
Training Data: [30 40 50 60 70 80 90 100]
Validation Data: [10 20]
------------------------
Fold: 2
Training Data: [10 20 50 60 70 80 90 100]
Validation Data: [30 40]
------------------------
Fold: 3
Training Data: [10 20 30 40 70 80 90 100]
Validation Data: [50 60]
------------------------
Fold: 4
Training Data: [10 20 30 40 50 60 90 100]
Validation Data: [70 80]
------------------------
Fold: 5
Training Data: [10 20 30 40 50 60 70 80]
Validation Data: [90 100]
------------------------
```

| Statement | Explanation |
|---|---|
| `KFold(n_splits=5)` | Defines a five-fold splitting strategy |
| `kf.split(data)` | Generates training and validation indices |
| `train_index` | Stores indices of training records |
| `test_index` | Stores indices of validation records |
| `data[train_index]` | Retrieves actual training records |
| `data[test_index]` | Retrieves actual validation records |

> 💡 **Observation:** Each record is used four times for training and exactly once for validation. This example only demonstrates splitting; no model is trained.

### 🔹 3.4 Example 2: K-Fold using the Iris dataset

> 🎯 **Objective:** Apply five-fold cross-validation using a Decision Tree classifier.

| Dataset property | Value |
|---|---|
| Dataset | Iris |
| Total records | 150 |
| Input features | 4 |
| Target classes | 3 |
| Training records per iteration | 120 |
| Validation records per iteration | 30 |

The four input features are sepal length, sepal width, petal length and petal width. The classes are Setosa, Versicolor and Virginica.

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score

# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create the model
model = DecisionTreeClassifier(random_state=42)

# Define five folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Train and validate automatically
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

# Display results
print("Accuracy of each fold:", scores)
print("Average Accuracy:", scores.mean())
print("Standard Deviation:", scores.std())
```

#### 🟠 Output

```text
Accuracy of each fold: [1.         0.96666667 0.93333333 0.93333333 0.93333333]
Average Accuracy: 0.9533333333333335
Standard Deviation: 0.02666666666666666
```

| Statement | Explanation |
|---|---|
| `load_iris()` | Loads the Iris dataset |
| `iris.data` | Stores input features |
| `iris.target` | Stores target class labels |
| `DecisionTreeClassifier()` | Creates an untrained Decision Tree model |
| `KFold()` | Defines the splitting strategy |
| `cross_val_score()` | Trains and evaluates a fresh model for each fold |
| `cv=kf` | Specifies the splitting strategy |
| `scoring='accuracy'` | Selects accuracy as the evaluation metric |
| `scores.mean()` | Calculates average accuracy |
| `scores.std()` | Calculates variation in scores across folds |

> 🎯 **Result:** Average accuracy is 95.33%; standard deviation is 2.67 percentage points. A smaller standard deviation indicates more consistent fold scores, not necessarily a better model.

### 🔹 3.5 Advantages and disadvantages

| Advantages | Disadvantages |
|---|---|
| Better utilization of data | Higher computational cost |
| Reduces reliance on one split | Multiple model training iterations |
| Useful for model comparison | Does not guarantee class proportions |
| Applicable to classification and regression | May be unsuitable for imbalanced classification data |

---

## 🟢 4. Stratified K-Fold Cross-Validation

Stratified K-Fold maintains approximately the same proportion of target classes in each fold as in the original dataset. It is particularly useful for classification problems with imbalanced classes.

### 🔹 4.1 Why stratification is required

Consider a dataset of 100 patients: 80 healthy (class 0) and 20 infected (class 1).

Ordinary K-Fold might produce the following illustrative distribution:

| Fold | Healthy | Infected | Total |
|---|---:|---:|---:|
| F1 | 18 | 2 | 20 |
| F2 | 15 | 5 | 20 |
| F3 | 20 | 0 | 20 |
| F4 | 17 | 3 | 20 |
| F5 | 10 | 10 | 20 |

Fold 3 contains no infected patients, so it cannot assess how well the model identifies infected patients.

Stratified K-Fold maintains the 80:20 ratio in this example:

| Fold | Healthy | Infected | Total |
|---|---:|---:|---:|
| F1 | 16 | 4 | 20 |
| F2 | 16 | 4 | 20 |
| F3 | 16 | 4 | 20 |
| F4 | 16 | 4 | 20 |
| F5 | 16 | 4 | 20 |

**Class distribution visualization** (each square = 4 patients)

| Fold | Class distribution |
|---|---|
| F1 | 🟦🟦🟦🟦🟧 |
| F2 | 🟦🟦🟦🟦🟧 |
| F3 | 🟦🟦🟦🟦🟧 |
| F4 | 🟦🟦🟦🟦🟧 |
| F5 | 🟦🟦🟦🟦🟧 |

🟦 Healthy &nbsp; · &nbsp; 🟧 Infected

Stratification preserves the original proportions; it does not make an imbalanced dataset balanced.

### 🔹 4.2 Python Syntax

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

| Parameter | Explanation |
|---|---|
| `n_splits` | Number of folds |
| `shuffle` | Shuffles samples within each class before splitting |
| `random_state` | Makes random splitting reproducible |

Ordinary K-Fold uses `kf.split(X)`, while Stratified K-Fold uses `skf.split(X, y)` because it requires target labels.

### 🔹 4.3 Example 1: Stratified K-Fold using a simple dataset

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Create input data and target labels
X = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

skf = StratifiedKFold(n_splits=5)
fold = 1

for train_index, test_index in skf.split(X, y):
    print("Fold:", fold)
    print("Training Data:", X[train_index])
    print("Training Labels:", y[train_index])
    print("Validation Data:", X[test_index])
    print("Validation Labels:", y[test_index])
    print("-------------------------")
    fold = fold + 1
```

#### 🟠 Output

```text
Fold: 1
Training Data: [20 30 40 50 70 80 90 100]
Training Labels: [0 0 0 0 1 1 1 1]
Validation Data: [10 60]
Validation Labels: [0 1]
-------------------------
Fold: 2
Training Data: [10 30 40 50 60 80 90 100]
Training Labels: [0 0 0 0 1 1 1 1]
Validation Data: [20 70]
Validation Labels: [0 1]
-------------------------
Fold: 3
Training Data: [10 20 40 50 60 70 90 100]
Training Labels: [0 0 0 0 1 1 1 1]
Validation Data: [30 80]
Validation Labels: [0 1]
-------------------------
Fold: 4
Training Data: [10 20 30 50 60 70 80 100]
Training Labels: [0 0 0 0 1 1 1 1]
Validation Data: [40 90]
Validation Labels: [0 1]
-------------------------
Fold: 5
Training Data: [10 20 30 40 60 70 80 90]
Training Labels: [0 0 0 0 1 1 1 1]
Validation Data: [50 100]
Validation Labels: [0 1]
-------------------------
```

#### 🟠 Visual representation

```text
Class 0:  10   20   30   40   50
Class 1:  60   70   80   90  100
          |    |    |    |    |
Fold:     F1   F2   F3   F4   F5
```

| Class | F1 | F2 | F3 | F4 | F5 |
|---|---|---|---|---|---|
| 🟦 Class 0 | 10 | 20 | 30 | 40 | 50 |
| 🟧 Class 1 | 60 | 70 | 80 | 90 | 100 |

Every validation fold contains one record from each class, maintaining the 50:50 class distribution.

### 🔹 4.4 Example 2: Stratified K-Fold using the Iris dataset

| Target label | Species | Records | Validation records per fold |
|---|---|---:|---:|
| 0 | Setosa | 50 | 10 |
| 1 | Versicolor | 50 | 10 |
| 2 | Virginica | 50 | 10 |
| | Total | 150 | 30 |

Each iteration uses 120 training records (40 from each class) and 30 validation records (10 from each class).

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create the model
model = DecisionTreeClassifier(random_state=42)

# Define five stratified folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

print("Accuracy of each fold:", scores)
print("Average Accuracy:", scores.mean())
print("Standard Deviation:", scores.std())
```

#### 🟠 Output

```text
Accuracy of each fold: [1.         0.96666667 0.93333333 0.96666667 0.9       ]
Average Accuracy: 0.9533333333333335
Standard Deviation: 0.03399346342395189
```

> 🎯 **Result:** Average accuracy is 95.33%; standard deviation is 3.40 percentage points.

### 🔹 4.5 Advantages and disadvantages

| Advantages | Disadvantages |
|---|---|
| Preserves class proportions approximately | Requires target class labels |
| Suitable for imbalanced classification | Not directly applicable to regression |
| Provides representative folds | Requires enough minority-class samples to represent all classes in every fold |
| Supports reliable model comparison | Multiple training iterations |

---

## 🟣 5. Practical Comparison of Class Distribution

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, StratifiedKFold

X, y = load_iris(return_X_y=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("K-Fold Class Distribution")
for fold, (train, test) in enumerate(kf.split(X), 1):
    print("Fold", fold, ":", np.bincount(y[test], minlength=3))

print("\nStratified K-Fold Class Distribution")
for fold, (train, test) in enumerate(skf.split(X, y), 1):
    print("Fold", fold, ":", np.bincount(y[test], minlength=3))
```

### 🔹 Output

```text
K-Fold Class Distribution
Fold 1 : [10  9 11]
Fold 2 : [13 10  7]
Fold 3 : [12 10  8]
Fold 4 : [ 8 10 12]
Fold 5 : [ 7 11 12]

Stratified K-Fold Class Distribution
Fold 1 : [10 10 10]
Fold 2 : [10 10 10]
Fold 3 : [10 10 10]
Fold 4 : [10 10 10]
Fold 5 : [10 10 10]
```

`np.bincount()` counts occurrences of non-negative integer values. For example:

```python
import numpy as np
y = np.array([0, 0, 0, 1, 1, 2])
print(np.bincount(y))
```

Output: `[3 2 1]` (three class-0 records, two class-1 records and one class-2 record).

### 🔹 5.1 Visual comparison

Each letter represents one Iris record: S = Setosa, V = Versicolor, G = Virginica.

```text
K-Fold
F1  SSSSSSSSSS    VVVVVVVVV    GGGGGGGGGGG
F2  SSSSSSSSSSSSS VVVVVVVVVV   GGGGGGG
F3  SSSSSSSSSSSS  VVVVVVVVVV   GGGGGGGG
F4  SSSSSSSS      VVVVVVVVVV   GGGGGGGGGGGG
F5  SSSSSSS       VVVVVVVVVVV  GGGGGGGGGGGG

Stratified K-Fold
F1  SSSSSSSSSS    VVVVVVVVVV   GGGGGGGGGG
F2  SSSSSSSSSS    VVVVVVVVVV   GGGGGGGGGG
F3  SSSSSSSSSS    VVVVVVVVVV   GGGGGGGGGG
F4  SSSSSSSSSS    VVVVVVVVVV   GGGGGGGGGG
F5  SSSSSSSSSS    VVVVVVVVVV   GGGGGGGGGG
```

## 🟣 6. Difference Between K-Fold and Stratified K-Fold

| Parameter | K-Fold | Stratified K-Fold |
|---|---|---|
| Splitting criteria | Number of records | Number of records and class proportions |
| Class distribution | Not guaranteed | Approximately maintained |
| Target labels required for splitting | No | Yes |
| Classification | Applicable | Generally preferred |
| Regression | Applicable | Not directly applicable |
| Imbalanced datasets | Generally less suitable | More suitable |
| Number of iterations | K | K |
| Python class | `KFold()` | `StratifiedKFold()` |
| Splitting function | `kf.split(X)` | `skf.split(X, y)` |

## 🟣 7. Comparison of Experimental Results

| Metric | K-Fold | Stratified K-Fold |
|---|---:|---:|
| Fold 1 accuracy | 100% | 100% |
| Fold 2 accuracy | 96.67% | 96.67% |
| Fold 3 accuracy | 93.33% | 93.33% |
| Fold 4 accuracy | 93.33% | 96.67% |
| Fold 5 accuracy | 93.33% | 90% |
| Average accuracy | 95.33% | 95.33% |
| Standard deviation (percentage points) | 2.67 | 3.40 |

The Iris accuracy and class distribution results were verified by running the code. Results may vary with library versions or settings.

> 💡 **Important:** Stratified K-Fold does not necessarily increase accuracy. Its main purpose is to maintain class proportions.

## 🔵 8. Important Points to Remember

- `KFold()` and `StratifiedKFold()` define how the dataset is split.
- `cross_val_score()` automatically trains and validates fresh model instances across folds.
- Each record is used exactly once for validation in these techniques.
- Common choices for K are 5 and 10.
- Choose K so that each class has enough samples for stratification.
- Fit preprocessing operations only on training folds, preferably using a Scikit-learn Pipeline, to prevent data leakage.
- For model selection and tuning, ideally retain an independent test set for final evaluation.
- Ordinary random folds may be unsuitable for time-series or grouped data.

## 🔵 9. Practice Questions

1. Define K-Fold Cross-Validation with a suitable example.
2. Explain Stratified K-Fold using an imbalanced dataset.
3. Differentiate between K-Fold and Stratified K-Fold.
4. Implement 10-Fold Cross-Validation using the Iris dataset.
5. Implement Stratified K-Fold using Random Forest.
6. Explain `n_splits`, `shuffle` and `random_state`.
7. A dataset contains 500 records: 400 positive and 100 negative. Calculate the class distribution in each validation fold using five-fold Stratified Cross-Validation.

## 🔵 10. Conclusion

K-Fold divides records into approximately equal-sized folds, whereas Stratified K-Fold also preserves the original class proportions. Use K-Fold for general regression tasks and prefer Stratified K-Fold for ordinary classification tasks, particularly when classes are imbalanced.

# <span style="color:#1565C0;">Elastic Net Regression</span>

## <span style="color:#2E7D32;">Definition</span>

**Elastic Net Regression** is a regularized regression technique that combines:

<span style="color:#C62828;"><b>Lasso Regression (L1)</b></span> + 
<span style="color:#6A1B9A;"><b>Ridge Regression (L2)</b></span>

### <span style="color:#EF6C00;">Elastic Net = Lasso + Ridge</span>

---

## <span style="color:#2E7D32;">Why do we use Elastic Net?</span>

Elastic Net is useful when:

- there are **many input features**,
- some features are **highly correlated**,
- we want to **reduce overfitting**,
- and we want to reduce the effect of **less important features**.

### Example

Suppose we predict **student marks** using:

- Study Hours
- Attendance
- Practice Hours
- Assignment Marks

Study Hours and Practice Hours may be closely related. Elastic Net can handle such correlated features while controlling unnecessary coefficients.

---

## <span style="color:#1565C0;">Understanding `alpha`</span>

```python
ElasticNet(alpha=0.005, l1_ratio=0.9)
```

<span style="color:#D84315;"><b>alpha</b></span> controls the **overall strength of regularization**.

| Alpha | Meaning |
|---|---|
| Small `alpha` | Less regularization |
| Large `alpha` | More regularization |
| Very large `alpha` | Model may become too simple |

### Easy Memory Trick

> **alpha = How much regularization?**

---

## <span style="color:#6A1B9A;">Understanding `l1_ratio`</span>

<span style="color:#6A1B9A;"><b>l1_ratio</b></span> decides the balance between **Lasso (L1)** and **Ridge (L2)**.

| `l1_ratio` | Behaviour |
|---|---|
| `1.0` | Lasso side |
| `0.0` | Ridge side |
| `0.5` | Mix of Lasso and Ridge |
| `0.9` | Mostly Lasso, some Ridge |

### Easy Memory Trick

> **l1_ratio = What type of regularization?**

---

## <span style="color:#00838F;">Code Example</span>

```python
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the diabetes dataset
# X = input features, y = target values
X, y = load_diabetes(return_X_y=True)

# Split the data: 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

# Create Elastic Net model
# alpha controls total regularization strength
# l1_ratio controls the Lasso-Ridge balance
model = ElasticNet(alpha=0.005, l1_ratio=0.9)

# Train the model
model.fit(X_train, y_train)

# Predict test values
y_pred = model.predict(X_test)

# Evaluate using R² score
print("R² Score:", r2_score(y_test, y_pred))
```

---

## <span style="color:#C62828;">Quick Summary</span>

- **Elastic Net = Lasso + Ridge**
- **alpha** → How much regularization?
- **l1_ratio** → How much Lasso vs Ridge?
- It is useful for **overfitting, correlated features, and feature control**.

# 🎯 Confusion Matrix 

> 🟦 **Topic:** Evaluation of Classification Models  
> 🎓 **Level:** Beginners | Machine Learning  
> 💡 **Learning goal:** Understand *TP, TN, FP, FN* and calculate model performance.

---

## 🟣 1. What is a Confusion Matrix?

A **confusion matrix** is a table that compares the **actual class** (correct answer) with the **predicted class** (answer given by the ML model).

It helps answer two questions:

- ✅ How many predictions are correct?
- ❌ What types of mistakes does the model make?

**Example:** A hospital uses a machine learning model to predict whether a patient has a disease.

- **Positive (1)** = Disease present
- **Negative (0)** = Disease absent

> 🟡 **Remember:** Confusion matrices are used for **classification**, not ordinary regression.

## 🔵 2. Structure of the Confusion Matrix

In these notes, **rows = actual values** and **columns = predicted values**.

<table>
<tr><th rowspan="2">Actual class ↓</th><th colspan="2">Predicted class →</th></tr>
<tr><th>🩺 Positive (1)</th><th>💚 Negative (0)</th></tr>
<tr><th>Positive (1)</th><td style="background-color:#d6f5df;color:#135b2e"><b>TP — True Positive ✅</b><br>Correctly predicts disease</td><td style="background-color:#ffe0dc;color:#92231d"><b>FN — False Negative ❌</b><br>Misses an actual disease</td></tr>
<tr><th>Negative (0)</th><td style="background-color:#ffe0dc;color:#92231d"><b>FP — False Positive ❌</b><br>Incorrect disease alert</td><td style="background-color:#d6f5df;color:#135b2e"><b>TN — True Negative ✅</b><br>Correctly predicts no disease</td></tr>
</table>

> 🟢 **Green cells = correct predictions.** 🔴 **Red cells = incorrect predictions.**
>
> ⚠️ Other books/software may swap the order of rows or labels. Always check the axis labels rather than memorizing cell positions.

## 🟠 3. Four Important Terms

| Term | Actual | Predicted | Meaning |
|---|---|---|---|
| 🟢 **TP** (True Positive) | Disease | Disease | Correct positive prediction |
| 🟢 **TN** (True Negative) | No disease | No disease | Correct negative prediction |
| 🔴 **FP** (False Positive) | No disease | Disease | False alarm |
| 🔴 **FN** (False Negative) | Disease | No disease | Missed disease |

**Memory trick:** **True/False** tells whether the prediction is correct; **Positive/Negative** tells what the model predicted.

## 🟦 4. Worked Example — 100 Patients

Suppose a model tests **100 patients** and produces these results:

| Actual \ Predicted | Disease (Positive) | No Disease (Negative) | Total |
|---|---:|---:|---:|
| **Disease (Positive)** | 🟢 TP = 40 | 🔴 FN = 10 | 50 |
| **No Disease (Negative)** | 🔴 FP = 5 | 🟢 TN = 45 | 50 |
| **Total** | 45 | 55 | **100** |

**Interpretation:**

- ✅ 40 ill patients were correctly detected.
- ✅ 45 healthy patients were correctly identified.
- ❌ 5 healthy patients were incorrectly classified as ill.
- ❌ 10 ill patients were missed by the model.

Thus, **correct predictions = 40 + 45 = 85**, and **incorrect predictions = 5 + 10 = 15**.

## 🟪 5. Evaluation Metrics with Step-by-Step Calculations

> 🎓 **Important:** A ratio such as $40/45$ equals approximately $0.8889$. To express it as a percentage, multiply the ratio by $100\%$: $0.8889\times100\%\approx88.89\%$. We show this conversion explicitly in every calculation.

### 🟢 A. Accuracy — Overall correctness

**Question:** Of all patients, how many predictions were correct?

$$
\begin{aligned}
\text{Accuracy} &= \frac{TP+TN}{TP+TN+FP+FN}\\
&=\frac{40+45}{40+45+5+10}\\
&=\frac{85}{100}=0.85\\
\text{Accuracy (\%)} &=0.85\times100\%=\boxed{85.00\%}
\end{aligned}
$$

**Meaning:** The model correctly classifies 85 of 100 patients in this example.

### 🔵 B. Precision — Trustworthiness of positive predictions

**Question:** Among patients predicted to have the disease, how many really have it?

$$
\begin{aligned}
\text{Precision} &= \frac{TP}{TP+FP}\\
&=\frac{40}{40+5}=\frac{40}{45}\\
&\approx0.8889\\
\text{Precision (\%)} &\approx0.8889\times100\%=\boxed{88.89\%}
\end{aligned}
$$

**Meaning:** Of 45 positive predictions, 40 are correct.

### 🟠 C. Recall / Sensitivity — Ability to find actual positives

**Question:** Among patients who really have the disease, how many does the model detect?

$$
\begin{aligned}
\text{Recall} &= \text{Sensitivity}=\frac{TP}{TP+FN}\\
&=\frac{40}{40+10}=\frac{40}{50}=0.80\\
\text{Recall (\%)} &=0.80\times100\%=\boxed{80.00\%}
\end{aligned}
$$

**Meaning:** It detects 40 of 50 patients who actually have the disease.

### 🟡 D. Specificity — Ability to identify actual negatives

**Question:** Among healthy patients, how many are correctly classified as healthy?

$$
\begin{aligned}
\text{Specificity} &=\frac{TN}{TN+FP}\\
&=\frac{45}{45+5}=\frac{45}{50}=0.90\\
\text{Specificity (\%)} &=0.90\times100\%=\boxed{90.00\%}
\end{aligned}
$$

### 🟣 E. F1 Score — Balance between precision and recall

$$
\text{F1 Score}=\frac{2\times\text{Precision}\times\text{Recall}}{\text{Precision}+\text{Recall}}
$$

Or directly from the confusion matrix (using exact counts instead of rounded intermediate values):

$$
\begin{aligned}
\text{F1 Score} &=\frac{2TP}{2TP+FP+FN}\\
&=\frac{2\times40}{2\times40+5+10}\\
&=\frac{80}{95}\approx0.842105\\
\text{F1 Score (\%)} &\approx0.842105\times100\%=\boxed{84.21\%}
\end{aligned}
$$

### 📊 Results at a Glance

| Metric | Result | What it tells us |
|---|---:|---|
| 🟢 Accuracy | **85.00%** | Overall correct predictions |
| 🔵 Precision | **88.89%** | Correctness of positive alerts |
| 🟠 Recall / Sensitivity | **80.00%** | Detection of actual disease |
| 🟡 Specificity | **90.00%** | Recognition of healthy patients |
| 🟣 F1 Score | **84.21%** | Balance of precision and recall |

---

## 🔴 6. Why Accuracy Alone Can Be Misleading

Imagine **1,000 patients**: 990 are healthy and only 10 have a disease.

A model predicts **every patient is healthy**.

- Correct predictions = 990
- **Accuracy = 99%** ✅
- **Recall for the disease = 0%** ❌

> 🚨 **Key lesson:** The model looks excellent by accuracy but misses **every diseased patient**. For imbalanced data, examine recall, precision, F1 and the confusion matrix—not just accuracy.

## 🩺 7. False Positive vs False Negative in Real Life

**False Positive (FP):** A healthy patient receives a disease warning. This may lead to worry or extra tests.

**False Negative (FN):** A patient with disease is classified as healthy. This may delay further testing or treatment.

> 🟠 For **medical screening**, missing a true case can be serious, so **recall/sensitivity is often prioritized**. The right metric still depends on the real-world costs of both errors.

## 💻 8. Python Example Using Scikit-Learn

The following code produces the same example as above. In `sklearn`, labels `[0, 1]` yield rows/columns in the order **Negative, Positive**, so the raw matrix is `[[TN, FP], [FN, TP]]`.

```python
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score
)

# 40 TP, 10 FN, 5 FP, 45 TN
y_true = [1]*50 + [0]*50
y_pred = [1]*40 + [0]*10 + [1]*5 + [0]*45

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
print("Confusion matrix:\n", cm)
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1 score:", f1_score(y_true, y_pred))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Healthy (0)", "Disease (1)"]
).plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix — Disease Prediction")
plt.show()
```

**Expected raw matrix:**

```text
[[45  5]
 [10 40]]
```

Read it as: `[[TN, FP], [FN, TP]]`.

## 🧠 9. Quick Revision / Exam Tips

- 🟢 **TP and TN** are correct predictions.
- 🔴 **FP and FN** are wrong predictions.
- 🔵 **Precision** asks: *When the model says positive, can we trust it?*
- 🟠 **Recall** asks: *How many actual positives did the model catch?*
- 🟡 **Specificity** asks: *How many actual negatives did the model catch?*
- 🟣 **F1 score** combines precision and recall.
- ✅ All example metrics are for the **positive class (Disease = 1)**.

### ✍️ Practice Questions

1. What does a confusion matrix compare?
2. Explain the difference between FP and FN using an example.
3. If TP = 30, TN = 50, FP = 10 and FN = 10, calculate accuracy and recall.
4. Why can 99% accuracy be misleading in disease screening?
5. Which metric measures the proportion of actual negative cases identified correctly?

<details>
<summary>✅ Show short answers</summary>

1. Actual labels and predicted labels.
2. FP = false alarm; FN = missed actual positive.
3. Accuracy = (30 + 50) / 100 = **80%**; Recall = 30 / (30 + 10) = **75%**.
4. The model may predict only the majority class and miss rare positive cases.
5. **Specificity**.

</details>

---

> 🌈 **One-line summary:** A confusion matrix shows **what a classifier gets right, what it gets wrong, and what kinds of mistakes it makes**.

*Note: Colored table backgrounds use HTML styling. Color rendering depends on the Markdown viewer; emoji color cues and the tables remain readable when HTML styles are not supported.*

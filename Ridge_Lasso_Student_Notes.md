# 🎓 Ridge (L2) & Lasso (L1) Regularization --- Student Notes

> **Topic flow:** Underfitting → Overfitting → Balanced Fit → MSE →
> Regularization → Ridge (L2) → Lasso (L1)

------------------------------------------------------------------------

## 🟠 1. Underfitting

### 📘 Definition

**Underfitting** occurs when a machine learning model is **too simple to
learn the underlying pattern in the training data**, resulting in poor
performance on both training and test data.

> 💡 **Simple words:** The model has **not learned enough** from the
> data.

### 🎾 Example from the PPT

Suppose we predict **matches won based on a player's age** using only a
straight line:

$$
\text{Matches Won}=\theta_0+\theta_1(\text{Age})
$$

If the true relationship is curved, a straight line cannot capture it
properly. This is **underfitting**.

### 👨‍🎓 Real-life example

A student studies only basic definitions without understanding the
concepts. The student may perform poorly on both familiar and new
questions.

> 🟠 **UNDERFITTING = Too Simple / Not Learned Enough**

------------------------------------------------------------------------

## 🔴 2. Overfitting

### 📘 Definition

**Overfitting** occurs when a machine learning model **learns the
training data too closely, including noise and unnecessary details**,
resulting in very good training performance but poor performance on
new/test data.

> 💡 **Simple words:** The model **memorizes the training data instead
> of learning the general pattern**.

### 🎾 Example from the PPT

A more complicated polynomial can be written as:

$$
\text{Matches Won}=\theta_0+\theta_1x+\theta_2x^2+\theta_3x^3+\theta_4x^4
$$

Such a flexible curve can follow the training observations very closely.
If it also learns random variations/noise, it may fail on new players.
This is **overfitting**.

### 👨‍🎓 Real-life example

A student memorizes exact answers from previous papers. The student
performs very well if the same questions appear, but struggles when
questions are changed.

> 🔴 **OVERFITTING = Too Complex / Memorization**

### ⚖️ Quick comparison

  Feature                🟠 Underfitting         🔴 Overfitting
  ---------------------- ----------------------- ---------------------------
  Model                  Too simple              Too complex
  Learning               Does not learn enough   Learns even noise/details
  Training performance   Poor                    Very good
  Test performance       Poor                    Poor
  Typical issue          High bias               High variance

**Memory trick:**\
🟠 Underfitting: *"I did not study enough."*\
🔴 Overfitting: *"I memorized the answers but did not understand
them."*\
🟢 Good fit: *"I understood the concept and can answer new questions
too."*

------------------------------------------------------------------------

# 📊 PPT Explanation: Ridge (L2) and Lasso (L1)

## 🟦 Slide 1 --- Ridge (L2) and Lasso (L1)

The presentation introduces **regularization**, a technique used to
control model complexity. The two forms covered are:

-   **Ridge Regression = L2 Regularization**
-   **Lasso Regression = L1 Regularization**

Regularization adds a penalty for large coefficients so that the model
does not become unnecessarily complex.

------------------------------------------------------------------------

## 🟦 Slide 2 --- Different fitted models

The PPT compares different equations for predicting **matches won from
age**.

### Straight-line model

$$
\text{Matches Won}=\theta_0+\theta_1(\text{Age})
$$

This model is simple and assumes a straight-line relationship. If the
actual pattern is curved, it may be too simple.

### Higher-degree polynomial

$$
\text{Matches Won}=\theta_0+\theta_1x+\theta_2x^2+\theta_3x^3+\theta_4x^4
$$

This model is much more flexible. It can bend to follow the
observations closely, but excessive complexity can lead to overfitting.

### Quadratic model

$$
\text{Matches Won}=\theta_0+\theta_1x+\theta_2x^2
$$

This provides a smoother curve and represents the PPT's
**balanced-fit** example.

> 🟢 **Goal:** Capture the important pattern without making the model
> unnecessarily complex.

------------------------------------------------------------------------

## 🟦 Slide 3 --- Underfit, Overfit and Balanced Fit

-   **Underfit:** $y=\theta_0+\theta_1x$ --- too simple for the curved
    pattern.
-   **Overfit:** uses several polynomial terms and follows individual
    observations too closely.
-   **Balanced fit:** captures the important pattern in the data without becoming unnecessarily complex.

$$
y=\theta_0+\theta_1x+\theta_2x^2
$$

This quadratic model is more flexible than a straight-line model but less complex than a high-degree polynomial. Therefore, it can capture the general trend while reducing the risk of overfitting.

> 🎯 **Balanced Fit = Learn the important pattern without learning unnecessary noise.**

> 🎯 A useful model should **generalize** to unseen data rather than
> merely fit the training observations.

------------------------------------------------------------------------

## 🟦 Slide 4 --- Intuition behind Regularization

The PPT's complex model contains terms such as $x^3$ and $x^4$:

$$
y=\theta_0+\theta_1x+\theta_2x^2+\theta_3x^3+\theta_4x^4
$$

The slide suggests making $\theta_3$ and $\theta_4$ **almost close to
zero**. If these coefficients become very small, their corresponding
terms have little effect, and the model behaves more like a simpler
model.

> 💡 **Regularization idea:** Penalize large coefficients so that
> unnecessary coefficients are encouraged to become small.

------------------------------------------------------------------------

## 🟦 Slide 5 --- Mean Squared Error (MSE)

The regression error is represented using:

$$
\mathrm{MSE}=\frac{1}{n}\sum_{i=1}^{n}\left(y_i-\hat{y}_i\right)^2
$$

Where:

- $n$ = number of observations
- $y_i$ = actual value for observation $i$
- $\hat{y}_i$ = predicted value for observation $i$

### Example

If actual matches won = **20** and predicted matches won = **17**:

$$
\mathrm{Error}=20-17=3
$$

$$
\mathrm{Squared\ Error}=3^2=9
$$

Squaring prevents positive and negative errors from cancelling each
other and gives larger errors greater influence.

------------------------------------------------------------------------

## 🟦 Slide 6 --- Prediction Function $h_{\boldsymbol{\theta}}(x_i)$

The predicted value can be written as $h_{\boldsymbol{\theta}}(x_i)$:

$$
\mathrm{MSE}=\frac{1}{n}\sum_{i=1}^{n}\left(y_i-h_{\boldsymbol{\theta}}(x_i)\right)^2
$$

For a polynomial model:

$$
h_{\boldsymbol{\theta}}(x)=\theta_0+\theta_1x+\theta_2x^2+\theta_3x^3
$$

The $\theta$ values are **parameters/coefficient values learned during
training**. Normal regression adjusts them to reduce prediction error.
Regularization additionally penalizes large coefficients.

------------------------------------------------------------------------

## 🟣 Slide 7 --- Ridge Regression / L2 Regularization

Ridge adds a **squared-coefficient penalty** to the prediction error:

$$
\boxed{J(\boldsymbol{\theta})=\frac{1}{n}\sum_{i=1}^{n}\left(y_i-h_{\boldsymbol{\theta}}(x_i)\right)^2+\lambda\sum_{j=1}^{p}\theta_j^2}
$$

Here, $p$ is the number of predictor coefficients. The intercept $\theta_0$ is typically not included in the penalty term.

### Two parts

1.  **Prediction error:** make predictions close to actual values.
2.  **L2 penalty:** discourage unnecessarily large coefficients.

### What does $\lambda$ do?

The parameter $\lambda$ controls the **strength of regularization**.

1. **$\lambda=0$: No regularization**  
   No penalty is applied to the coefficients. The model behaves like ordinary regression.

2. **Small $\lambda$: Mild regularization**  
   A small penalty is applied. The coefficients are slightly reduced, helping control model complexity.

3. **Larger $\lambda$: Stronger regularization**  
   A larger penalty is applied. The coefficients are reduced more strongly, resulting in a simpler model.

4. **Excessively large $\lambda$: Risk of underfitting**  
   If $\lambda$ is too large, the coefficients may become too small. The model can become overly simple and may fail to capture important patterns in the data.

> 💡 **Memory Trick:**

$$
\lambda \uparrow \quad \Rightarrow \quad \text{Regularization Strength} \uparrow
$$

$$
\lambda \downarrow \quad \Rightarrow \quad \text{Regularization Strength} \downarrow
$$

> 🟣 **RIDGE = Shrinks coefficients, but generally does not make them
> exactly zero.**

------------------------------------------------------------------------

## 🟢 Slide 8 --- Lasso Regression / L1 Regularization

Lasso adds an **absolute-value penalty**:

$$
\boxed{J(\boldsymbol{\theta})=\frac{1}{n}\sum_{i=1}^{n}\left(y_i-h_{\boldsymbol{\theta}}(x_i)\right)^2+\lambda\sum_{j=1}^{p}|\theta_j|}
$$

As with Ridge, the penalty is applied to the predictor coefficients $\theta_1,\ldots,\theta_p$; the intercept $\theta_0$ is typically not penalized.

### Key difference

**Ridge (L2):**

$$
\lambda\sum_{j=1}^{p}\theta_j^2
$$

**Lasso (L1):**

$$
\lambda\sum_{j=1}^{p}|\theta_j|
$$

Lasso can make some coefficients **exactly zero**, so corresponding
features effectively disappear from the model. Therefore, Lasso can also
perform **feature selection**.

> 🟢 **LASSO = Shrink coefficients + potentially select features.**

------------------------------------------------------------------------

# ⭐ Final Ridge vs Lasso Summary

  ---------------------------------------------------------------------------
  Property                🟣 Ridge (L2)             🟢 Lasso (L1)
  ----------------------- ------------------------- -------------------------
  Penalty                 Squared coefficients      Absolute coefficients

  Formula                 $\lambda\sum_{j=1}^{p}\theta_j^2$   $\lambda\sum_{j=1}^{p}|\theta_j|$

  Shrinks coefficients    ✅ Yes                    ✅ Yes

  Coefficients can become Generally ❌              ✅ Yes
  exactly zero                                      

  Feature selection       Generally no              Yes

  Main idea               Reduce coefficient        Reduce and potentially
                          magnitude                 eliminate coefficients
  ---------------------------------------------------------------------------

## 🧠 Final Memory Trick

> 🟣 **Ridge = Reduce**\
> 🟢 **Lasso = Reduce + Remove**

$$
\boxed{L2=Ridge=\theta^2}
$$

$$
\boxed{L1=Lasso=|\theta|}
$$

------------------------------------------------------------------------

### 📌 Student Takeaway

A model should neither be **too simple (underfit)** nor **unnecessarily
complex (overfit)**. Ridge and Lasso regularization help control model
complexity by penalizing large coefficients.

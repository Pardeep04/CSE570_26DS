# Regression Evaluation: MAE, MSE, RMSE, and R-Squared

These four metrics evaluate predictions for a numerical target, such as marks, house prices, or temperature. MAE, MSE, and RMSE measure prediction error. R-squared compares the model's squared error with a mean-based baseline.

## 1. Notation and common example

- **y_i:** Actual value for observation i.
- **ŷ_i:** Predicted value for observation i.
- **ȳ:** Mean of the actual values in the evaluation set.
- **n:** Number of observations.
- **Residual:** Actual value minus predicted value, e_i = y_i − ŷ_i.

A positive residual indicates underprediction; a negative residual indicates overprediction.

| Student | Actual marks | Predicted marks | Residual | Absolute error | Squared error |
|---|---:|---:|---:|---:|---:|
| 1 | 50 | 48 | 2 | 2 | 4 |
| 2 | 60 | 63 | −3 | 3 | 9 |
| 3 | 70 | 65 | 5 | 5 | 25 |
| **Total** | | | | **10** | **38** |

Simply averaging signed errors can be misleading because positive and negative errors cancel. Absolute values and squares prevent this cancellation.

## 2. MAE: Mean Absolute Error

**Definition:** The average of the absolute differences between actual and predicted values.

$$
\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

Plain-text formula: MAE = sum of absolute errors / number of observations.

### Worked calculation

$$
\mathrm{MAE}=\frac{|50-48|+|60-63|+|70-65|}{3}
=\frac{2+3+5}{3}=3.33\text{ marks}
$$

**Interpretation:** The predictions differ from the actual marks by 3.33 marks on average, regardless of direction.

### Key properties

- Lower is better; zero means perfect predictions.
- It uses the same units as the target.
- Each additional unit of absolute error adds the same amount to the total absolute error.
- It is less sensitive to extreme errors than squared-error measures, but it is not unaffected by outliers.

**Useful when:** An understandable average error is needed and the cost of error grows approximately linearly.

## 3. MSE: Mean Squared Error

**Definition:** The average of the squared differences between actual and predicted values.

$$
\mathrm{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

Plain-text formula: MSE = sum of squared errors / number of observations.

### Worked calculation

$$
\mathrm{MSE}=\frac{2^2+(-3)^2+5^2}{3}
=\frac{4+9+25}{3}=12.67\text{ marks}^2
$$

### Why square errors?

Squaring gives larger errors greater influence. An error of 2 contributes 4; an error of 10 contributes 100. The error is five times larger, but its squared contribution is 25 times larger.

### Key properties

- Lower is better; zero means perfect predictions.
- Its units are the square of the target units.
- Large errors have a strong influence.
- It is commonly used as a regression training objective because it is differentiable with respect to predictions.

**Useful when:** Large errors should receive a disproportionately large penalty.

**Interpretation caution:** MSE = 12.67 marks² does not mean predictions are off by 12.67 marks on average.

## 4. RMSE: Root Mean Squared Error

**Definition:** The square root of MSE.

$$
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
=\sqrt{\mathrm{MSE}}
$$

Plain-text formula: RMSE = square root of the average squared error.

### Worked calculation

$$
\mathrm{RMSE}=\sqrt{\frac{38}{3}}\approx3.56\text{ marks}
$$

**Interpretation:** RMSE expresses error in marks while retaining the stronger influence of large errors. It is not the ordinary average absolute error; that is MAE.

### Key properties

- Lower is better; zero means perfect predictions.
- It has the same units as the target.
- For the same observations and equal weighting, RMSE is always greater than or equal to MAE.
- RMSE equals MAE when all absolute errors are equal.
- MSE and RMSE rank models identically on the same evaluation data because the square-root function preserves order.

**Useful when:** Large mistakes matter, but results should remain interpretable in the target's original units.

## 5. Why MAE and RMSE can favour different models

| Model | Absolute errors | MAE | MSE | RMSE |
|---|---|---:|---:|---:|
| A | 3, 3, 3 | 3.00 | 9.00 | 3.00 |
| B | 0, 0, 9 | 3.00 | 27.00 | 5.20 |

Both models have the same average absolute error. Model B makes one much larger mistake, so its MSE and RMSE are higher.

A model with lower MAE does not necessarily have lower RMSE. Choose a metric according to how much large errors matter in the application.

## 6. R-Squared: Coefficient of Determination

**Definition:** R-squared measures the reduction in squared error relative to predicting the mean of the actual values in the evaluation set.

$$
R^2=1-\frac{SS_{\mathrm{res}}}{SS_{\mathrm{tot}}}
$$

where:

$$
SS_{\mathrm{res}}=\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

$$
SS_{\mathrm{tot}}=\sum_{i=1}^{n}(y_i-\bar{y})^2
$$

Plain-text formula: R² = 1 − (model squared error / mean-baseline squared error).

### Step 1: Find the actual-value mean

$$
\bar{y}=\frac{50+60+70}{3}=60
$$

The mean-based reference predicts 60 marks for every student.

### Step 2: Calculate the baseline squared error

| Actual marks | Baseline prediction | Baseline squared error |
|---:|---:|---:|
| 50 | 60 | 100 |
| 60 | 60 | 0 |
| 70 | 60 | 100 |
| **Total** | | **200** |

Thus, SS_tot = 200. The model's squared error, SS_res, is 38.

### Step 3: Calculate R-squared

$$
R^2=1-\frac{38}{200}=0.81
$$

**Interpretation:** On these observations, the model reduces squared error by 81% relative to the mean baseline. This is commonly described as explaining 81% of the variation in marks.

**R² = 0.81 does not mean that 81% of predictions are correct.**

### Interpreting R-squared values

| Value | Interpretation for a nonconstant target |
|---|---|
| R² = 1 | Perfect predictions |
| R² = 0.81 | Squared error is 81% lower than the mean baseline |
| R² = 0 | Same squared error as the mean baseline |
| R² < 0 | Greater squared error than the mean baseline |

For example, if SS_res = 300 and SS_tot = 200:

$$
R^2=1-\frac{300}{200}=-0.5
$$

The model performs worse than predicting the mean.

### Important limitations

- Higher is better when comparing predictions for the same target on the same evaluation observations.
- R-squared has no units and has a maximum of 1, but no finite lower bound.
- A high training R-squared does not guarantee good performance on unseen data.
- It does not establish causation or prove that a model is appropriate.
- Its value depends on the variation in the evaluation target, so comparisons across different datasets require care.
- If all actual values are identical, SS_tot = 0 and the standard formula is undefined. Software may apply special handling.
- The evaluation-set mean is used to define the R-squared reference. This does not mean test targets should be used to train a deployable baseline model.

### Brief note on adjusted R-squared

For ordinary least squares with an intercept, adding predictors cannot reduce training R-squared on the same observations, even when the extra predictors are not useful.

Adjusted R-squared accounts for the number of predictors:

$$
\bar{R}^2=1-(1-R^2)\frac{n-1}{n-p-1}
$$

Here, p is the number of predictor terms excluding the intercept, and n must exceed p + 1. Adjusted R-squared can decrease when an added predictor contributes too little improvement. It does not replace evaluation on unseen data.

## 7. Comparison table

| Property | MAE | MSE | RMSE | R-squared |
|---|---|---|---|---|
| Main idea | Average absolute error | Average squared error | Square root of MSE | Improvement over mean baseline |
| Better direction | Lower | Lower | Lower | Higher |
| Ideal value | 0 | 0 | 0 | 1 |
| Units | Target units | Squared target units | Target units | Unitless |
| Can be negative? | No | No | No | Yes |
| Large errors | Linear contribution | Squared contribution | Based on squared errors | Based on squared errors |
| Example result | 3.33 marks | 12.67 marks² | 3.56 marks | 0.81 |

Do not compare the raw numerical sizes of different metrics to decide which is better. They have different scales and meanings.

## 8. Complete Python example

This example uses standard Python and requires no additional packages.

```python
from math import sqrt

actual = [50, 60, 70]
predicted = [48, 63, 65]

if len(actual) != len(predicted) or len(actual) == 0:
    raise ValueError("Inputs must have equal, nonzero lengths.")

n = len(actual)
errors = [y - y_hat for y, y_hat in zip(actual, predicted)]

mae = sum(abs(error) for error in errors) / n
ss_res = sum(error ** 2 for error in errors)
mse = ss_res / n
rmse = sqrt(mse)

actual_mean = sum(actual) / n
ss_tot = sum((y - actual_mean) ** 2 for y in actual)

# The standard R-squared formula requires target variation.
r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R-squared: {r_squared:.2f}")
```

Expected output:

```text
MAE: 3.33
MSE: 12.67
RMSE: 3.56
R-squared: 0.81
```

## 9. Quick exam revision

- **MAE:** Absolute value → Average.
- **MSE:** Square → Average.
- **RMSE:** Square → Average → Square root.
- **R-squared:** 1 − model squared error / mean-baseline squared error.
- MAE, MSE, and RMSE are nonnegative; R-squared can be negative.
- RMSE ≥ MAE for the same errors and equal weighting.
- MSE and RMSE give identical model rankings on the same data.
- For a fixed nonconstant evaluation target, lower MSE also means higher R-squared.
- R-squared is not classification accuracy.
- Evaluate generalization on held-out data, not only training data.
- There is no universal acceptable MAE or RMSE: judge them using target units, a baseline, and the practical cost of mistakes.

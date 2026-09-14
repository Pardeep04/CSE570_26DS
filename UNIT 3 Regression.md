# 📘 Linear Regression: Student Notes

### Simple regression • Least squares • Multiple regression • Dummy variables • Polynomial regression

**Based on:** the supplied 29-slide presentation, *Linear Regression.pptx*. These notes expand the slide concepts with original teaching examples, step-by-step calculations, revision tables, and practice questions. All numerical scenarios below are illustrative.

> 🟦 **Concept** = explanation &nbsp; 🟩 **Remember** = key learning &nbsp; 🟨 **Example** = application &nbsp; 🟥 **Watch out** = common mistake

> **Viewing tip:** Open this file in a Markdown preview. Colored symbols work in most viewers. Colored headings use optional HTML styling, which some viewers remove. Equations display best in a viewer that supports LaTeX math.

---

## <span style="color:#2563eb">🟦 1. What will you learn?</span>

By the end, you should be able to:

- Identify input features and a numerical target.
- Explain the intercept, slope, predicted value, and residual.
- Calculate a simple regression line using ordinary least squares.
- Use several features together to make a prediction.
- Encode categorical features and explain the dummy variable trap.
- Explain why polynomial regression can draw a curve while remaining linear in its coefficients.
- Recognize basic model limitations and evaluate prediction errors.

## <span style="color:#2563eb">🟦 2. What is regression?</span>

**Regression is a method for estimating a numerical outcome from one or more input variables.** It learns a relationship from examples where both the inputs and the correct outcomes are known. This makes regression a form of **supervised learning**.

Imagine that a company wants to estimate a salary for a person with four years of experience. It has historical records of employees' experience and salary. Regression uses those records to estimate how salary changes with experience.

| Scenario | Input feature(s) | Numerical target |
|---|---|---|
| Salary estimation | Years of experience | Salary in ₹ |
| House price estimation | Area, age, location | Price in ₹ |
| Student performance | Study hours, attendance | Marks |
| Business planning | R&D, administration, marketing expenditure | Profit |

**Regression versus classification:** predicting a student's numerical marks is regression. Predicting the category *pass* or *fail* is classification.

### Essential vocabulary

| Term | Symbol | Meaning |
|---|---|---|
| Independent variable / feature / predictor | $x$ | An input used for prediction |
| Dependent variable / target / response | $y$ | The actual outcome we want to predict |
| Predicted value | $\hat{y}$ | The model's estimated outcome; pronounced “y-hat” |
| Observation | $(x_i,y_i)$ | One recorded example |
| Coefficient | $b_j$ | A number learned from the data |
| Residual | $e_i=y_i-\hat{y}_i$ | The difference between an actual and a fitted value |

> 🟩 **Remember:** “Independent variable” is a traditional name for an input. It does not mean that all input features must be statistically independent of each other.

## <span style="color:#2563eb">🟦 3. Simple linear regression</span>

**Simple linear regression uses one input feature and fits a straight-line relationship.**

$$
\hat{y}=b_0+b_1x
$$

| Part | Name | Meaning |
|---|---|---|
| $\hat{y}$ | Prediction | Estimated target value |
| $b_0$ | Intercept | Predicted target when $x=0$ |
| $b_1$ | Slope | Change in the prediction when $x$ increases by one unit |
| $x$ | Input | The known feature value |

The observed outcome will usually differ from the prediction. A statistical model therefore includes an error term:

$$
y=\beta_0+\beta_1x+\varepsilon
$$

Here, $\varepsilon$ represents variation the model does not explain. The fitted values $b_0$ and $b_1$ estimate the unknown population coefficients $\beta_0$ and $\beta_1$.

### 🟨 Example: salary and experience

Suppose a fitted model is:

$$
\widehat{\text{Salary}}=25{,}000+5{,}000\times\text{Experience}
$$

Salary is **monthly salary in ₹**, and experience is **years**.

- **Intercept = ₹25,000:** the model predicts ₹25,000 per month at zero years of experience.
- **Slope = ₹5,000 per year of experience:** one additional year of experience increases the predicted monthly salary by ₹5,000.

For four years of experience:

$$
\widehat{\text{Salary}}=25{,}000+5{,}000(4)=₹45{,}000
$$

| Experience (years) | Calculation | Predicted monthly salary |
|---:|---|---:|
| 0 | 25,000 + 5,000 × 0 | ₹25,000 |
| 1 | 25,000 + 5,000 × 1 | ₹30,000 |
| 2 | 25,000 + 5,000 × 2 | ₹35,000 |
| 4 | 25,000 + 5,000 × 4 | ₹45,000 |

> 🟥 **Watch out:** This slope describes a fitted association. It does not guarantee a pay increase for every employee. Education, job role, employer, and other factors may also affect salary.

### What does the slope's sign tell us?

| Slope | Meaning | Example |
|---|---|---|
| Positive | Prediction increases as the input increases | Salary may increase with experience |
| Negative | Prediction decreases as the input increases | A used car's price may decrease with age |
| Zero | The fitted line is horizontal | This fitted model predicts no change with the input |

**Intercept caution:** If a house-price model uses floor area, its intercept predicts the price at zero area. That may have no useful practical interpretation. The intercept still helps position the fitted line.

## <span style="color:#2563eb">🟦 4. Actual values, predictions, and residuals</span>

Suppose the model predicts ₹45,000 for an employee whose actual salary is ₹48,000.

$$
e=y-\hat{y}=48{,}000-45{,}000=₹3{,}000
$$

The residual is positive because the model **underpredicted** the salary.

| Actual value | Prediction | Residual | Interpretation |
|---:|---:|---:|---|
| ₹48,000 | ₹45,000 | +₹3,000 | Underprediction |
| ₹42,000 | ₹45,000 | −₹3,000 | Overprediction |
| ₹45,000 | ₹45,000 | ₹0 | Exact prediction for this observation |

On a scatter plot, the actual observation is a point and the prediction lies on the regression line at the same $x$ value. The residual is the **vertical difference** between them.

> 🟩 **Remember:** Ordinary least squares minimizes squared vertical residuals. It does not minimize perpendicular distances from points to the line.

## <span style="color:#2563eb">🟦 5. Ordinary least squares: finding the best-fit line</span>

Many straight lines can be drawn through a scatter plot. **Ordinary least squares (OLS)** selects coefficients that minimize the **sum of squared errors**, also called the **residual sum of squares**.

$$
\text{SSE}=\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
=\sum_{i=1}^{n}\left[y_i-(b_0+b_1x_i)\right]^2
$$

The symbol $\sum$ means “add across all observations,” and $n$ is the number of observations.

### Why square the residuals?

1. Positive and negative residuals cannot cancel. Residuals of +3 and −3 add to zero, but their squares add to 18.
2. Large errors receive a larger penalty. An error of 10 contributes 100, while an error of 2 contributes 4.
3. The squared-error objective supports convenient mathematical optimization.

**Trade-off:** the large penalty also makes OLS sensitive to outliers.

### 🟨 Worked example: calculate a regression line by hand

Use this small illustrative dataset:

| Student | Study hours $x$ | Marks $y$ |
|---|---:|---:|
| A | 1 | 2 |
| B | 2 | 4 |
| C | 3 | 5 |

**Step 1: Calculate the means.**

$$
\bar{x}=\frac{1+2+3}{3}=2,
\qquad
\bar{y}=\frac{2+4+5}{3}=\frac{11}{3}\approx3.667
$$

**Step 2: Calculate the slope.**

For simple OLS with an intercept:

$$
b_1=\frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sum(x_i-\bar{x})^2}
$$

The numerator measures how $x$ and $y$ vary together. The denominator measures the spread of $x$.

| $x$ | $y$ | $x-\bar{x}$ | $y-\bar{y}$ | Product of deviations | $(x-\bar{x})^2$ |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | −1 | −5/3 | 5/3 | 1 |
| 2 | 4 | 0 | 1/3 | 0 | 0 |
| 3 | 5 | 1 | 4/3 | 4/3 | 1 |
| **Total** | | | | **3** | **2** |

$$
b_1=\frac{3}{2}=1.5
$$

**Step 3: Calculate the intercept.**

$$
b_0=\bar{y}-b_1\bar{x}=\frac{11}{3}-1.5(2)=\frac{2}{3}
$$

**Step 4: Write the fitted model.**

$$
\boxed{\hat{y}=\frac{2}{3}+1.5x}
$$

**Step 5: Calculate predictions and squared residuals.**

| $x$ | Actual $y$ | Prediction $\hat{y}$ | Residual $y-\hat{y}$ | Squared residual |
|---:|---:|---:|---:|---:|
| 1 | 2 | 13/6 ≈ 2.167 | −1/6 | 1/36 |
| 2 | 4 | 11/3 ≈ 3.667 | 1/3 | 1/9 |
| 3 | 5 | 31/6 ≈ 5.167 | −1/6 | 1/36 |

$$
\text{SSE}=\frac{1}{36}+\frac{1}{9}+\frac{1}{36}=\frac{1}{6}\approx0.167
$$

For comparison, the candidate line $\hat{y}=2x$ predicts 2, 4, and 6. Its SSE is $0^2+0^2+(-1)^2=1$. OLS gives a smaller SSE, so it fits these observations better by this criterion.

> 🟩 **Remember:** A best-fit line need not pass through every point. With an intercept, the OLS line passes through $(\bar{x},\bar{y})$.

> 🟥 **Watch out:** If every input has the same value, the slope formula's denominator is zero. The data then cannot identify a separate slope and intercept.

### How are the coefficients learned in software?

OLS specifies the **objective** to minimize. Software can solve it using linear algebra. An iterative method such as gradient descent can also minimize the same objective by repeatedly adjusting coefficients. Gradient descent is not required for every linear regression fit.

## <span style="color:#7c3aed">🟪 6. Multiple linear regression</span>

**Multiple linear regression uses two or more input features to predict one numerical target.**

$$
\hat{y}=b_0+b_1x_1+b_2x_2+\cdots+b_px_p
$$

Here, $p$ is the number of input features. The model adds each feature's weighted contribution to the intercept.

### 🟨 Example: estimating business profit

Suppose a fitted model is:

$$
\widehat{\text{Profit}}=10+0.8R+0.2M-0.1A
$$

All monetary values are in **₹ lakh**. $R$ is R&D spending, $M$ is marketing spending, and $A$ is administration spending.

For $R=20$, $M=10$, and $A=5$:

$$
\widehat{\text{Profit}}=10+0.8(20)+0.2(10)-0.1(5)
=10+16+2-0.5=₹27.5\text{ lakh}
$$

### Interpreting coefficients correctly

| Coefficient | Interpretation within this fitted model |
|---|---|
| $0.8$ for R&D | Increasing R&D by ₹1 lakh increases predicted profit by ₹0.8 lakh, **holding marketing and administration constant** |
| $0.2$ for marketing | Increasing marketing by ₹1 lakh increases predicted profit by ₹0.2 lakh, holding the other inputs constant |
| $-0.1$ for administration | Increasing administration by ₹1 lakh decreases predicted profit by ₹0.1 lakh, holding the other inputs constant |

> 🟩 **Remember:** “Holding other inputs constant” is essential when interpreting a multiple regression coefficient.

> 🟥 **Watch out:** These coefficients describe the fitted data relationship. They do not establish that changing a company's spending will cause exactly these profit changes.

### Simple versus multiple regression

| Comparison | Simple regression | Multiple regression |
|---|---|---|
| Number of inputs | One | Two or more |
| Example | Salary from experience | Salary from experience and education |
| Geometry | A line in an input–target plot | A plane for two inputs, or a hyperplane for more |
| Slope interpretation | Change associated with the single input | Change associated with one input while the others stay fixed |

## <span style="color:#7c3aed">🟪 7. Dummy variables: representing categories with numbers</span>

A **dummy variable** converts a category into a number that a regression model can use.

- **1 means yes:** the observation belongs to that category.
- **0 means no:** the observation does not belong to that category.

For example, a house-price model may use **area** and **city**. Area is already numerical, but city names need a numerical representation.

### 🟨 Step 1: Convert the city into a dummy variable

Suppose there are only two cities: **Delhi and Mumbai**. Create one variable called `Mumbai`.

| House | City | Area (sq. ft.) | Mumbai dummy |
|---|---|---:|---:|
| A | Delhi | 1,000 | 0 |
| B | Mumbai | 1,000 | 1 |
| C | Delhi | 1,500 | 0 |
| D | Mumbai | 1,500 | 1 |

**Delhi is the reference category:** the category used as the starting point for comparison.

> 🟩 **Remember:** In this two-city example, `Mumbai = 0` means Delhi. It does not mean that the city is missing.

### 🟨 Step 2: Use the dummy variable in a regression equation

Suppose an illustrative fitted model is:

$$
\widehat{\text{Price}}=10+0.04(\text{Area})+20(\text{Mumbai})
$$

Price is measured in **₹ lakh**, and area is measured in **square feet**. These numbers are for learning, not actual property-price estimates.

For a **1,000 sq. ft. house in Delhi**, the Mumbai dummy is 0:

$$
\widehat{\text{Price}}=10+0.04(1000)+20(0)=10+40+0=\boxed{₹50\text{ lakh}}
$$

For a **1,000 sq. ft. house in Mumbai**, the Mumbai dummy is 1:

$$
\widehat{\text{Price}}=10+0.04(1000)+20(1)=10+40+20=\boxed{₹70\text{ lakh}}
$$

The dummy variable works like an **on/off switch** for the Mumbai contribution.

| City | Dummy contribution | Predicted price |
|---|---:|---:|
| Delhi | 20 × 0 = 0 | ₹50 lakh |
| Mumbai | 20 × 1 = 20 | ₹70 lakh |

**Meaning of the coefficient 20:** For houses with the **same area**, the model predicts Mumbai prices to be **₹20 lakh higher than Delhi prices**.

### 🟨 Step 3: Extend the idea to three cities

Suppose the cities are **Delhi, Mumbai, and Chennai**. Choose Delhi as the reference and create two dummy variables:

| City | Mumbai dummy | Chennai dummy |
|---|---:|---:|
| Delhi | 0 | 0 |
| Mumbai | 1 | 0 |
| Chennai | 0 | 1 |

**Both zero means Delhi.** Every city can be identified even though Delhi has no separate dummy column.

For example:

$$
\widehat{\text{Price}}=10+0.04(\text{Area})+20(\text{Mumbai})+8(\text{Chennai})
$$

For an area of 1,000 sq. ft.:

| City | Calculation | Predicted price |
|---|---|---:|
| Delhi | 10 + 40 + 0 + 0 | ₹50 lakh |
| Mumbai | 10 + 40 + 20 + 0 | ₹70 lakh |
| Chennai | 10 + 40 + 0 + 8 | ₹58 lakh |

Each dummy coefficient describes a difference **from Delhi**, keeping area the same.

### Why not use Delhi = 1, Mumbai = 2, Chennai = 3?

Using those codes as one numerical input would tell the model that the cities have an order and equal numerical steps. It would assign the same predicted change to moving from Delhi to Mumbai as from Mumbai to Chennai.

City names do not naturally have this relationship. **Dummy variables let each city have its own estimated difference from the reference city.**

## <span style="color:#dc2626">🟥 8. Why do we use k − 1 dummy variables?</span>

Here, **k means the number of categories**.

> 🟩 **Main idea:** One category acts as the reference. We need only **k − 1 dummy variables** to represent the remaining categories when the model includes an intercept.

### 🟦 Reason 1: The last category is already known

For two cities, one dummy is enough:

| Mumbai dummy | City identified |
|---:|---|
| 0 | Delhi |
| 1 | Mumbai |

If Mumbai is 0, the city must be Delhi. A second column would repeat information already available.

A simple city-only model is:

$$
\widehat{\text{Price}}=b_0+b_1(\text{Mumbai})
$$

- **Delhi:** the prediction is $b_0$.
- **Mumbai:** the prediction is $b_0+b_1$.

The **intercept gives the reference city's prediction**, and the dummy coefficient gives the difference for Mumbai. When other inputs such as area are included, their contributions are added for both cities.

### 🟨 What does “fitted price” mean?

**Fitted price is the price estimated by the trained regression model for an observation in the training data.** “Fitting” means learning the model's coefficients from that data.

Suppose the training data contains:

| City | Actual house prices |
|---|---|
| Delhi | ₹48 lakh, ₹52 lakh |
| Mumbai | ₹68 lakh, ₹72 lakh |

If we fit an **ordinary least-squares model using only city**, it learns each city's average:

- **Delhi:** (48 + 52) ÷ 2 = **₹50 lakh**.
- **Mumbai:** (68 + 72) ÷ 2 = **₹70 lakh**.

These averages become the model's **fitted prices** for houses in the corresponding cities.

| House | City | Actual price | Fitted price |
|---|---|---:|---:|
| A | Delhi | ₹48 lakh | ₹50 lakh |
| B | Delhi | ₹52 lakh | ₹50 lakh |
| C | Mumbai | ₹68 lakh | ₹70 lakh |
| D | Mumbai | ₹72 lakh | ₹70 lakh |

For House A:

- **Actual price:** ₹48 lakh, the recorded price.
- **Fitted price:** ₹50 lakh, the model's estimate.
- **Residual:** actual − fitted = 48 − 50 = **−₹2 lakh**.

> 🟩 **Remember:** “Fitted” means calculated by the model after learning from the data. A fitted price does not have to equal the actual price.

**Why the same fitted price within each city?** This example uses only city. If the model also included features such as area and age, houses in the same city could have different fitted prices. Fitted prices are not always simple city averages.

### 🟥 Reason 2: All dummies plus an intercept create a problem

The problem is that **the model cannot decide how much of the price belongs to the intercept and how much belongs to the city coefficient**.

Let us use only two cities and ignore other features for now. All prices below are in **₹ lakh**.

#### 🟦 1. What does the model calculate?

$$
\widehat{\text{Price}}=b_0+b_1D_{\text{Delhi}}+b_2D_{\text{Mumbai}}
$$

The dummy values are:

| City | Delhi dummy | Mumbai dummy |
|---|---:|---:|
| Delhi | 1 | 0 |
| Mumbai | 0 | 1 |

For **Delhi**, the model becomes:

$$
\widehat{\text{Price}}=b_0+b_1(1)+b_2(0)=b_0+b_1
$$

For **Mumbai**, it becomes:

$$
\widehat{\text{Price}}=b_0+b_1(0)+b_2(1)=b_0+b_2
$$

If the fitted prices are ₹50 lakh and ₹70 lakh, we need:

$$
b_0+b_1=50
$$

$$
b_0+b_2=70
$$

#### 🟨 2. We have two equations but three unknowns

The unknowns are **$b_0$, $b_1$, and $b_2$**.

We can choose any intercept and adjust the other coefficients to compensate:

| Choose intercept $b_0$ | Delhi coefficient $b_1=50-b_0$ | Mumbai coefficient $b_2=70-b_0$ |
|---:|---:|---:|
| 0 | 50 | 70 |
| 10 | 40 | 60 |
| 30 | 20 | 40 |
| 50 | 0 | 20 |

Every row gives the same predictions.

For example:

- First row: Delhi = 0 + 50 = **50**, Mumbai = 0 + 70 = **70**.
- Second row: Delhi = 10 + 40 = **50**, Mumbai = 10 + 60 = **70**.

> 🟩 **The model knows the totals, but it cannot uniquely divide each total into “intercept + city effect.”**

It is like asking: **“Two numbers add up to 50. What are the two numbers?”** They could be 10 and 40, 20 and 30, or many other pairs.

#### 🟥 3. Why is the information redundant?

The intercept can be written as $b_0\times1$. Therefore, the model's input columns look like this:

| City | Intercept column | Delhi dummy | Mumbai dummy |
|---|---:|---:|---:|
| Delhi | 1 | 1 | 0 |
| Mumbai | 1 | 0 | 1 |

In every row:

$$
\text{Intercept column}=\text{Delhi dummy}+\text{Mumbai dummy}
$$

One column can be calculated exactly from the other two. That is **perfect multicollinearity**. In this categorical encoding setup, it is called the **dummy variable trap**.

Adding more observations from these same two cities does not remove this exact relationship.

#### 🟩 4. How does removing one dummy solve it?

Remove the Delhi dummy and use Delhi as the reference:

$$
\widehat{\text{Price}}=b_0+b_2D_{\text{Mumbai}}
$$

**For Delhi**, Mumbai = 0:

$$
50=b_0+b_2(0)\quad\Rightarrow\quad b_0=50
$$

**For Mumbai**, Mumbai = 1:

$$
70=50+b_2\quad\Rightarrow\quad b_2=20
$$

The model is:

$$
\boxed{\widehat{\text{Price}}=50+20D_{\text{Mumbai}}}
$$

The coefficients now have clear meanings:

- **50:** Delhi's baseline price.
- **20:** Mumbai's additional price compared with Delhi.

**We still represent both cities. We have removed the redundant column, not the Delhi observations.**

#### 🟨 5. Connecting this model to fitted prices

**Fitted price means the price predicted by a regression model for a training observation after it has learned from the training data.**

| Term | Meaning |
|---|---|
| **Actual price $y$** | The recorded selling price |
| **Fitted price $\hat{y}$** | The model's estimated price for that training observation |

Using the model above:

- **Delhi:** 50 + 20(0) = **₹50 lakh**.
- **Mumbai:** 50 + 20(1) = **₹70 lakh**.

These are the model's **fitted prices**. Individual actual prices may differ, as shown in the house-price table just before this explanation. Because this model uses **only city**, it assigns the same fitted price to every house in the same city.

> 🟩 **“Fitted prices of ₹50 lakh and ₹70 lakh” simply means “the prices the model predicts for Delhi and Mumbai.”**

### Quick counting examples

| Number of categories (k) | Dummy variables with an intercept (k − 1) | How the reference is represented |
|---:|---:|---|
| 2 | 1 | The dummy is 0 |
| 3 | 2 | Both dummies are 0 |
| 4 | 3 | All three dummies are 0 |

> 🟩 **Memory sentence:** **k categories = one reference category + k − 1 comparisons.** Omitting a dummy column does **not** remove that category's observations from the dataset.

**Small exception:** For a single categorical feature, you can use all k dummy variables if you remove the intercept. The usual beginner-friendly approach is to **keep the intercept and use k − 1 dummies**. Other exact redundancies among inputs must also be avoided.

### 🟨 Check your understanding

**Question:** A department feature contains CSE, ECE, Mechanical, and Civil. With an intercept, how many dummy variables are needed? If Civil is the reference, how is it represented?

**Answer:** There are four categories, so use **4 − 1 = 3 dummies**: CSE, ECE, and Mechanical. A Civil observation has **0, 0, 0** in these columns.

## <span style="color:#0891b2">🟦 9. Polynomial regression</span>

Sometimes the relationship between an input and a target is curved. A straight line may miss a systematic pattern.

**Polynomial regression includes powers of an input as additional features.**

$$
\hat{y}=b_0+b_1x+b_2x^2+\cdots+b_dx^d
$$

The highest power, $d$, is the **degree** of the polynomial.

| Degree | Model | Typical shape |
|---:|---|---|
| 1 | $b_0+b_1x$ | Straight line |
| 2 | $b_0+b_1x+b_2x^2$ | Quadratic curve |
| 3 | $b_0+b_1x+b_2x^2+b_3x^3$ | Cubic curve, potentially with two turning points |

### 🟨 Example: curved salary growth

Suppose a teaching model predicts monthly salary in **₹ thousands** from experience in years:

$$
\widehat{\text{Salary}}=20+3x+0.5x^2
$$

At four years of experience:

$$
\widehat{\text{Salary}}=20+3(4)+0.5(4^2)
=20+12+8=40
$$

The prediction is **₹40,000 per month**.

| Experience $x$ | Squared experience $x^2$ | Predicted salary (₹ thousands) |
|---:|---:|---:|
| 0 | 0 | 20.0 |
| 1 | 1 | 23.5 |
| 2 | 4 | 28.0 |
| 3 | 9 | 33.5 |
| 4 | 16 | 40.0 |

The increases are 3.5, 4.5, 5.5, and 6.5 thousand rupees. Because the increase changes with $x$, the graph curves.

### Why is polynomial regression still called linear?

**“Linear” refers to the unknown coefficients, not necessarily the shape of the graph in the original input.**

In $b_0+b_1x+b_2x^2$, each coefficient enters as a multiplier of a known feature. The coefficients are not squared, multiplied together, or placed inside nonlinear functions.

Define $z_1=x$ and $z_2=x^2$. Then:

$$
\hat{y}=b_0+b_1z_1+b_2z_2
$$

This is a linear regression using transformed input features.

| Expression | Linear in its coefficients? | Reason |
|---|---|---|
| $b_0+b_1x$ | Yes | Coefficients enter additively as multipliers |
| $b_0+b_1x+b_2x^2$ | Yes | Powers apply to the input, not the coefficients |
| $b_0+b_1\log x$, for $x>0$ | Yes | The input transformation is known |
| $b_0+\exp(b_1x)$ | No | The unknown coefficient is inside an exponential |

> 🟥 **Watch out:** In a quadratic model, $b_1$ alone is not the overall change per unit of $x$. The slope varies with $x$. For students familiar with differentiation, it is $b_1+2b_2x$.

### Choosing the polynomial degree

- **Underfitting:** the model is too simple to capture the pattern, such as a straight line for strongly curved data.
- **Overfitting:** the model follows training noise and predicts new observations poorly.
- **Degree selection:** compare performance on validation data or through cross-validation. Keep the final test set for the final evaluation.

A higher degree can reduce training error without improving predictions on unseen data. It may also produce extreme values outside the observed input range.

## <span style="color:#15803d">🟩 10. Evaluating a regression model</span>

This section extends the slides to explain how to judge predictions. Use **held-out data** when evaluating generalization.

| Metric | Formula | Meaning |
|---|---|---|
| MAE | $\frac{1}{n}\sum\lvert y_i-\hat{y}_i\rvert$ | Average absolute prediction error |
| MSE | $\frac{1}{n}\sum(y_i-\hat{y}_i)^2$ | Average squared prediction error |
| RMSE | $\sqrt{\text{MSE}}$ | Error measure in the target's original units |
| $R^2$ | $1-\frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}$ | Performance relative to predicting the mean of the evaluated targets |

**Lower MAE, MSE, and RMSE are better when comparing models on the same target and evaluation data.** MAE and RMSE use the target's units. MSE uses squared units.

### 🟨 Mini-example

Actual values are **10, 20, 30** and predictions are **12, 18, 33**.

- Residuals: **−2, +2, −3**.
- MAE: $(2+2+3)/3=7/3\approx2.33$.
- MSE: $(4+4+9)/3=17/3\approx5.67$.
- RMSE: $\sqrt{17/3}\approx2.38$.
- Actual mean: 20. Total squared deviation from the mean: $100+0+100=200$.
- $R^2=1-17/200=0.915$.

> 🟥 **Watch out:** $R^2=0.915$ does not mean “91.5% prediction accuracy.” It describes squared-error improvement relative to the mean baseline. $R^2$ can be negative for poor predictions, particularly on test data. Its usual formula is undefined when all evaluated target values are identical.

## <span style="color:#15803d">🟩 11. A sensible modelling workflow</span>

1. **Define the prediction task.** Identify the target, input features, and units.
2. **Inspect the data.** Look for missing values, incorrect records, unusual observations, and relationships.
3. **Split the data.** Separate training data from evaluation data before fitting preprocessing steps. For time-ordered data, preserve time order where appropriate.
4. **Prepare inputs.** Learn imputations and encoding rules from the training data. Apply those same rules to validation and test data.
5. **Fit a baseline model.** Begin with a suitable simple or multiple linear regression.
6. **Check performance and residuals.** Curved residual patterns suggest the model has missed a systematic relationship.
7. **Select improvements using validation data.** For example, compare polynomial degrees without repeatedly using the final test set.
8. **Evaluate the selected model on the test data.** Report error metrics and practical limitations.

> 🟥 **Data leakage:** Information unavailable at prediction time must not influence model training. For example, do not include the final total marks as an input when predicting those same final marks.

### Assumptions and practical checks

| Check | Why it matters |
|---|---|
| Appropriate mean relationship | The chosen features should describe how the average target varies. Polynomial features can represent curvature. |
| Errors have conditional mean zero | For unbiased OLS coefficient estimates, errors should not systematically depend on the predictors. Omitted relevant factors can violate this. |
| No perfect multicollinearity | Exact redundant columns prevent unique identification of all coefficients. |
| Dependence between observations | Repeated measurements or time dependence need suitable splitting and statistical treatment. |
| Error variance | Constant error variance supports conventional standard errors; changing variance may require robust inference. |
| Outliers and influential observations | A few unusual observations can strongly affect squared-error fitting. Investigate before removing them. |

**Normality clarification:** Normally distributed errors support conventional exact small-sample confidence intervals and hypothesis tests. Normality is not required merely to calculate an OLS fit or make a point prediction.

**Extrapolation caution:** A model trained on experience from 0 to 10 years has limited evidence for predictions at 30 years. Predictions outside the observed range are extrapolations and can be unreliable, especially for high-degree polynomials.

## <span style="color:#15803d">🟩 12. Revision sheet</span>

| Topic | Key idea | Formula or rule |
|---|---|---|
| Simple regression | One input, straight-line fit | $\hat{y}=b_0+b_1x$ |
| Intercept | Predicted target at zero input | $b_0$ |
| Slope | Predicted change per input unit | $b_1$ in simple linear regression |
| Residual | Actual minus predicted | $e=y-\hat{y}$ |
| OLS | Minimize squared residuals | $\min\sum(y_i-\hat{y}_i)^2$ |
| Multiple regression | Several input features | $\hat{y}=b_0+\sum_{j=1}^{p}b_jx_j$ |
| Dummy variable | Indicates category membership | 0 or 1 |
| Dummy trap | Intercept plus all category indicators creates dependence | Use a reference category in the standard setup |
| Polynomial regression | Include powers of the input | $\hat{y}=b_0+b_1x+\cdots+b_dx^d$ |
| Generalization | Predict new observations well | Evaluate on held-out data |

### Common mistakes to avoid

| Mistake | Correct understanding |
|---|---|
| The best line must pass through every point | OLS minimizes total squared residuals |
| A negative residual is an invalid result | It means the prediction exceeds the actual value |
| The intercept always has a meaningful real-world interpretation | Zero input may be outside the useful data range |
| The largest raw coefficient identifies the most important input | Coefficients depend on input units and relationships among features |
| More features or a higher degree always improve the model | Training fit may improve while test performance worsens |
| Polynomial regression is nonlinear in its coefficients | It is linear in coefficients and can be curved in the original input |
| A regression coefficient proves causation | Causal conclusions require suitable study design and assumptions |

## <span style="color:#d97706">🟨 13. Practice questions</span>

Try these before reading the solutions.

1. A model is $\hat{y}=12+4x$. Identify the intercept and slope, and predict $y$ when $x=5$.
2. An actual value is 60 and its prediction is 66. Find the residual and squared residual. Is this underprediction or overprediction?
3. Fit a simple OLS line to $(1,3)$, $(2,5)$, and $(3,7)$. Predict the target at $x=4$.
4. A profit model is $\hat{y}=5+0.6R+0.4M$, with all amounts in ₹ lakh. Predict profit for $R=10$ and $M=5$. Interpret the R&D coefficient.
5. A department variable contains CSE, ECE, Mechanical, and Civil. With an intercept and Civil as the reference category, construct the dummy encoding table.
6. A fitted model is $\hat{y}=15+2x+4D$, where $D=1$ for Group A and $D=0$ for Group B. Predict both groups at $x=3$ and interpret the dummy coefficient.
7. For $\hat{y}=2+3x+x^2$, predict $y$ at $x=3$. Explain why this is linear in its coefficients.
8. Model A has training RMSE 1.2 and validation RMSE 1.4. Model B has training RMSE 0.1 and validation RMSE 4.8. Which would you select, assuming the same data and target units? Explain.

## <span style="color:#15803d">🟩 14. Worked solutions</span>

### Solution 1: interpreting a line

Intercept = **12**, slope = **4**.

$$
\hat{y}=12+4(5)=\boxed{32}
$$

Each additional unit of $x$ adds four units to the model's prediction.

### Solution 2: residual

$$
e=60-66=\boxed{-6},\qquad e^2=\boxed{36}
$$

This is **overprediction** because the prediction exceeds the actual value.

### Solution 3: fitting a line

The means are $\bar{x}=2$ and $\bar{y}=5$.

$$
b_1=\frac{(-1)(-2)+(0)(0)+(1)(2)}{(-1)^2+0^2+1^2}
=\frac{4}{2}=2
$$

$$
b_0=5-2(2)=1
$$

The fitted model is $\boxed{\hat{y}=1+2x}$. At $x=4$, the prediction is $\boxed{9}$. All three training residuals are zero because these points lie exactly on a line. The prediction at 4 is outside the training range, so it is an extrapolation.

### Solution 4: multiple regression

$$
\hat{y}=5+0.6(10)+0.4(5)=5+6+2=\boxed{₹13\text{ lakh}}
$$

Holding marketing spending constant, ₹1 lakh more R&D spending is associated with ₹0.6 lakh more predicted profit in this model.

### Solution 5: categorical encoding

Use **three** dummy variables because $k-1=4-1=3$.

| Department | $D_{\text{CSE}}$ | $D_{\text{ECE}}$ | $D_{\text{Mechanical}}$ |
|---|---:|---:|---:|
| Civil | 0 | 0 | 0 |
| CSE | 1 | 0 | 0 |
| ECE | 0 | 1 | 0 |
| Mechanical | 0 | 0 | 1 |

Civil is the reference category. Each dummy coefficient compares its department with Civil while holding other included features constant.

### Solution 6: dummy coefficient

- Group A: $15+2(3)+4(1)=\boxed{25}$.
- Group B: $15+2(3)+4(0)=\boxed{21}$.

At the same $x$, the model predicts Group A to be **four target units higher** than Group B.

### Solution 7: polynomial regression

$$
\hat{y}=2+3(3)+3^2=2+9+9=\boxed{20}
$$

The coefficients multiply the known features $1$, $x$, and $x^2$ additively. The squared term applies to the input, so the model remains linear in its coefficients.

### Solution 8: model selection

Select **Model A** based on its lower validation RMSE, **1.4 versus 4.8**. Model B's much lower training error and higher validation error suggest overfitting. Evaluate the selected model on a separate test set for the final performance estimate.

---

> 🟩 **Core idea:** Regression learns coefficients that connect input features to a numerical target. Choose a suitable form, understand what each coefficient means, and judge the model using predictions on data it did not train on.

**Source alignment:** Slides 1–10 introduce simple regression and OLS. Slides 11–22 cover multiple regression, dummy variables, and the dummy variable trap. Slides 23–29 introduce polynomial regression and linearity in coefficients. Evaluation, practical checks, and exercises are supplementary teaching material.

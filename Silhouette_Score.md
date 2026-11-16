# 🌈 Silhouette Score in K-Means Clustering

## 🎯 1. Objective

To evaluate the quality of K-Means customer segmentation using the **Silhouette Score** and identify a suitable number of clusters (`k`) for the Mall Customers dataset.

> 🟦 **Dataset:** Mall Customers · **Customers:** 200 · **Best k:** 5 · **Top score:** 0.5539

In this example, customers are grouped using two features:
- **Annual Income**
- **Spending Score**

## 🌈 2. What is the Silhouette Score?

The **Silhouette Score** measures how well a data point fits within its assigned cluster compared with other clusters. It evaluates:

1. **Cohesion:** How close a customer is to other customers in the **same** cluster.
2. **Separation:** How far a customer is from customers in the **nearest different** cluster.

For customer `i`, the silhouette coefficient is:

$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$

Here:
- **a(i):** Average distance between customer `i` and other customers in the same cluster.
- **b(i):** Smallest average distance between customer `i` and the members of any other cluster.

The overall Silhouette Score is the mean of the silhouette coefficients of all customers.

### 🔵 Understanding a(i) and 🟠 b(i) with an example

🔵 **a(i) — Average distance within the same cluster (cohesion)**

For a customer `i`, `a(i)` is the average distance from that customer to **all other customers in their own cluster** (excluding the customer itself).

For example, Customer A belongs to Cluster 1. The distances to three other customers in Cluster 1 are **2, 3, and 4**:

$$
a(i) = \frac{2 + 3 + 4}{3} = 3
$$

A **smaller a(i)** means the customer is closer to other customers in its assigned cluster.

🟠 **b(i) — Smallest average distance to another cluster (separation)**

For the same customer, calculate the **average** distance to every customer in each *other* cluster. Then choose the smallest of those cluster averages. This is `b(i)`; it is **not** simply the distance to the single closest customer or centroid.

| Alternative cluster | Average distance from Customer A |
|---|---:|
| Cluster 2 | **8** |
| Cluster 3 | 12 |
| Cluster 4 | 10 |
| Cluster 5 | 15 |

Therefore, `b(i) = 8` because Cluster 2 is the nearest **alternative cluster by average distance**. A **larger b(i)** generally indicates better separation.

🟣 **Calculate the individual Silhouette Score:**

$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
     = \frac{8 - 3}{\max(3,8)}
     = \frac{5}{8} = 0.625
$$

This means Customer A is comparatively close to its own cluster and farther from the nearest alternative cluster.

| Term | Meaning | Desired behavior |
|---|---|---|
| `a(i)` | Average distance to other points **inside the same cluster** | Smaller is better |
| `b(i)` | **Smallest** average distance to points in **another cluster** | Larger is better |

💡 **Memory trick:** `a(i)` = *Apna cluster* (own cluster); `b(i)` = *Bahar wala nearest cluster* (nearest other cluster).

### 🚦 How to interpret the score

| Score | Interpretation |
|---|---|
| 🟢 Near **+1** | Very well-separated clusters |
| 🔵 **0.5 to 1** | Generally good separation |
| 🟡 Near **0** | Clusters overlap or points lie near a cluster boundary |
| 🔴 Below **0** | Some customers may fit better in another cluster |

> These descriptions are **rules of thumb**, not strict quality thresholds. A value close to +1 indicates that a point fits its current cluster well; near 0 indicates weak separation; and a negative value suggests it is closer, on average, to another cluster. The overall Silhouette Score is the **average of the individual scores** across all customers.

For your 200-customer dataset, the overall Silhouette Score is **0.5539** when `k = 5`. This is generally a reasonably good result, and it is the highest score **among the tested values of k = 2 through 10**.

## 🔍 3. Why calculate it for multiple values of k?

K-Means requires us to specify the number of clusters in advance. Testing `k = 2` to `10` allows us to compare clustering quality and select the value with the highest average Silhouette Score **among those tested**.

We start at `k = 2` because the Silhouette Score cannot be calculated for a single cluster.

## 💻 4. Python code with comments

> **Prerequisite:** The notebook has already loaded the Mall Customers data and created `X` containing the Annual Income and Spending Score columns.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Create an empty list to store the score for each cluster count.
silhouette_scores = []

# Test k = 2, 3, ..., 10 (11 is excluded by range).
k_values = range(2, 11)

for k in k_values:
    # Initialize K-Means for the current number of clusters.
    candidate_model = KMeans(
        n_clusters=k,
        init='k-means++',
        random_state=42,
        n_init=10
    )

    # Train the model and assign a cluster label to every customer.
    cluster_labels = candidate_model.fit_predict(X)

    # Compare how close customers are to their own and other clusters.
    score = silhouette_score(X, cluster_labels)

    # Save the score and print it for easy comparison.
    silhouette_scores.append(score)
    print(f'Number of clusters: {k}, Silhouette Score: {score:.4f}')

# np.argmax returns the index of the largest score in the list.
# Convert k_values to a list to retrieve the matching number of clusters.
best_k = list(k_values)[np.argmax(silhouette_scores)]

print(f'\nBest k by Silhouette Score (among 2 to 10): {best_k}')
print(f'Highest Silhouette Score: {max(silhouette_scores):.4f}')

# Plot how the average Silhouette Score changes as k changes.
plt.plot(list(k_values), silhouette_scores, marker='o')
plt.axvline(best_k, linestyle='--', color='gray',
            label=f'Best silhouette k = {best_k}')
plt.title('Silhouette Score vs. Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Average Silhouette Score')
plt.xticks(list(k_values))
plt.legend()
plt.show()
```

## 📊 5. Actual output from your Colab run

The following results are from the output shown in your notebook screenshot:

| Number of clusters (k) | Silhouette Score |
|---:|---:|
| 2 | 0.2969 |
| 3 | 0.4676 |
| 4 | 0.4932 |
| 🏆 **5** | **0.5539** |
| 6 | 0.5398 |
| 7 | 0.5288 |
| 8 | 0.4548 |
| 9 | 0.4561 |
| 10 | 0.4411 |

```text
Best k by Silhouette Score (among 2 to 10): 5
Highest Silhouette Score: 0.5539
```

### ✅ Result interpretation

- The Silhouette Score rises from **0.2969** at `k = 2` to **0.5539** at `k = 5`.
- The **highest score is 0.5539 at `k = 5`**, so five clusters are preferred among the values tested.
- Scores at `k = 6` (**0.5398**) and `k = 7` (**0.5288**) are slightly lower; creating extra groups did not improve this particular evaluation metric.
- For `k = 8` to `10`, scores drop further.
- A score of **0.5539** suggests reasonably distinct customer groups, though some overlap may remain.

## 🧪 6. Evaluate the final five-cluster model

```python
# Create the final model using the best cluster count from the comparison.
kmeans = KMeans(n_clusters=5, init='k-means++',
                random_state=42, n_init=10)

# Assign each customer to one of five clusters.
y_kmeans = kmeans.fit_predict(X)

# Evaluate the clustering arrangement.
final_silhouette = silhouette_score(X, y_kmeans)

# Count how many customers were assigned to each cluster.
print(f'Silhouette Score for the 5-cluster model: {final_silhouette:.4f}')
print(f'Cluster sizes: {np.bincount(y_kmeans)}')
```

Your output:

```text
Silhouette Score for the 5-cluster model: 0.5539
Cluster sizes: [81 39 22 35 23]
```

This means **200 customers** were assigned to five groups:

| Cluster label | Number of customers | Percentage |
|---|---:|---:|
| 0 | 81 | 40.5% |
| 1 | 39 | 19.5% |
| 2 | 22 | 11.0% |
| 3 | 35 | 17.5% |
| 4 | 23 | 11.5% |
| **Total** | **200** | **100%** |

**Note:** Cluster numbers (`0`, `1`, etc.) are arbitrary identifiers. They do not imply any ranking or business meaning. To describe clusters as “high income/high spending,” for example, inspect the cluster centroids or summary statistics first.

## ⚖️ 7. Elbow Method vs. Silhouette Score

| Elbow Method | Silhouette Score |
|---|---|
| Uses **WCSS / inertia**. | Uses **within-cluster cohesion** and **between-cluster separation**. |
| Select a bend (“elbow”) in the WCSS curve. | Prefer a higher average score among candidate `k` values. |
| WCSS naturally decreases as `k` increases. | The Silhouette Score may increase or decrease as `k` changes. |
| Helps choose a compact clustering arrangement. | Helps assess how distinct the clusters are. |

Both methods are useful. If they suggest different values, examine the cluster plots, stability and practical interpretation before choosing.

## 🛍️ 8. Simple real-world example

Imagine that a shopping mall wants to understand how customers differ by income and spending behavior. K-Means can group customers with similar data values. The Silhouette Score then checks whether customers in each group tend to be closer to one another than to customers in other groups.

With your data, **five clusters** give the strongest average separation score among the nine tested options. This provides evidence for using five groups in the customer segmentation exercise.

## 🏁 9. Conclusion for the practical file

**Result:** K-Means clustering was applied to the Mall Customers dataset using Annual Income and Spending Score. The Silhouette Score was calculated for cluster counts from 2 to 10. The highest score, **0.5539**, occurred at **k = 5**. Therefore, **five clusters** were selected as the preferred number among those tested. The final five-cluster model grouped the **200 customers** into groups of **81, 39, 22, 35, and 23** customers.

⭐ **Key takeaway:** The Silhouette Score helps us judge clustering quality and compare different numbers of clusters—not just whether K-Means ran successfully.

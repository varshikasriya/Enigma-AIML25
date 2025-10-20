# Level-2 Model Performance Analysis: Linear Regression (SGD) vs. Perceptron

## Summary of Results

| Model | Dataset | Accuracy | Time to Convergence (s) | Iterations |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression (SGD)** | Linear | 0.8800 | 0.0515 | 300 |
| **Perceptron** | Linear | 0.8433 | 1.0757 | 1000 |
| **Linear Regression (SGD)** | Non-Linear | 0.7900 | 0.0780 | 450 |
| **Perceptron** | Non-Linear | 0.7963 | 1.7120 | 1000 |

*(Note: Actual convergence times for SGD will vary based on hardware and dataset size, but the trend of lower time-per-epoch compared to the full Perceptron scan holds true for large datasets.)*

## Theoretical Analysis

### 1. Performance on Linearly Separable Data (`binary_classification.csv`)

| Model | Result | Rationale |
| :--- | :--- | :--- |
| **Perceptron** | **Lower Accuracy** (0.8433) | The Perceptron is an **online, greedy algorithm**. In the provided sample, it failed to find a perfect boundary (iterations reached 1000 with errors remaining) likely due to the initial learning rate and data order preventing true convergence to $100\%$. It only stops when *all* points are classified correctly. |
| **LR (SGD)** | **Higher Accuracy** (0.8800) | Linear Regression (even with SGD) seeks to **minimize the overall Mean Squared Error (MSE)**. Its continuous, probabilistic nature allows it to fit a more stable decision boundary by factoring in the "confidence" of its prediction (closeness to 0 or 1), leading to a better generalized fit than the rigid Perceptron rule. |
| **Convergence Time** | **LR (SGD) is faster** | The Perceptron's time is high because it executes a full loop through all data points for *every iteration* and is forced to run for the full 1000 epochs due to misclassification errors. LR (SGD) uses an efficient vector calculation for its update check, allowing it to converge on a stable *minimum loss* state much faster than the Perceptron finds a perfect *error-free* classification. |

### 2. Performance on Non-Linearly Separable Data (`binary_classification_non_lin.csv`)

Both models exhibit nearly identical, low accuracy (around 79%).

* **Equal Struggle:** Since both Linear Regression and the Perceptron are fundamentally **linear classifiers**, they are unable to create the curved or complex boundaries required to separate non-linear data. They both find the single best straight line, which yields poor performance.
* **Perceptron's Slight Edge:** The Perceptron's marginally higher accuracy (0.7963 vs 0.7900) can be attributed to its **error-driven weight updates**. Since it only updates when a misclassification occurs, it may create a slightly more adaptive boundary for the highly irregular, non-linear pattern, though the effect is minimal.

### Conclusion

The experiment demonstrates that while the **Perceptron** is theoretically guaranteed to find a perfect boundary on linearly separable data, its training is less stable and much slower than **Linear Regression (SGD)**. For real-world problems where data is often noisy, the MSE-minimizing approach of **Linear Regression** often provides a more robust and faster solution, even for binary classification. Neither model is suitable for non-linear data.
# Level 2 – Models from Scratch

This project implements **Linear Regression (Gradient Descent)** and a **Perceptron** from scratch using only Python.  
It compares their performance on a simple binary classification dataset.

---
# Linear Regression (Gradient Descent)

Goal: Fit a straight line (or plane) that best predicts output values.

Process:
- Start with random weights.
- Predict using a linear combination of features → y_pred = w·x + b.
- Compute error = (predicted − actual).
- Update all weights using gradient descent to minimize total squared error:

w = w - learning_rate * gradient
- Repeat until the loss (error) stops improving.

Output: Real numbers → turned into 0/1 by thresholding at 0.5 for classification.
Key trait: Always converges (because loss is convex).

---
# Perceptron

Goal: Find a line (or hyperplane) that separates two classes.

Process:
1. Start with random weights.
2. For each training example:
- Predict class using sign(w·x + b).
- If prediction is wrong → update weights in direction of the true label:
w = w + y_true * x
3. Repeat for many epochs.

Output: 0 or 1 directly (based on sign of w·x + b).
Key trait: Only converges if the data is linearly separable — otherwise it keeps updating forever.

---

## Datasets
| Dataset | Description | Separability | File |
|----------|--------------|--------------|------|
| **binary_classification.csv** | Synthetic data generated with `make_classification()`. | Linear | `/content/binary_classification.csv` |
| **binary_classification_non_lin.csv** | Two-moons shaped data generated with `make_moons()`. | Nonlinear | `/content/binary_classification_non_lin.csv` |

---

## Models Implemented
| Model | Key Idea | Convergence Behavior | Expected Strength |
|:-------|:-----------|:---------------------|:------------------|
| **LinearRegressionGD** | Minimizes MSE using batch gradient descent with a 0.5 threshold for classification. | Fast, stable (convex loss). | Works well on linearly separable data. |
| **Perceptron** | Online mistake-driven updates for binary labels (+1/-1). | Converges only for separable data. | Simple, interpretable baseline. |

---
## Results Snapshot
| Dataset Type | Linear Regression Accuracy | Perceptron Accuracy | Key Observation |
|:--------------|:----------------------------|:--------------------|:----------------|
| **Linear** | ~0.99 | ~0.99 | Both models achieve near-perfect classification. |
| **Nonlinear** | ↓ 0.6–0.7 | ↓ 0.6–0.7 | Linear decision boundaries underfit curved structures. |


---

## Interpretation
- Linear regression is **fast and consistent** but assumes linear separability.  
- The perceptron is **sensitive to non-separable data** and may not converge.  
- The accuracy drop on nonlinear data demonstrates **model bias** rather than optimization failure.

---

## Limitations
- Deterministic split (no shuffle/stratification).  
- Binary datasets only.  
- Linear decision boundary only.  
- No confusion matrices or ROC/PR curves.

---

## Changes made after feedback:
1. Objective of the Change
 
Originally, the code evaluated Linear Regression (GD) and Perceptron on a single dataset (binary_classification.csv).
The goal of the modification was to extend the same experiment to include a second dataset (binary_classification_non_lin.csv) representing a nonlinearly separable problem.
This enables comparative analysis of model performance under two fundamentally different conditions.
| Area                   | Single-Dataset Code                                       | Dual-Dataset Code                                                                                                                          | Purpose                                                    |
| ---------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------- |
| **Execution Flow**     | Ran `run_one()` once                                      | Added a wrapper `run_and_save_for_dataset()` that runs `run_one()` twice — once for each dataset                                           | Automates multi-dataset runs                               |
| **Data Inputs**        | Only `/content/binary_classification.csv`                 | Added `/content/binary_classification_non_lin.csv`                                                                                         | Enables contrast between linear and nonlinear separability |
| **Output Structure**   | Single `metrics.csv` and `metrics.json`                   | Per-dataset output folders:<br>• `results_linear/metrics.csv`<br>• `results_nonlinear/metrics.csv`<br>Plus a merged `metrics_combined.csv` | Organized and reusable outputs                             |
| **Result Aggregation** | One dataset summary                                       | Combined DataFrame from both runs with new column `dataset_type` (`linear` / `nonlinear`)                                                  | Facilitates grouped comparison                             |
| **Plots**              | Individual model metrics per dataset                      | Added grouped bar charts comparing **accuracy**, **training time**, and **prediction time** across both datasets                           | Clear visualization of performance drop                    |
| **Helper Namespacing** | Shared `_dot` and `_add_bias` (risk of function override) | Renamed to `_dot_lr`, `_add_bias_lr`, `_dot_pp`, `_add_bias_pp`                                                                            | Prevents cross-module shadowing                            |
| **Schema Consistency** | Missing values handled implicitly                         | Added `None` defaults for missing fields (`final_loss`, `mistakes_last_epoch`)                                                             | Stable CSV/JSON schema for combined logging                |
| **Dataset Management** | Assumed CSV pre-exists                                    | Added dataset generation cells (for `make_classification` and `make_moons`)                                                                | Enables reproducibility in Colab                           |

2. Impact of the Changes
- Functional Impact

The pipeline can now evaluate both linear and nonlinear separability without manual reruns or path edits.

Produces a comprehensive combined report (metrics_combined.csv) with consistent columns for both models and datasets.

Enables side-by-side visualization of model generalization gaps.

- Analytical Impact

Demonstrates the bias–variance tradeoff concretely:

Linear models → excellent on linearly separable data (≈99% accuracy).

Poor performance on nonlinear data (≈60–70% accuracy).

Highlights that underperformance is due to model bias, not code or optimization error.

- Conceptual Impact

Transforms a one-off experiment into a controlled comparative study.

Makes the codebase extendable — new datasets or models can be added in a few lines.

Lays groundwork for more advanced comparisons (e.g., logistic regression, polynomial features, kernels).

# Overall Outcome

The refactoring elevated the experiment from a basic implementation test to a comparative analysis framework.
It now quantifies how model linearity limits performance — turning a simple code demo into a meaningful ML bias experiment.

---

# Summary

Linear Regression (GD): Converges quickly and gives high accuracy.

Perceptron: Reaches similar accuracy but takes longer and may not fully converge.

Both models show strong performance on a linearly separable dataset.

# Analysis – Level 2: Models from Scratch

**Dataset:** `datasets/binary_classification.csv`  
**Models:** Linear Regression (Gradient Descent) and Perceptron  
**Utilities:** `load_csv_xy`, `train_test_split`, `standardize_fit/apply`, `accuracy`, `avg_predict_time_us`  

---

## Experimental Setup

- **Train/Test Split:** 80/20 (deterministic order)
- **Standardization:** z-score normalization applied on training mean/std
- **Hardware:** Google Colab CPU runtime
- **Stopping Criteria:**
  - Linear Regression: relative loss improvement < 1e-6
  - Perceptron: maximum 2000 epochs or zero mistakes in an epoch

---

## Results

| Dataset | Model | Accuracy | Train Time (s) | Time / Prediction (µs) | Iterations | Converged |
|:---------|:--------|:-----------:|:--------------:|:---------------------:|:-----------:|:-----------:|
| binary_classification | Linear Regression | **0.990** | 0.0221 | 0.46 | 39 | true |
| binary_classification | Perceptron | **0.990** | 0.9772 | 0.45 | 2000 | false |

---

## Observations & Interpretation

### Linear Regression (Gradient Descent)
- **Convergence:** Reached a stable minimum in just 39 iterations due to smooth convex loss and well-scaled inputs.
- **Performance:** Fastest training time and consistent convergence.
- **Behavior:** Despite being a regression model, thresholding at 0.5 produced near-perfect classification — suggesting an easily separable dataset.

### Perceptron
- **Convergence:** Hit the iteration limit (2000) without full convergence — indicating either noisy or non-linearly separable data.
- **Accuracy:** Still achieved 0.99 accuracy, showing that even partial convergence produced an effective decision boundary.
- **Training Dynamics:** Mistake-driven updates are slower than batch gradient descent, which explains the longer training time.

---

## Limitations

- **No Shuffling:** Current train/test split is deterministic — if the CSV rows are ordered, the reported accuracy may be over-optimistic.
- **Uncalibrated Probabilities:** Linear Regression outputs raw values thresholded at 0.5; logistic regression would be a better baseline for true classification tasks.
- **No Analysis of Misclassifications:** Confusion matrices and learning curves would reveal where errors persist.

---

## Conclusion

Both Linear Regression and the Perceptron achieve high performance on this dataset, but for different reasons:
- Linear Regression converges quickly and stably due to convex optimization.
- The Perceptron’s long training and lack of convergence highlight its sensitivity to data separability.


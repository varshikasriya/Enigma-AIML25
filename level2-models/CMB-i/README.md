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

# Output:

| Dataset | Model | Accuracy | Train Time (s) | Time / Prediction (µs) | Iterations | Converged |
|:---------|:--------|:-----------:|:--------------:|:---------------------:|:-----------:|:-----------:|
| binary_classification | Linear Regression | **0.990** | 0.0221 | 0.46 | 39 | True |
| binary_classification | Perceptron | **0.990** | 0.9772 | 0.45 | 2000 | False |

- accuracy:

<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/3306586c-d8cd-4690-9e39-fcdf8a1854a9" />

- training time:

  <img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/c7cdc670-2ee2-4f75-8845-87cbe2a01340" />

- confusion matrix:

  <img width="507" height="455" alt="image" src="https://github.com/user-attachments/assets/a4b22eaf-7bf1-4c4c-ae39-bd90c7e0672a" />

---

# Summary

Linear Regression (GD): Converges quickly and gives high accuracy.

Perceptron: Reaches similar accuracy but takes longer and may not fully converge.

Both models show strong performance on a linearly separable dataset.

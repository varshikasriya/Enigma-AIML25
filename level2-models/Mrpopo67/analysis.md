# 📊 Model Performance Analysis

---

## 🔬 Datasets Overview

| Dataset      | Assumed Nature              |
|--------------|------------------------------|
| Dataset A    | Likely **linearly separable** |
| Dataset B    | Likely **non-linearly separable** |

---

## 📈 Results Summary

| Model             | Dataset     | Accuracy | Time to Converge (s) | Time per Prediction (s) |
|------------------|-------------|----------|-----------------------|--------------------------|
| Linear Regression| Dataset A   | 0.79125  | 0.00899               | 0.0                      |
| Linear Regression| Dataset B   | 0.87167  | 0.01099               | 0.0                      |
| Perceptron       | Dataset A   | 0.79625  | 2.42139               | 0.0                      |
| Perceptron       | Dataset B   | 0.84333  | 1.68010               | 0.0                      |

---

## 🤖 Observations & Insights

### 1. **Dataset A**

- The **Perceptron** performed slightly better than Linear Regression on accuracy (**0.79625 vs 0.79125**), although it required significantly more training time (**~2.42s vs ~0.009s**).
- This dataset appears to be **linearly separable**, as both models reached reasonably high accuracy.
- The faster convergence of Linear Regression is expected due to its use of **gradient descent with continuous loss**, while the Perceptron uses discrete updates which can take longer to converge even on simple data.

### 2. **Dataset B**

- Surprisingly, **Linear Regression** outperformed the Perceptron on this dataset (**0.87167 vs 0.84333**), despite the data likely being **non-linearly separable**.
- The Perceptron still performed decently, but its convergence time (**1.68s**) was again much higher than Linear Regression (**0.011s**).
- This may be due to the regression model being able to **approximate non-linearity** better through the continuous outputs and thresholding.

---

## 📌 Key Takeaways

- **Linear Regression** showed competitive performance on both datasets and **converged quickly**, making it a solid baseline model.
- **Perceptron** slightly edged out Linear Regression on Dataset A (linearly separable), as expected.
- On Dataset B (likely non-linear), Linear Regression surprisingly achieved **higher accuracy**, possibly due to its **generalization from continuous space**, while the Perceptron’s discrete nature limited its ability to model non-linear boundaries.
- **Training time difference** between the models is stark. The Perceptron took **200–300x more time to converge**, which becomes important for large datasets.

---

## 🧠 Reflections

- The results show that **model choice heavily depends on data characteristics** — especially linearity.
- **Perceptron** is a powerful model for strictly linearly separable problems but struggles with anything more complex.
- **Linear Regression**, though not meant for classification, can still be surprisingly effective when combined with rounding — but lacks probabilistic interpretation.
- This experiment reinforces the need for **flexible models** (e.g., Logistic Regression, SVMs, or Neural Networks) for more complex real-world data.

---

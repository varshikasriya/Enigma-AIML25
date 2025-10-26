# Level 2 Analysis - varshikasriya

## Dataset 1: binary_classification.csv
--------------------------------
Linear Regression Accuracy: 0.925
Linear Regression Train Time: 0.0138s
Linear Regression Prediction Time: 0.00000118s
Perceptron Accuracy: 0.867
Perceptron Train Time: 4.4482s
Perceptron Prediction Time: 0.00000019s

Observations:
- Linear Regression outperformed Perceptron on this linearly separable dataset (92.5% vs 86.67%).
- Linear Regression treats output as continuous, then thresholded at 0.5 for classification.
- Perceptron updates weights only for misclassified samples; better for discrete classes.
- Linear Regression converged significantly faster (0.014s vs 4.45s) due to smooth gradient descent updates on all samples.
- Perceptron did not converge within 1000 iterations, suggesting the data may have some noise or overlap.
- Training time difference is substantial - Linear Regression is ~322x faster to train.

## Dataset 2: binary_classification_non_lin.csv
--------------------------------
Linear Regression Accuracy: 0.763
Linear Regression Train Time: 0.0238s
Linear Regression Prediction Time: 0.00000006s
Perceptron Accuracy: 0.788
Perceptron Train Time: 6.2877s
Perceptron Prediction Time: 0.00000014s

Observations:
- Perceptron slightly outperformed Linear Regression on this non-linear dataset (78.75% vs 76.25%).
- Both models struggled with non-linear boundaries as they can only create linear decision boundaries.
- Accuracy dropped for both models compared to Dataset 1, confirming the non-linear nature of the data.
- Linear Regression remained much faster to train (0.024s vs 6.29s), maintaining ~264x speed advantage.
- Higher MSE values (0.168 for LR, 0.213 for Perceptron) indicate worse fit due to non-linear patterns.
- Neither model converged optimally, hitting maximum iterations without finding perfect separation.
- Training time depends on number of features, epochs, and data complexity.

## Overall Analysis

### Key Insights:

**Dataset 1 (Linearly Separable):**
- Linear Regression achieved better accuracy (92.5%) because it optimizes a smooth continuous function
- The data appears mostly linearly separable with minimal noise
- Gradient descent efficiently finds the optimal decision boundary
- Perceptron's discrete updates make it more sensitive to data ordering and noise

**Dataset 2 (Non-Linear):**
- Both models performed worse (~76-79% accuracy) due to inherent non-linearity
- Perceptron slightly better (78.75% vs 76.25%) possibly due to its more flexible, instance-based learning
- Linear assumptions break down when true boundary is curved or complex
- Without feature engineering (polynomial features, kernels), linear models cannot capture non-linear patterns

### Performance Trade-offs:

**Training Speed:**
- Linear Regression: 0.014s - 0.024s (extremely fast)
- Perceptron: 4.45s - 6.29s (much slower)
- Linear Regression is 200-300x faster due to batch gradient updates vs iterative single-sample updates

**Prediction Speed:**
- Both models have negligible prediction time (microseconds)
- Linear Regression slightly faster due to simpler computation

**Convergence:**
- Linear Regression converged in ~239-240 iterations with smooth progress
- Perceptron hit max iterations (1000) without converging, indicating:
  - Data not perfectly separable
  - Possible need for lower learning rate
  - Presence of outliers or noise

### Conclusions:

1. **For Linearly Separable Data:** Linear Regression is superior - better accuracy, much faster training
2. **For Non-Linear Data:** Both struggle equally; neither is clearly better without feature engineering
3. **Speed vs Accuracy:** Linear Regression offers best overall trade-off for these datasets
4. **Real-World Application:** For production systems, Linear Regression's 300x speed advantage makes it preferable when accuracy is comparable

### Recommendations:
- For non-linear problems, consider: Neural Networks, SVMs with RBF kernels, Decision Trees, or feature engineering
- If using Perceptron, experiment with lower learning rates for better convergence
- Linear Regression with polynomial features could improve Dataset 2 performance

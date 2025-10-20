# Level 2 Analysis: Linear Regression vs Perceptron

**Dataset:** `C:\Users\gshra\OneDrive\Desktop\Enigma-AIML25\datasets\binary_classification.csv`  
**Shape:** 1000 samples × 2 features  
**Split:** 650 train / 150 val / 200 test  

## Results Summary

### Linear Regression
- **Train Accuracy:** 0.9846
- **Validation Accuracy:** 1.0000
- **Test Accuracy:** 1.0000
- **Test F1 Score:** 1.0000
- **Epochs:** 233 (converged: True)
- **Training Time:** 0.0135s

### Perceptron
- **Train Accuracy:** 0.9846
- **Validation Accuracy:** 1.0000
- **Test Accuracy:** 1.0000
- **Test F1 Score:** 1.0000
- **Epochs:** 1000 (converged: False)
- **Training Time:** 2.1538s

## Detailed Analysis

### Overfitting Check
- **Linear Regression:** Train-Test gap = -0.0154
- **Perceptron:** Train-Test gap = -0.0154

### Convergence Behavior
- **Linear Regression:** Converged after 233 epochs using gradient descent with loss=0.042318
- **Perceptron:** Did not converge after 1000 epochs (final mistakes: 9)

### Model Comparison

**Accuracy:**
- Both models achieve similar test accuracy

**Speed:**
- Linear Regression training: 0.0135s
- Perceptron training: 2.1538s
- Linear Regression trains 159.4x faster

### Key Insights

**Linear Regression:**
- Uses squared loss and gradient descent for optimization
- Feature standardization enables stable training
- L2 regularization helps prevent overfitting
- Produces continuous outputs (probability-like scores)
- More robust to noisy/overlapping data

**Perceptron:**
- Uses online learning with threshold-based updates
- Only updates on misclassified examples
- No built-in regularization (prone to overfitting)
- Produces binary outputs directly
- Guaranteed to converge only if data is linearly separable
- Sensitive to class imbalance and outliers

### Recommendations

Based on the results:
- The data is likely **not perfectly linearly separable**, explaining why perceptron struggles
- Linear Regression's probabilistic approach handles noisy boundaries better
- Both models perform similarly, suggesting the decision boundary is approximately linear
- For production use, consider Linear Regression due to better generalization
- To improve Perceptron, consider using averaged perceptron or kernel methods

# Level 2 Analysis: Linear Regression vs Perceptron

**Dataset:** `datasets/binary_classification.csv`  
**Shape:** 1000 samples × 2 features  
**Split:** 650 train / 150 val / 200 test  

## Results Summary

### Linear Regression
- **Train Accuracy:** 0.8338
- **Validation Accuracy:** 0.8067
- **Test Accuracy:** 0.8600
- **Test F1 Score:** 0.8654
- **Epochs:** 259 (converged: True)
- **Training Time:** 0.0083s

### Perceptron
- **Train Accuracy:** 0.5846
- **Validation Accuracy:** 0.5533
- **Test Accuracy:** 0.5950
- **Test F1 Score:** 0.5970
- **Epochs:** 1000 (converged: False)
- **Training Time:** 1.9524s

## Detailed Analysis

### Overfitting Check
- **Linear Regression:** Train-Test gap = 0.8338 - 0.8600 = -0.0262  
- **Perceptron:** Train-Test gap = 0.5846 - 0.5950 = -0.0104  

*No significant overfitting detected.*

### Convergence Behavior
- **Linear Regression:** Converged after 259 epochs using gradient descent with loss=<final_loss>  
- **Perceptron:** Did not converge after 1000 epochs (final mistakes: <final_mistakes>)

### Model Comparison

**Accuracy:**
- Linear Regression performs better by 0.2650 on test set

**Speed:**
- Linear Regression training: 0.0083s  
- Perceptron training: 1.9524s  
- Linear Regression trains ~235x faster than Perceptron

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
- Both models perform differently; Linear Regression clearly generalizes better
- For production use, consider Linear Regression due to better generalization
- To improve Perceptron, consider using averaged perceptron or kernel methods

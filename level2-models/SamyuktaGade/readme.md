# Level 2: Models From Scratch

This directory contains implementations of basic machine learning algorithms built from scratch using only NumPy.

## 📁 Project Structure

```

level2-models/SamyuktaGade/
├── linear_regression.py    # Linear Regression with Gradient Descent
├── perceptron.py           # Perceptron classifier
├── train_compare.py        # Training and comparison script
├── results.json            # Numerical results
├── analysis.md             # Detailed analysis report
└── README.md               # This file

````

## 🎯 Implementations

### 1. Linear Regression (Gradient Descent)

**File:** `linear_regression.py`

Linear Regression implemented from scratch using gradient descent optimization for binary classification.

**Features:**
- Batch gradient descent
- Feature standardization
- L2 regularization
- Optional bias/intercept term
- Validation loss tracking
- Convergence detection
- Continuous and binary predictions

**Key Parameters:**
```python
LinearRegressionGD(
    lr=0.01,              
    max_epochs=5000,      
    tol=1e-6,             
    l2=0.01,              
    fit_intercept=True,   
    standardize=True,     
    random_state=42       
)
````

**Advantages:**

* Handles noisy/overlapping data
* Produces probability-like outputs
* Regularization prevents overfitting

**Limitations:**

* Assumes linear decision boundary
* Sensitive to learning rate
* Requires feature scaling

### 2. Perceptron

**File:** `perceptron.py`

Classic binary perceptron using online learning.

**Features:**

* Online learning (per-sample updates)
* Shuffling each epoch
* Early stopping with patience
* Mistake tracking
* Decision function and margin computation

**Key Parameters:**

```python
Perceptron(
    lr=0.1,
    max_epochs=1000,
    fit_intercept=True,
    shuffle=True,
    early_stopping=True,
    patience=10,
    random_state=42
)
```

**Advantages:**

* Simple and interpretable
* Fast training on linearly separable data
* Memory-efficient

**Limitations:**

* Only works well on linearly separable data
* No probabilistic outputs
* Prone to overfitting
* Sensitive to outliers

## 🚀 Usage

### Installation

```bash
pip install numpy
```

### Running the Comparison

```bash
# Basic usage
python train_compare.py

# Custom dataset
python train_compare.py --data path/to/dataset.csv

# Generate more challenging dataset
python train_compare.py --noise 1.5 --overlap 0.5

# Verbose output
python train_compare.py --verbose
```

### Using Models in Code

**Linear Regression:**

```python
from linear_regression import LinearRegressionGD
model = LinearRegressionGD(lr=0.01, l2=0.05, max_epochs=5000)
model.fit(X_train, y_train, X_val, y_val)
predictions = model.predict_classes(X_test, threshold=0.5)
print(model.converged_, model.epochs_run_, model.last_loss_)
```

**Perceptron:**

```python
from perceptron import Perceptron
model = Perceptron(lr=0.1, max_epochs=1000, early_stopping=True)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(model.converged_, model.epochs_run_, model.mistake_history_)
```

## 📊 Dataset Format

CSV format:

```csv
feature1,feature2,label
-1.234567,2.345678,0
3.456789,-0.123456,1
...
```

* Header optional
* Last column is binary label {0,1}
* Features are numeric

**Automatic Generation:**

* Two Gaussian clusters
* Configurable noise and overlap
* Feature correlation
* Balanced classes

## 📈 Metrics Tracked

* Epochs, convergence, training time
* Accuracy, Precision, Recall, F1, Confusion Matrix
* Prediction time
* Evaluation on train, validation, and test splits

## 📝 Output Files

**results.json** – structured numerical results
**analysis.md** – human-readable analysis, overfitting check, convergence, insights, recommendations

## 🔍 Understanding the Results

**Good Results:**

* Train accuracy <95%, train-test gap <10%, test accuracy >70%

**Red Flags:**

* Perfect train accuracy → Overfitting
* Large train-test gap (>15%) → Overfitting
* Test accuracy ~50% → Not learning

**Typical Results (default dataset):**

* Linear Regression: 82–88% test accuracy, converges in 100–300 epochs
* Perceptron: 78–85% test accuracy, may not converge

## 🎓 Learning Objectives

* Gradient descent vs online learning
* Regularization (L2)
* Feature standardization
* Train/validation/test splits
* Algorithm selection
* Convergence understanding

## 🐛 Troubleshooting

* Models achieve 100% → increase noise/overlap
* Perceptron doesn’t converge → expected for non-linearly separable data
* Linear Regression diverges → reduce lr or increase l2
* Large train-test gap → increase regularization, reduce epochs, or get more data

## 📚 References

* Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*
* Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*
* Rosenblatt, F. (1958). "The Perceptron"
* Novikoff, A. B. (1962). "On Convergence Proofs on Perceptrons"

## 🤝 Contributing

* Cross-validation support
* Learning rate scheduling
* More metrics (ROC-AUC, precision-recall)
* Decision boundary visualization
* Kernelized perceptron
* Multi-class support
* SGD for Linear Regression

## ✨ Acknowledgments

Built as part of Enigma-AIML25, demonstrating fundamental ML algorithms from scratch using only NumPy.

```


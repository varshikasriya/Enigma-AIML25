import numpy as np
import pandas as pd
import time
import json
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os


# 1. Linear Regression with Gradient Descent
class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
    
    def fit(self, X, y):
        m = len(X)
        self.theta = np.zeros(X.shape[1])
        self.intercept = 0
        start_time = time.time()
        
        for _ in range(self.num_iterations):
            predictions = self.predict(X)
            error = predictions - y
            gradient = (1/m) * X.T.dot(error)
            self.theta -= self.learning_rate * gradient
            self.intercept -= self.learning_rate * error.mean()
        
        self.time_to_convergence = time.time() - start_time
    
    def predict(self, X):
        return X.dot(self.theta) + self.intercept
    
    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = accuracy_score(y, np.round(predictions))
        return accuracy

# 2. Perceptron
class Perceptron:
    def __init__(self, learning_rate=0.1, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
    
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        start_time = time.time()
        
        for _ in range(self.num_iterations):
            for i in range(len(X)):
                prediction = np.dot(X[i], self.weights) + self.bias
                if y[i] * prediction <= 0:
                    self.weights += self.learning_rate * y[i] * X[i]
                    self.bias += self.learning_rate * y[i]
        
        self.time_to_convergence = time.time() - start_time
    
    def predict(self, X):
        return np.sign(np.dot(X, self.weights) + self.bias)
    
    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        return accuracy

# 3. Load Data (assuming data is in CSV format)
def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values  # features
    y = data.iloc[:, -1].values  # target
    return X, y

# 4. Evaluate models and log metrics
def evaluate_model(model, X, y, dataset_name):
    start_time_per_prediction = time.time()
    model.fit(X, y)
    time_per_prediction = time.time() - start_time_per_prediction
    
    accuracy = model.score(X, y)
    metrics = {
        "accuracy": accuracy,
        "time_to_convergence": model.time_to_convergence,
        "time_per_prediction": time_per_prediction
    }
    
    with open(f"{dataset_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

def plot_decision_boundary(model, X, y, title, filename):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01  # step size in the mesh

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    if isinstance(model, LinearRegressionGD):
        Z = np.round(model.predict(grid))
    else:  # Perceptron
        Z = model.predict(grid)

    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.grid(True)
    
    # Create output folder if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{filename}")
    plt.close()


# Example of running this code on two datasets
dataset1 = r"C:\Personal\Code\Hacktober25\Enigma-AIML25\level2-models\datasets\binary_classification.csv"
dataset2 = r"C:\Personal\Code\Hacktober25\Enigma-AIML25\level2-models\datasets\binary_classification_non_lin.csv"

# Load datasets
X1, y1 = load_data(dataset1)
X2, y2 = load_data(dataset2)

# Initialize models
linear_regression = LinearRegressionGD()
perceptron = Perceptron()

# Evaluate models
linear_regression_metrics_1 = evaluate_model(linear_regression, X1, y1, "dataset1_linear_regression")
perceptron_metrics_1 = evaluate_model(perceptron, X1, y1, "dataset1_perceptron")

plot_decision_boundary(linear_regression, X1, y1, "Linear Regression - Dataset 1", "linear_regression_dataset1.png")
plot_decision_boundary(perceptron, X1, y1, "Perceptron - Dataset 1", "perceptron_dataset1.png")

linear_regression_metrics_2 = evaluate_model(linear_regression, X2, y2, "dataset2_linear_regression")
perceptron_metrics_2 = evaluate_model(perceptron, X2, y2, "dataset2_perceptron")

plot_decision_boundary(linear_regression, X2, y2, "Linear Regression - Dataset 2", "linear_regression_dataset2.png")
plot_decision_boundary(perceptron, X2, y2, "Perceptron - Dataset 2", "perceptron_dataset2.png")

# 5. Analysis (in a markdown or text file)
analysis = """
### Model Comparison Analysis

**Dataset 1 (Linearly Separable):**

- Linear Regression Accuracy: {0}
- Perceptron Accuracy: {1}

**Dataset 2 (Non-Linearly Separable):**

- Linear Regression Accuracy: {2}
- Perceptron Accuracy: {3}

### Observations:

- Linear Regression performed better on Dataset 1 due to its linear nature.
- Perceptron was able to classify better on Dataset 1 but struggled on Dataset 2 as it is non-linear.
""".format(
    linear_regression_metrics_1["accuracy"], 
    perceptron_metrics_1["accuracy"], 
    linear_regression_metrics_2["accuracy"], 
    perceptron_metrics_2["accuracy"]
)

with open("analysis.md", "w") as f:
    f.write(analysis)


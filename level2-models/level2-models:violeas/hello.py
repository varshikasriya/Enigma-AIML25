import numpy as np
import pandas as pd
import time
import json

# Load dataset
data = pd.read_csv("datasets/binary_classification.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split into train/test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ---------------- Linear Regression using Gradient Descent ----------------
class LinearRegressionGD:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        start_time = time.time()
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.w) + self.b
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            self.w -= self.lr * dw
            self.b -= self.lr * db

        end_time = time.time()
        self.time_to_converge = end_time - start_time

    def predict(self, X):
        start_time = time.time()
        preds = np.dot(X, self.w) + self.b
        preds = np.where(preds >= 0.5, 1, 0)
        end_time = time.time()
        self.time_per_pred = (end_time - start_time) / len(X)
        return preds

# ---------------- Perceptron ----------------
class Perceptron:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        start_time = time.time()
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        y_ = np.where(y <= 0, 0, 1)
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.w) + self.b
                y_pred = np.where(linear_output >= 0, 1, 0)
                update = self.lr * (y_[idx] - y_pred)
                self.w += update * x_i
                self.b += update
        end_time = time.time()
        self.time_to_converge = end_time - start_time

    def predict(self, X):
        start_time = time.time()
        linear_output = np.dot(X, self.w) + self.b
        preds = np.where(linear_output >= 0, 1, 0)
        end_time = time.time()
        self.time_per_pred = (end_time - start_time) / len(X)
        return preds

# ---------------- Evaluation ----------------
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

# Train + test Linear Regression
linreg = LinearRegressionGD()
linreg.fit(X_train, y_train)
y_pred_lr = linreg.predict(X_test)
acc_lr = accuracy(y_test, y_pred_lr)

# Train + test Perceptron
perc = Perceptron()
perc.fit(X_train, y_train)
y_pred_p = perc.predict(X_test)
acc_p = accuracy(y_test, y_pred_p)

# ---------------- Save metrics ----------------
metrics = {
    "LinearRegression": {
        "Accuracy": acc_lr,
        "Time_to_convergence": linreg.time_to_converge,
        "Time_per_prediction": linreg.time_per_pred
    },
    "Perceptron": {
        "Accuracy": acc_p,
        "Time_to_convergence": perc.time_to_converge,
        "Time_per_prediction": perc.time_per_pred
    }
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Metrics saved in metrics.json")
print(metrics)
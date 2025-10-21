import numpy as np
import time
import csv

# LINEAR REGRESSION FROM SCRATCH
class LinearRegressionGD:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        start = time.time()
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        end = time.time()
        self.train_time = end - start

    def predict(self, X):
        start = time.time()
        preds = np.dot(X, self.weights) + self.bias
        end = time.time()
        self.pred_time = end - start
        return preds

# PERCEPTRON (BINARY CLASSIFIER)
class Perceptron:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        start = time.time()
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        # Convert labels to {0, 1}
        y = np.where(y <= 0, 0, 1)

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation(linear_output)
                update = self.lr * (y[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update

        end = time.time()
        self.train_time = end - start

    def predict(self, X):
        start = time.time()
        linear_output = np.dot(X, self.weights) + self.bias
        preds = self.activation(linear_output)
        end = time.time()
        self.pred_time = end - start
        return preds

    def activation(self, x):
        return np.where(x >= 0, 1, 0)


# METRICS & DATA HANDLING
def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


def load_csv_data(path):
    X, y = [], []
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header if any
        for row in reader:
            *features, label = map(float, row)
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)


# MAIN EXECUTION
if __name__ == "__main__":
    print("=== Models from Scratch ===")

    # Load dataset
    X, y = load_csv_data("../../datasets/binary_classification.csv")

    # Split dataset (80-20)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # ---------------- Linear Regression ----------------
    lin_reg = LinearRegressionGD(lr=0.01, epochs=1000)
    lin_reg.fit(X_train, y_train)
    y_pred_lr = lin_reg.predict(X_test)
    # Round predictions to nearest class (0/1)
    y_pred_lr_cls = np.where(y_pred_lr >= 0.5, 1, 0)
    acc_lr = accuracy_score(y_test, y_pred_lr_cls)

    # ---------------- Perceptron ----------------
    perceptron = Perceptron(lr=0.01, epochs=1000)
    perceptron.fit(X_train, y_train)
    y_pred_p = perceptron.predict(X_test)
    acc_p = accuracy_score(y_test, y_pred_p)

    # ---------------- Log Results ----------------
    print("\nResults:")
    print(f"Linear Regression Accuracy: {acc_lr:.3f}")
    print(f"Linear Regression Train Time: {lin_reg.train_time:.4f}s")
    print(f"Linear Regression Prediction Time: {lin_reg.pred_time:.6f}s")

    print(f"\nPerceptron Accuracy: {acc_p:.3f}")
    print(f"Perceptron Train Time: {perceptron.train_time:.4f}s")
    print(f"Perceptron Prediction Time: {perceptron.pred_time:.6f}s")

    # Write to analysis.txt
    with open("analysis.txt", "w") as f:
        f.write("Level 2 Analysis - Charith Reddy\n")
        f.write("--------------------------------\n")
        f.write(f"Linear Regression Accuracy: {acc_lr:.3f}\n")
        f.write(f"Linear Regression Train Time: {lin_reg.train_time:.4f}s\n")
        f.write(f"Linear Regression Prediction Time: {lin_reg.pred_time:.6f}s\n\n")

        f.write(f"Perceptron Accuracy: {acc_p:.3f}\n")
        f.write(f"Perceptron Train Time: {perceptron.train_time:.4f}s\n")
        f.write(f"Perceptron Prediction Time: {perceptron.pred_time:.6f}s\n\n")

        f.write("Observations:\n")
        f.write("- Linear Regression treats output as continuous, then thresholded for classification.\n")
        f.write("- Perceptron updates weights only for misclassified samples; better for discrete classes.\n")
        f.write("- Training time depends on number of features and epochs.\n")
        f.write("- Accuracy may differ depending on dataset separability.\n")

    print("\n✅ Analysis written to analysis.txt")

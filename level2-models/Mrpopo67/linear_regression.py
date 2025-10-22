import numpy as np
import time

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, max_iter=1000, tolerance=1e-6):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance

    def fit(self, X, y):
        start_time = time.time()
        X = np.c_[np.ones(X.shape[0]), X]  # Add bias
        self.theta = np.zeros(X.shape[1])

        for i in range(self.max_iter):
            predictions = X.dot(self.theta)
            errors = predictions - y
            gradient = (1 / len(y)) * X.T.dot(errors)
            prev_theta = self.theta.copy()
            self.theta -= self.lr * gradient

            if np.linalg.norm(self.theta - prev_theta) < self.tolerance:
                break

        self.train_time = time.time() - start_time

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        start = time.time()
        preds = X.dot(self.theta)
        end = time.time()
        self.pred_time = (end - start) / len(X)
        return np.round(preds).astype(int)  # For accuracy comparison

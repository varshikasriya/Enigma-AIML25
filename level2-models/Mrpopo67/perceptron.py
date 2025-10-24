import numpy as np
import time

class Perceptron:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.lr = learning_rate
        self.max_iter = max_iter

    def fit(self, X, y):
        start_time = time.time()
        X = np.c_[np.ones(X.shape[0]), X]  # Add bias
        self.weights = np.zeros(X.shape[1])

        for _ in range(self.max_iter):
            error_count = 0
            for xi, target in zip(X, y):
                pred = self._predict_row(xi)
                update = self.lr * (target - pred)
                if update != 0:
                    self.weights += update * xi
                    error_count += 1
            if error_count == 0:
                break

        self.train_time = time.time() - start_time

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        start = time.time()
        preds = np.where(X.dot(self.weights) >= 0, 1, 0)
        end = time.time()
        self.pred_time = (end - start) / len(X)
        return preds

    def _predict_row(self, x):
        return 1 if np.dot(x, self.weights) >= 0 else 0

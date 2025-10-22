import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.W = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        self.b = 0

        y_ = np.array(y)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.W) + self.b
                y_predicted = 1 if linear_output >= 0 else 0

                update = self.lr * (y_[idx] - y_predicted)
                self.W += update * x_i
                self.b += update

    def predict(self, X):
        linear_output = np.dot(X, self.W) + self.b
        y_predicted = np.where(linear_output >= 0, 1, 0)
        return y_predicted

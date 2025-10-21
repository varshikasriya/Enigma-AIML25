import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, convergence_tol=1e-6):
        self.learning_rate = learning_rate
        self.convergence_tol = convergence_tol
        self.W = None
        self.b = None

    def initialize(self, n_features):
        self.W = np.random.randn(n_features) * 0.01
        self.b = 0

    def sigmoid(self, z):
         return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def forward(self, X):
        z = np.dot(X, self.W) + self.b
        return self.sigmoid(z)

    def compute_cost(self, predictions, y):
        m = len(y)
        epsilon = 1e-10
        cost = - (1 / m) * np.sum(
            y * np.log(predictions + epsilon) +
            (1 - y) * np.log(1 - predictions + epsilon)
        )
        return cost

    def backward(self, X, y, predictions):   
        m = len(y)
        self.dW = np.dot(X.T, (predictions - y)) / m
        self.db = np.sum(predictions - y) / m

    def fit(self, X, y, iterations=1000):
        self.initialize(X.shape[1])
        prev_cost = float('inf')

        for i in range(iterations):
            predictions = self.forward(X)
            cost = self.compute_cost(predictions, y)
            self.backward(X, y, predictions)

            self.W -= self.learning_rate * self.dW
            self.b -= self.learning_rate * self.db

            if abs(prev_cost - cost) < self.convergence_tol:
                print(f"Converged at iteration {i}")
                break

            prev_cost = cost
            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.6f}")

    def predict(self, X):
       
        probs = self.forward(X)
        return (probs >= 0.5).astype(int)

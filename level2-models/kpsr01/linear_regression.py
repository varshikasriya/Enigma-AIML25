import numpy as np


class LinearRegression:    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.iterations_to_converge = 0
    
    def fit(self, X, y):
        """
        Args:
            X: Training features 
            y: Training labels
        """
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for iteration in range(self.max_iterations):
            # Predict
            y_pred = np.dot(X, self.weights) + self.bias
            
            #gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if np.linalg.norm(dw) < self.tolerance:
                self.iterations_to_converge = iteration + 1
                break
        else:
            self.iterations_to_converge = self.max_iterations
    
    def predict(self, X):
        """
        Make predictions
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return (linear_output >= 0.5).astype(int)

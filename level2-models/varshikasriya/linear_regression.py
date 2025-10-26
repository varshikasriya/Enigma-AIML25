import numpy as np

class LinearRegression:

    def __init__(self, learning_rate=0.01, n_iterations=1000, tolerance=1e-6):
        """
        Args:
            learning_rate: How big steps we take when updating weights (default: 0.01)
            n_iterations: Maximum number of training steps (default: 1000)
            tolerance: Stop training if improvement is smaller than this (default: 1e-6)
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.convergence_iteration = None
        
    def fit(self, X, y):
        """
        Args:
            X: Training features (shape: n_samples x n_features)
            y: Training labels (shape: n_samples)
        """
       
        n_samples, n_features = X.shape
        
       
        self.weights = np.zeros(n_features)
        self.bias = 0
        
       
        prev_loss = float('inf')
        
        
        for iteration in range(self.n_iterations):
        
            y_predicted = np.dot(X, self.weights) + self.bias
            
           
            error = y_predicted - y
        
            loss = np.mean(error ** 2)
            
            
            dw = (2 / n_samples) * np.dot(X.T, error)  # Gradient for weights
            db = (2 / n_samples) * np.sum(error)        # Gradient for bias
            
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check if we've converged (stopped improving)
            if abs(prev_loss - loss) < self.tolerance:
                self.convergence_iteration = iteration + 1
                break
            
            prev_loss = loss
        
        
        if self.convergence_iteration is None:
            self.convergence_iteration = self.n_iterations
    
    def predict(self, X):
        """
        
        Args:
            X: Features to predict on (shape: n_samples x n_features)
            
        Returns:
            Predictions (shape: n_samples)
        """
        return np.dot(X, self.weights) + self.bias
    
    def get_params(self):
        
        return {
            'weights': self.weights.tolist() if self.weights is not None else None,
            'bias': float(self.bias) if self.bias is not None else None,
            'convergence_iteration': self.convergence_iteration
        }
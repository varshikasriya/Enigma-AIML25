import numpy as np

class Perceptron:    
    """
    Implements the Perceptron learning algorithm for binary classification.
    """
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.iterations_to_converge = 0
    
    def fit(self, X, y):
        """Trains the model to find the separating hyperplane."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Standard Perceptron uses labels {-1, 1}
        y_perc = np.where(y == 0, -1, 1) 
        
        for iteration in range(self.max_iterations):
            errors = 0
            
            for idx in range(n_samples):
                x_i = X[idx]
                y_i = y_perc[idx]
                
                # Predict based on linear output
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = 1 if linear_output >= 0 else -1 # Predicted label is -1 or 1
                
                if y_pred != y_i:
                    # Update rule: W = W + LR * y_i * x_i
                    update = self.learning_rate * y_i
                    self.weights += update * x_i
                    self.bias += update
                    errors += 1
            
            if errors == 0:
                # Convergence theorem met: data is separable, and we found the solution
                self.iterations_to_converge = iteration + 1
                break
        else:
            self.iterations_to_converge = self.max_iterations
    
    def predict(self, X):
        """Returns binary predictions (0 or 1)."""
        linear_output = np.dot(X, self.weights) + self.bias
        # Convert output to 0/1 for classification report
        return (linear_output >= 0).astype(int)
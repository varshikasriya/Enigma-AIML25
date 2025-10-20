import numpy as np

class LinearRegressionSGD:
    """
    Implements Linear Regression using Stochastic Gradient Descent (SGD).
    Suitable for large datasets as it updates weights sample-by-sample.
    """
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-5):
        self.lr = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.iterations_to_converge = 0

    def fit(self, X, y):
        """Trains the model using SGD."""
        N_samples, N_features = X.shape
        self.weights = np.zeros(N_features)
        self.bias = 0
        
        # Track previous loss for convergence check
        prev_loss = np.inf

        for iteration in range(self.max_iterations):
            # Shuffle data for true SGD behavior
            indices = np.random.permutation(N_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Epoch loop (iterate over every single sample)
            for idx in range(N_samples):
                X_i, y_i = X_shuffled[idx], y_shuffled[idx]
                
                # Predict (linear output)
                y_predicted = np.dot(X_i, self.weights) + self.bias
                
                # Error (for MSE cost)
                error = y_predicted - y_i 
                
                # Calculate gradient for one sample
                dw = error * X_i
                db = error

                # Update parameters
                self.weights -= self.lr * dw
                self.bias -= self.lr * db
            
            # Check convergence by calculating MSE on the full dataset every epoch
            y_pred_full = self.predict_continuous(X)
            current_loss = np.mean((y - y_pred_full) ** 2)

            if abs(prev_loss - current_loss) < self.tolerance:
                self.iterations_to_converge = iteration + 1
                break
            prev_loss = current_loss
        else:
            self.iterations_to_converge = self.max_iterations

    def predict_continuous(self, X):
        """Returns the raw continuous output (used for loss calculation)."""
        return np.dot(X, self.weights) + self.bias
        
    def predict(self, X):
        """Returns binary prediction (0 or 1) for classification."""
        return (self.predict_continuous(X) >= 0.5).astype(int)
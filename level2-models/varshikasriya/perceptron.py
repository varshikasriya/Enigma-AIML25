import numpy as np

class Perceptron:
   
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
            learning_rate: How much to adjust weights when we make a mistake (default: 0.01)
            n_iterations: Maximum number of passes through the data (default: 1000)
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.convergence_iteration = None
    
    def _activation(self, x):
        
        return np.where(x >= 0, 1, 0)
    
    def fit(self, X, y):
        """
        Args:
            X: Training features (shape: n_samples x n_features)
            y: Training labels (shape: n_samples), should be 0 or 1
        """
        n_samples, n_features = X.shape
        
       
        self.weights = np.zeros(n_features)
        self.bias = 0
        
    
        y_ = np.array(y)
        if set(np.unique(y_)) == {-1, 1}:
           
            y_ = np.where(y_ == 1, 1, 0)
        
    
        for iteration in range(self.n_iterations):
            errors = 0  
            
        
            for idx, x_i in enumerate(X):
               
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._activation(linear_output)
                
              
                error = y_[idx] - y_predicted
                
              
                if error != 0:
                    self.weights += self.learning_rate * error * x_i
                    self.bias += self.learning_rate * error
                    errors += 1
            
            
            if errors == 0:
                self.convergence_iteration = iteration + 1
                break
        
     
        if self.convergence_iteration is None:
            self.convergence_iteration = self.n_iterations
    
    def predict(self, X):
        """
        Args:
            X: Features to predict on (shape: n_samples x n_features)
            
        Returns:
            Predictions (shape: n_samples), values are 0 or 1
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return self._activation(linear_output)
    
    def get_params(self):
        
        return {
            'weights': self.weights.tolist() if self.weights is not None else None,
            'bias': float(self.bias) if self.bias is not None else None,
            'convergence_iteration': self.convergence_iteration
        }
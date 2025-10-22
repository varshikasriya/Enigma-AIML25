# Support Vector Regressor from scratch

import numpy as np
from .base_model import BaseModel

class SVR(BaseModel):
    """
    Support Vector Regressor.
    """
    def __init__(self, kernel='linear', C=1.0, epsilon=0.1, learning_rate=0.001, n_iters=1000):
        """
        Initializes the Support Vector Regressor.

        Args:
            kernel (str): The kernel to use. Currently only 'linear' is supported.
            C (float): Regularization parameter.
            epsilon (float): Epsilon in the epsilon-insensitive loss function.
            learning_rate (float): The learning rate for gradient descent.
            n_iters (int): The number of iterations for gradient descent.
        """
        super().__init__(_forward_has_training_logic=False)
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.lr = learning_rate
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Trains the SVR model using gradient descent.

        Args:
            X (np.ndarray): Training vectors, where n_samples is the number of samples and
                            n_features is the number of features.
            y (np.ndarray): Target values.
        """
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.w) + self.b
            error = y - y_pred
            
            # Gradients
            dw = np.zeros(n_features)
            db = 0
            
            # Regularization part
            dw += self.w
            
            # Loss part
            margin_error_pos = error - self.epsilon
            margin_error_neg = -error - self.epsilon
            
            for i in range(n_samples):
                if margin_error_pos[i] > 0:
                    dw += -self.C * X[i]
                    db += -self.C
                elif margin_error_neg[i] > 0:
                    dw += self.C * X[i]
                    db += self.C
            
            # Update weights and bias
            self.w -= self.lr * dw / n_samples
            self.b -= self.lr * db / n_samples

        self.hard_set_trained(True)

    def forward(self, X):
        """
        Predicts the target values for the given data. This is a wrapper around the predict method.

        Args:
            X (np.ndarray): The input samples.

        Returns:
            np.ndarray: The predicted target values.
        """
        assert self.is_trained, "Call .fit() before .predict()"
        return self.predict(X)

    def predict(self, X):
        """
        Predicts the target values for the given data.

        Args:
            X (np.ndarray): The input samples.

        Returns:
            np.ndarray: The predicted target values.
        """
        return np.dot(X, self.w) + self.b

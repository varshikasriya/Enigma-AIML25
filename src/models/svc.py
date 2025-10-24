# Support Vector Classifier from scratch

import numpy as np
from .base_model import BaseModel

class SVC(BaseModel):
    """
    Support Vector Classifier.
    """
    def __init__(self, C=1.0, learning_rate=0.001, n_iters=1000):
        """
        Initializes the Support Vector Classifier.

        Args:
            C (float): Regularization parameter.
            learning_rate (float): The learning rate for gradient descent.
            n_iters (int): The number of iterations for gradient descent.
        """
        super().__init__(_forward_has_training_logic=False)
        self.C = C
        self.lr = learning_rate
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Trains the SVC model using gradient descent.

        Args:
            X (np.ndarray): Training vectors, where n_samples is the number of samples and
                            n_features is the number of features.
            y (np.ndarray): Target values. Assumes binary classification with labels 0 and 1.
        """
        n_samples, n_features = X.shape
        
        # Convert labels to -1 and 1
        y_ = np.where(y <= 0, -1, 1)
        
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            # Hinge loss condition
            miss = y_ * (np.dot(X, self.w) + self.b) < 1
            
            # Gradients
            dw = self.w - self.C * np.dot(X[miss].T, y_[miss])
            db = -self.C * np.sum(y_[miss])
            
            # Update weights and bias
            self.w -= self.lr * dw
            self.b -= self.lr * db

        self.hard_set_trained(True)

    def forward(self, X):
        """
        Predicts the class labels for the given data. This is a wrapper around the predict method.

        Args:
            X (np.ndarray): The input samples.

        Returns:
            np.ndarray: The predicted class labels.
        """
        return self.predict(X)

    def predict(self, X):
        """
        Predicts the class labels for the given data.

        Args:
            X (np.ndarray): The input samples.

        Returns:
            np.ndarray: The predicted class labels (0 or 1).
        """
        assert self.is_trained, "Call .fit() before .predict()"
        linear_output = np.dot(X, self.w) + self.b
        # Convert back to 0 and 1
        return np.where(linear_output >= 0, 1, 0)

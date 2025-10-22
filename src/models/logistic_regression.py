# Simple logistic regression model implementation

import numpy as np

from .base_model import BaseModel
from typing import Optional, Dict, Union

class LogisticRegression(BaseModel):
    """Simple Logistic Regression model for binary classification."""
    def __init__(
        self,
        in_features: int,
        bias_term: bool = True,
    ) -> None:
        """
        Initializes the LogisticRegression model.
        Weights and bias are initialized during forward pass (if not already).

        Args:
            in_features (int): Number of input features.
            bias_term (bool): If True, includes a bias term in the model.
        Returns:
            None
        """
        super().__init__(_forward_has_training_logic=False)
        self.in_features = in_features
        self.bias_term = bias_term
        self.weights = None
        self.bias = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Computes the sigmoid function."""
        return 1 / (1 + np.exp(-z))

    def forward(
        self,
        Xb: np.ndarray,
        yb: Optional[np.ndarray] = None,
        loss_fn: Optional[callable] = None,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Perform forward pass and compute loss.
        The model's weights and bias are initialized during the first forward pass if they are None.

        Args:
            Xb (np.ndarray): Input batch of shape (n_samples, n_features).
            yb (Optional[np.ndarray]): Target batch of shape (n_samples,).
            loss_fn (Optional[callable]): Loss function.
        Returns:
            Dict[str, Union[float, np.ndarray]]: Dict containing predictions, probabilities, and loss.
        """
        # weight init
        if self.weights is None:
            self.weights = np.random.randn(self.in_features)
        if self.bias_term and self.bias is None:
            self.bias = 0.0
        
        out = {}  # return dict

        # calculate logits
        logits = Xb @ self.weights
        if self.bias_term:
            logits += self.bias
        
        # get probabilities
        probabilities = self._sigmoid(logits)
        out["probabilities"] = probabilities

        # get predictions
        predictions = (probabilities >= 0.5).astype(int)
        out["y_pred"] = predictions

        if yb is not None and loss_fn is not None:
            loss = loss_fn(probabilities, yb)
            out["loss"] = loss
            
        return out

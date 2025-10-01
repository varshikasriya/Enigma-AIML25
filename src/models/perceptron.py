# Simple perceptron model implementation

import numpy as np
from .base_model import BaseModel
from typing import Dict

class Perceptron(BaseModel):
    """A simple perceptron model for binary classification."""
    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        bias: bool = True,
    ) -> None:
        """
        Initializes the Perceptron model.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output classes. Default is 1 for binary classification.
            bias (bool): If True, includes a bias term. Default is True.
        Returns:
            None
        """
        super().__init__(_forward_has_training_logic=True)
        self.input_size = input_size
        self.weights = np.zeros(input_size)
        self.bias = 0.0 if bias else None

    def forward(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        loss_fn: callable = None,
        print_logs: bool = False,
        return_logs: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Perform forward pass.

        The perceptron does have separate training logic in forward.
        As a result, you are expected to set the model to training mode using model.train(), indicating that the forward method should execute training logic.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
        Returns:
            np.ndarray: Predicted class labels.
        """
        out = {}
        # get linear output
        logits = np.dot(X, self.weights)
        if self.bias is not None:
            logits += self.bias
        out["logits"] = logits

        # get predictions
        preds = np.where(logits >= 0, 1, 0)
        out["preds"] = preds

        # train vs inference logic
        out["loss"] = None
        # calculate loss and update weights if model in training mode
        # finally, loss is set to not None and stored in out-dict
        return out

# Simple linear regression model implementation

import numpy as np

from .base_model import BaseModel
from typing import Optional, Dict, Union

class LinearRegression(BaseModel):
    """Simple Linear Regression model."""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias_term: bool = True,
    ) -> None:
        """
        Initializes the LinearRegression model.
        Weights and bias are initialized during forward pass (if not already).

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias_term (bool): If True, includes a bias term in the model.
        Returns:
            None
        """
        super().__init__(_forward_has_training_logic=False)
        self.in_features = in_features
        self.out_features = out_features
        self.bias_term = bias_term
        self.weights = None
        self.bias = None
    
    def forward(
        self,
        Xb: np.ndarray,
        yb: Optional[np.ndarray] = None,
        loss_fn: Optional[callable] = None,
        print_logs: bool = False,
        return_logs: bool = False,
    ) -> Dict[str, Union[float, list, np.ndarray]]:
        """
        Perform forward pass and compute loss.
        If in training mode (self._training is True), performs training logic.
        If in evaluation mode (self._training is False), performs inference logic.
        The model's weights and bias are initialized during the first forward pass if they are None.

        Known issue(s):
        - Model's weight initialization is not deterministic
        - Weight initialization should ideally be done in __init__
        - Initialized weights tend to be large, causing instability during optimization

        Args:
            Xb (np.ndarray): Input batch.
            yb (Optional[np.ndarray]): Target batch.
            loss_fn (Optional[callable]): Loss function.
            print_logs (bool): Print logs if True.
            return_logs (bool): Return logs if True.
        Returns:
            Dict[str, Union[float, list, np.ndarray]]: Dict containing loss, predictions, and other logs (if training).
        """
        # weight init to be done
        if self.weights is None:
            self.weights = np.random.randn(self.in_features, self.out_features)
        if self.bias_term and self.bias is None:
            self.bias = np.zeros((1, self.out_features))

        out = {}  # return dict
        if self._training:
            # training logic here, like:
            # weight init and bias init (if bias_term is True)
            # optimize and log information
            # optimization logic here
            # logging logic here
            # if log_speed is some int, print average loss, accuracy, etc every log_speed epochs
            self._is_trained = False
            pass
            # return the return_logs
        else:
            # inference logic here
            pass
        
        return out

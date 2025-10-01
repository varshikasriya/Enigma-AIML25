# Simple linear regression model implementation

import numpy as np

from .base_model import BaseModel
from typing import Optional, Dict, Union

class LinearRegression(BaseModel):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias_term: bool = True,
    ) -> None:
        super().__init__(_forward_has_training_logic=True)
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
    ) -> Union[Dict[str, list], np.ndarray]:
        """
        Perform forward pass and compute loss,

        Args:
            Xb (np.ndarray): Input batch.
            yb (Optional[np.ndarray]): Target batch.
            loss_fn (Optional[callable]): Loss function.
            print_logs (bool): Print logs if True.
            return_logs (bool): Return logs if True.
        Returns:
            Optional[Dict[str, list]]: Logs if return_logs is True.
        """
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
        return None
        # return the predictions


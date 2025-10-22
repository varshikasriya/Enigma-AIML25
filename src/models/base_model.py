# Base model class for all custom model implementations.
# All torch models will retain their torch-API.
# Docstring expected to be made for all child classes.

import torch
import numpy as np

from typing import Union

class BaseModel:
    """Base class for all models."""
    def __init__(
        self,
        _forward_has_training_logic: bool,
        _is_trained: bool = False,
        _training: bool = False,
    ) -> None:
        """
        Initializes the BaseModel.
        
        Following attributes are initialized:
        - self._forward_has_training_logic (bool): Indicates if the model has distinct training logic.
        - self._is_trained (bool): Indicates if the model has been trained.
        - self._training (bool): Indicates if the model is in training mode (for training logic).

        For self._training to be effective, the child class must:
        1. Implement training logic in the forward method.
        2. Set self._forward_has_training_logic = True in its __init__ method.
        If self._forward_has_training_logic is False, self._training has no effect and an error is raised if train() is called.
        
        Args:
            _forward_has_training_logic (bool): If True, model has distinct training logic in forward.
            _is_trained (bool): If True, model is considered trained.
            _training (bool): If True, model is in training mode.
        Returns:
            None

        Note for child classes, BaseModel's __init__ must be called as:
        ```python
        super().__init__( _forward_has_training_logic=True/False, ...)
        ```
        Its recommended to **not override** other attributes of BaseModel in child class __init__.
        """
        self._forward_has_training_logic = False
        self._is_trained = False
        self._training = False

    def train(self) -> None:
        """
        Sets the model to training mode.
        Different from torch's train() method.
        Here the model's forward method will run training logic if self._training is True.
        Only modules with training logic in forward will use this flag (like linear-regression).

        It's is very important to understand this distinction. Models like LinearRegression have different logic for training and inference in the same forward method.
        
        Models that can be trained with a separate optimizer, like neural networks, will not use this flag.
        """
        assert self._forward_has_training_logic, "Model does not have training logic implemented."
        self._training = True
    def eval(self) -> None:
        """
        Sets the model to evaluation mode.
        Different from torch's eval() method.
        Here the model's forward method will run inference logic only.

        Note that this method can be called even if the model does not have training logic in forward.
        Unlike train(), eval() is a safe operation for all models, and will not raise an error (even if self._forward_has_training_logic is False).
        """
        self._training = False

    def hard_set_trained(
        self,
        val: bool,
    ) -> None:
        """
        Sets the model trained status manually.
        Useful for models that do not have distinct training logic in forward, but are trained externally (eg: neural networks trained with optimizers).
        """
        self._is_trained = val

    @property
    def is_trained(self) -> bool:
        """Returns the trained status of the model."""
        return self._is_trained

    def forward():
        """BaseModel forward method. To be overridden by the child class."""
        raise NotImplementedError("Fit method must be implemented by the subclass.")
    
    def save(
        self,
        f_path: str,
    ) -> None:
        """
        Saves the model to the specified file path.
        Models are saved in .pkl format using python's pickle module.

        Torch models are saved using torch's save API (.pth or .pt format).
        
        Args:
            f_path (str): Path to save the model.
        Returns:
            None
        """
        # logic for model save to be implemented...
        raise NotImplementedError("Save method must be implemented by the subclass.")
    
    def load(
        self,
        f_path: str,
    ) -> None:
        """
        Loads the model from the specified file path.
        Models are loaded from .pkl format using python's pickle module.
        Model is initialized with the loaded parameters, loaded model instance is not returned.
        
        Torch models are loaded using torch's load API (.pth or .pt format).

        Args:
            f_path (str): Path to load the model from.
        Returns:
            None
        """
        # logic for model loading to be implemented...
        raise NotImplementedError("Load method must be implemented by the subclass.")

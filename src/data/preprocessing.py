import torch
import numpy as np

from typing import Union, Optional

class Preprocessor:
    """
    Preprocessing class for data normalization, standardization, and augmentation.

    Example usage:
    
    ```python
    # for performing some pre-processing step directly on data
    from augment import some_augment_function
    preprocessor = Preprocessor()
    data = preprocessor.normalize(data)
    data = preprocessor.compose(data, augment_fn=some_augment_function)

    # if you wish to setup a preprocessing pipeline
    preprocessor = Preprocessor(as_pipeline=True, compose_fns=[...], normalize=True, standardize=True)
    processed_data = preprocessor(data)
    ```
    """
    def __init__(
        self,
        as_pipeline: bool = False,
        *args,
        **kwargs
    ) -> None:
        """
        Initializes the Preprocessor.

        Args:
            as_pipeline (bool): If True, sets up as a pipeline.
        """
        self.as_pipeline = as_pipeline
        if as_pipeline:
            # Setup pipeline components here
            pass

    def normalize(
        self,
        data: np.ndarray,
        axis=None,
        kind: str = "minmax",
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalize data using specified normalization method.

        Args:
            data (np.ndarray): Input data.
            axis (int or tuple, optional): Axis or axes along which to normalize.
            kind (str): Type of normalization to apply. Options:
                - "minmax": Scale data to [0, 1] range (default)
                - "l1": L1 normalization (sum of absolute values = 1)
                - "l2": L2 normalization (sum of squares = 1, for cosine similarity)
        Returns:
            np.ndarray: Normalized data.
        """
        if kind == "minmax":
            data_min = np.min(data, axis=axis, keepdims=True)
            data_max = np.max(data, axis=axis, keepdims=True)
            return (data - data_min) / (data_max - data_min + 1e-8)
        elif kind == "l1":
            norm = np.sum(np.abs(data), axis=axis, keepdims=True)
            return data / (norm + 1e-8)
        elif kind == "l2":
            norm = np.sqrt(np.sum(data ** 2, axis=axis, keepdims=True))
            return data / (norm + 1e-8)
        else:
            raise ValueError(f"Unknown normalization kind: {kind}. Choose from 'minmax', 'l1', or 'l2'.")

    def standardize(
        self,
        data: np.ndarray,
        axis=None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Standardize data to zero mean and unit variance.

        Args:
            data (np.ndarray): Input data.
            axis (int or tuple, optional): Axis or axes along which to standardize.
        Returns:
            np.ndarray: Standardized data.
        """
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        return (data - mean) / (std + 1e-8)

    def compose(
        self,
        data: Union[np.ndarray, torch.Tensor],
        augment_fn: Optional[callable] = None,
        **kwargs,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Compose preprocessing with augmentation.

        Args:
            data (np.ndarray): Input data.
            augment_fn (callable, optional): Augmentation function from augment.py.
            **kwargs: Additional arguments for augment_fn.
        Returns:
            np.ndarray: Preprocessed (and possibly augmented) data.
        """
        if augment_fn is not None:
            data = augment_fn(data, **kwargs)
        
        return data

    def __call__(
        self,
        data: Union[np.ndarray, torch.Tensor],
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply the preprocessing pipeline to data.

        Args:
            data (np.ndarray): Input data.
            **kwargs: Additional arguments for pipeline functions.
        Returns:
            np.ndarray: Preprocessed data.
        """
        assert self.as_pipeline, "Preprocessor not set up as a pipeline."
        
        raise NotImplementedError("Pipeline functionality not yet implemented.")

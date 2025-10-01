import os
import json
import random
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from src.models import BaseModel
from src.data import DataLoader, EndOfDataLoader

from tqdm.auto import tqdm
from typing import Callable, Optional, List, Dict, Union, Literal

class Trainer:
    """
    Trainer class to handle training and validation of models.
    The trainer supports both numpy-based models (subclass of BaseModel) and PyTorch models (subclass of nn.Module)

    Trainer also handles logging losses and other metrics during training and validation of models.
    The logs are stored in a directory as a JSON file.
    Logs are also returned if requested during training.

    This class handles both forms of DataLoaders:
    1. Custom DataLoader for numpy arrays (from src.data)
    2. PyTorch DataLoader (torch.utils.data.DataLoader)
    
    So, when using Trainer with PyTorch models, you have to use torch DataLoaders.
    When using Trainer with BaseModel models, you have to use custom DataLoaders.
    """
    def __init__(
        self,
        model: Union[BaseModel, nn.Module],
        loss_fn: Callable,
        *metrics: Callable,
        train_dataloader: Union[DataLoader, torch.utils.data.DataLoader],
        val_dataloader: Optional[Union[DataLoader, torch.utils.data.DataLoader]] = None,
        log_dir: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Initializes the Trainer.

        Args:
            model (Union[BaseModel, nn.Module]): The model to train.
            loss_fn (Callable): The loss function.
            *metrics (Callable): Additional metric functions.
            train_dataloader (Union[DataLoader, torch.utils.data.DataLoader]): DataLoader for training data.
            val_dataloader (Optional[Union[DataLoader, torch.utils.data.DataLoader]]): DataLoader for validation data.
            log_dir (Optional[str]): Directory to save logs, if None, logs are saved to `src/runs/<date_time>/logs.json`.
            **kwargs: Additional arguments used for torch model training (like optimizer, device, etc...).
        Returns:
            None
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.loss_fn = loss_fn
        self.metrics = metrics
        
        self._model_type = type(model)
        if issubclass(self._model_type, BaseModel):
            self._backend = "numpy"
        elif issubclass(self._model_type, nn.Module):
            self._backend = "torch"
        else:
            raise ValueError(f"Unsupported model type: {self._model_type}")
        
        self.log = {
            "train_loss": [],
            "val_loss": [],
        }
        for metric in self.metrics:
            self.log[f"train_{metric.__name__}"] = []
            self.log[f"val_{metric.__name__}"] = []
        self.log_keys = list(self.log.keys())  # we will use this to log stuff easily during train

        # setup stuff for torch models
        if self._backend == "torch":
            if not isinstance(train_dataloader, torch.utils.data.DataLoader):
                raise ValueError("For torch models, train_dataloader must be of type torch.utils.data.DataLoader")
            if val_dataloader is not None and not isinstance(val_dataloader, torch.utils.data.DataLoader):
                raise ValueError("For torch models, val_dataloader must be of type torch.utils.data.DataLoader")
            
            self.optimizer = kwargs.get("optimizer", None)
            if self.optimizer is None:
                raise ValueError("For torch models, optimizer must be provided in kwargs")
            self.device = kwargs.get("device", torch.device("cpu"))
            self.model.to("cpu")  # model moved to device during training only
    
    def train(
        self,
        epochs: int,
        return_log: bool = False,
        print_running_logs: bool = True,
        save_model: bool = True,
        save_best_only: bool = False,
        best_model_metric: Optional[Union[Callable, Literal["loss"]]] = None,
        best_model_mode: Optional[str] = None,
    ) -> Optional[Dict[str, List[float]]]:
        """
        **IMPORTANT**: This method is not yet defined, will be done soon.

        Trains the model for a specified number of epochs.

        **Model evaluation and logging**:
        Model is evaluated on validation set (if not None) after each epoch on training.
        Logs are printed after each epoch if `print_running_logs` is True.

        **Model saving**:
        Model is saved after training if `save_model` is True.
        If `save_best_only` is True, only the best model (based on `best_model_metric`) is saved.
            In cases where validation dataloader is None, the trainer will save model based on training loss.
        If `best_model_metric` is None, "loss" is used as the metric to monitor for best model saving.
        `best_model_mode` is used to compare the best metric value, if "min", lower is better, if "max", higher is better.

        Args:
            epochs (int): Number of epochs to train.
            return_log (bool): If True, returns the training log after training.
            print_running_logs (bool): If True, prints running logs after each epoch.
            save_model (bool): If True, saves the model after training.
            save_best_only (bool): If True, saves only the best model based on a specified metric.
            best_model_metric (Optional[Union[Callable, Literal["loss"]]]): Metric function or "loss" to monitor for best model saving.
            best_model_mode (Optional[str]): "min" or "max" to indicate whether the best metric is the minimum or maximum value.
        Returns:
            Optional[Dict[str, List[float]]]: Training log if return_log is True, else None.
        """
        # create log directory if not exists
        # log dir to be created with training start time and date
        if log_dir is None:
            log_dir = os.path.join("src", "runs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        
        ##############################
        # MODEL TRAINING AND LOGGING #
        ##############################

        # once model training is done, save the model and logs
        # idk what the code below is doing, copilot wrote it. If it leads to errors, we shall degub them (or use claude lol).
        # too much work to verify it rn, i might do it later... :)
        # it looks fine to me tho, we will see
        if save_model:
            # save model
            model_path = os.path.join(log_dir, "model.pth" if self._backend == "torch" else "model.pkl")
            if save_best_only:
                if best_model_metric is None:
                    best_model_metric = "loss"
                if best_model_mode is None:
                    best_model_mode = "min"
                if self._backend == "torch":
                    torch.save(self._best_model_state, model_path)
                else:
                    self.model.save(model_path)
            else:
                if self._backend == "torch":
                    torch.save(self.model.state_dict(), model_path)
                else:
                    self.model.save(model_path)
            # save logs
            log_path = os.path.join(log_dir, "logs.json")
            with open(log_path, "w") as f:
                json.dump(self.log, f, indent=4)

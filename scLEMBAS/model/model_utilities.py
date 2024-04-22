"""
Helper functions for building the model.
"""
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
import torchmetrics

def np_to_torch(arr: np.array, dtype: torch.float32, device: str = 'cpu'):
    """Convert a numpy array to a torch.tensor

    Parameters
    ----------
    arr : np.array
        
    dtype : torch.dtype, optional
        datatype to store values in torch, by default torch.float32
    device : str
        whether to use gpu ("cuda") or cpu ("cpu"), by default "cpu"
    """
    return torch.tensor(arr, dtype=dtype, device = device)

def update_with_defaults(default_parameters: dict, user_parameters: dict, additional_parameters: Optional[List[str]] = None)->dict:
    """Updates a dictionary of user-provided parameters with default parameters where missing. 

    Parameters
    ----------
    default_parameters : dict
        all default parameters
    user_parameters : dict
        user-provided parameters
    additional_parameters : Optional[List[str]], optional
        additional keys that are not provided in default parameters, by default None

    Returns
    -------
    params : dict
        the user-provided parameters, updated for missing values
    """

    allowed_params = list(default_parameters.keys())
    if additional_parameters:
        allowed_params += additional_parameters
    
    params = {**default_parameters.copy(), **user_parameters}
    params = {k: v for k,v in params.items() if k in allowed_params}

    return params

def get_lr(epoch: int, max_epoch: int, max_height: float = 1e-3, 
             start_height: float=1e-5, end_height: float=1e-5, 
             peak: int = 1000):
    """Calculates learning rate for a given epoch during training.

    Parameters
    ----------
    epoch : int
        the current epochs
    max_epoch : int
        the maximum number of training epochss
    max_height : float, optional
        tuning parameters for learning for the first 95% of epochss, by default 1e-3
    start_height : float, optional
        tuning parameter for learning rate before peak epochss, by default 1e-5
    end_height : float, optional
        tuning parameter for learning rate afer peak epochss, by default 1e-5
    peak : int, optional
        the first # of epochss to calculate lr on (should be less than 95% 
        of max_epoch), by default 1000

    Returns
    -------
    lr : float
        the learning rate
    """

    phase_length = 0.95 * max_epoch
    if epoch<=peak:
        effective_epoch = epoch/peak
        lr = (max_height-start_height) * 0.5 * (np.cos(np.pi*(effective_epoch+1))+1) + start_height
    elif epoch<=phase_length:
        effective_epoch = (epoch-peak)/(phase_length-peak)
        lr = (max_height-end_height) * 0.5 * (np.cos(np.pi*(effective_epoch+2))+1) + end_height
    else:
        lr = end_height
    return lr

class StandardDeviationMetric(torchmetrics.Metric):
    def __init__(self):
        super().__init__(dist_sync_on_step=False)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_of_squares", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds):
        self.sum += torch.sum(preds)
        self.sum_of_squares += torch.sum(preds ** 2)
        self.total += preds.numel()

    def compute(self):
        mean = self.sum / self.total
        return torch.sqrt(self.sum_of_squares / self.total - mean ** 2)

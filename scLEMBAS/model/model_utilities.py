"""
Helper functions for building the model.
"""
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn


def freeze_model(model):
    model.eval() 
    for param in model.parameters():
        param.requires_grad = False
        
def unfreeze_model(model):
    model.train()
    for param in model.parameters():
        param.requires_grad = True

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

def L2_reg(lambda_L2: float, parameter: torch.Tensor):
    if lambda_L2 == 0:
        return torch.tensor(0.0, device=parameter.device, dtype=parameter.dtype)
    else:
        return lambda_L2*torch.sum(torch.square(parameter))

def L1_reg(lambda_L1: float, parameter: torch.Tensor):
    if lambda_L1 == 0:
        return torch.tensor(0.0, device=parameter.device, dtype=parameter.dtype)
    else:
        return lambda_L1*torch.sum(torch.abs(parameter))


def kl_divergence_normal(empirical_values, mu=0.0, sigma=1.0, eps=1e-12):
    """
    KL divergence between empirical distribution of weights (assumed Gaussian) 
    and N(mu, sigma^2). Only non-zero weights are considered.
    
    empirical_values is a 1D torch tensor of the empirical distribution that should match the target
    mu is the target distribution mean, sigma is the target distribution standard deviation
    epsilon is a minmum value to add to avoid dividing by zero
    """
    q_mean = empirical_values.mean()
    q_std = empirical_values.std(unbiased=False) + eps

    # KL divergence between N(q_mean, q_std^2) and N(mu, sigma^2)
    kl = torch.log(sigma / q_std) + (q_std**2 + (q_mean - mu)**2) / (2 * sigma**2) - 0.5
    return kl

    
# class WeightUpdateMonitor:
#     def __init__(self, model, atol=1e-6):
#         self.model = model
#         self.atol = atol
#         self._saved_weights = {}

#     def save(self):
#         """Save a snapshot of current weights and gradients (if exist)."""
#         self._saved_weights = {
#             name: param.detach().clone()
#             for name, param in self.model.named_parameters()
#         }
#         self._saved_grads = {
#             name: param.grad.detach().clone() if param.grad is not None else None
#             for name, param in self.model.named_parameters()
#         }

#     def check(self, verbose=True):
#         """Report norm deltas only when weight or gradient has changed."""
#         changed = False
#         for name, param in self.model.named_parameters():
#             old_weight = self._saved_weights[name]
#             new_weight = param.detach()
#             delta = new_weight - old_weight
#             delta_norm = delta.norm().item()
#             weight_changed = delta_norm > self.atol
#             changed = changed or weight_changed

#             # Check gradient
#             grad = param.grad
#             grad_old = self._saved_grads.get(name, None)
#             grad_changed = False
#             grad_norm = None

#             if grad is not None:
#                 grad_norm = grad.norm().item()
#                 if grad_old is None:
#                     grad_changed = True
#                 else:
#                     grad_changed = not torch.allclose(grad, grad_old, atol=self.atol)


#             if verbose and (weight_changed or grad_changed):
#                 print(f"{name}:")
#                 if weight_changed:
#                     print(f" weight norm change: {delta_norm:.3e}")
#                 if grad_changed:
#                     print(f"grad norm:   {grad_norm:.3e}")
#             if not verbose:
#                 return changed
 
    

# def get_spectral_radius(weights: nn.parameter.Parameter):
#     """_summary_

#     Parameters
#     ----------
#     weights : nn.parameter.Parameter
#         the interaction weights

#     Returns
#     -------
#     spectral_radius : np.ndarray
#         a single element numpy array representing the denominator of the scaling factor for weights 
#     """
#     A = scipy.sparse.csr_matrix(weights.detach().numpy())
#     eigen_value, _ = eigs(A, k = 1) # first eigen value
#     spectral_radius = np.abs(eigen_value)
#     return spectral_radius

# def expected_uniform_distribution(Y_full: torch.Tensor, target_min: float = 0.0, target_max: float = None):
#     """Calculate the distance between the signaling network node values and a desired uniform distribution of the node values

#     Parameters
#     ----------
#     Y_full : torch.Tensor
#         the signaling network scaled by learned interaction weights. Shape is (samples x network nodes). 
#         Output of BioNet.
#     target_min : float, optional
#         minimum values for nodes in Y_full to take on, by default 0.0
#     target_max : float, optional
#         maximum values for nodes in Y_full to take on, by default 0.8

#     Returns
#     -------
#     loss : torch.Tensor
#         the regularization term
#     """
#     target_distribution = torch.linspace(target_min, target_max, Y_full.shape[0], dtype=Y_full.dtype, device=Y_full.device).reshape(-1, 1)

#     sorted_Y_full, _ = torch.sort(Y_full, axis=0) # sorts each column (signaling network node) in ascending order
#     dist_loss = torch.sum(torch.square(sorted_Y_full - target_distribution)) # difference in distribution
#     below_range = torch.sum(Y_full.lt(target_min) * torch.square(Y_full-target_min)) # those that are below the minimum value
#     above_range = torch.sum(Y_full.gt(target_max) * torch.square(Y_full-target_max)) # those that are above the maximum value
#     loss = dist_loss + below_range + above_range
#     return loss

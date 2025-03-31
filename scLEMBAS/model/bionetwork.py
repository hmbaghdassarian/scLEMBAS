"""
Builds the core signaling-network topology based RNN.
"""

from annotated_types import Ge
from collections import OrderedDict
import copy
from typing import Dict, List, Optional, Annotated
import warnings

import pandas as pd
import numpy as np 
import scipy
from scipy.sparse.linalg import eigs

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utilities import np_to_torch, update_with_defaults, L2_reg, L1_reg
from .activation_functions import activation_function_map
from .model_components import GaussianVariationalEncoder
from ..utilities import set_seeds


class BioNetBase(nn.Module):
    """Builds the RNN on the signaling network topology."""
    
    DEFAULT_PARAMETERS = {'target_steps': 100, 'max_steps': 300, 'exp_factor': 20, 'leak': 0.01, 'tolerance': 1e-5}

    def __init__(self, edge_list: np.array, 
                 edge_MOA: np.array, 
                 input_node_idx: torch.Tensor,
                 n_network_nodes: int, 
                 activation_function: str = 'MML', 
                 bionet_params: Optional[Dict[str, float]] = None, 
                 dtype: torch.dtype=torch.float32, 
                device: str = 'cpu', 
                seed: int = 888):
        """Initialization method.

        Parameters
        ----------
        edge_list : np.array
            a (2, net.shape[0]) array where the first row represents the indices for the target node and the 
            second row represents the indices for the source node. net.shape[0] is the total # of interactions
            output from  `SignalingModel.parse_network` 
        edge_MOA : np.array
            a (2, net.shape[0]) array where the first row is a boolean of whether the interactions are stimulating and the 
            second row is a boolean of whether the interactions are inhibiting
            output from  `SignalingModel.parse_network`
        input_node_idx : torch.Tensor
            array of indeces representing the ligand nodes in the signaling network. 
            stored in `ProjectInput.input_node_idx`
        n_network_nodes : int
            the number of nodes in the network
        bionet_params : Dict[str, float]
            training parameters for the model, by default None
            see `SignalingModel.set_training_parameters`
        activation_function : str, optional
            RNN activation function, by default 'MML'
            options include:
                - 'MML': Michaelis-Menten-like
                - 'leaky_relu': Leaky ReLU
                - 'sigmoid': sigmoid 
        dtype : torch.dtype, optional
           datatype to store values in torch, by default torch.float32
        device : str
            whether to use gpu ("cuda") or cpu ("cpu"), by default "cpu"
        seed : int
            random seed for torch and numpy operations, by default 888
        """
        super().__init__()

        if not bionet_params:
            bionet_params = {}
        self.bionet_params = self.set_bionet_parameters(**bionet_params)

        self.dtype = dtype
        self.device = device
        self.seed = seed
        self._ss_seed_counter = 0
        self._prescaled_weights = False
        self.input_node_idx = input_node_idx

        self.n_network_nodes = n_network_nodes
        # TODO: delete these _in _out?
        self.n_network_nodes_in = n_network_nodes
        self.n_network_nodes_out = n_network_nodes

        self.edge_list = (np_to_torch(edge_list[0,:], dtype = torch.int32, device = 'cpu'), 
                          np_to_torch(edge_list[1,:], dtype = torch.int32, device = 'cpu'))
        self.edge_MOA = np_to_torch(edge_MOA, dtype=torch.bool, device = self.device)

        # initialize weights and biases
        self.initialize_weights()
        
        # activation function
        self.activation = activation_function_map[activation_function]['activation']
        self.delta = activation_function_map[activation_function]['delta']
        self.onestepdelta_activation_factor = activation_function_map[activation_function]['onestepdelta']
        
#     def get_device(self):
#         if self.device == 'cuda':
#             device = next(self.parameters()).device
#             return device.type + ':{}'.format(device.index)
#         else:
#             return self.device
        
    def initialize_weight_values(self):
        raise ValueError('This must be overwritten by the class that inherits it')
    def make_mask(self):
        raise ValueError('This must be overwritten by the class that inherits it')
    def implement_mask(self):
        raise ValueError('This must be overwritten by the class that inherits it')
    def forward(self):
        raise ValueError('This must be overwritten by the class that inherits it')

    def L2_reg(self):
        """Get the L2 regularization term for the neural network parameters."""
        raise ValueError('Define in each specific class')

    def set_bionet_parameters(self, **attributes):
        """Set the parameters for various calculations in the model. Overrides default parameters with attributes if specified.
        Adapted from LEMBAS `trainingParameters`
    
        Parameters
        ----------
        attributes : dict
            keys are parameter names and values are parameter value
        """
        # #set defaults

        params = update_with_defaults(default_parameters = self.DEFAULT_PARAMETERS, 
                                      user_parameters = attributes, 
                                      additional_parameters = ['spectral_target'])
        # spectral_target not in defaults because default is to calculate
        # it as a function of 'tolerance' and 'target_steps'
        if 'spectral_target' not in params.keys():
            params['spectral_target'] = np.exp(np.log(params['tolerance'])/params['target_steps'])
    
        return params

    def initialize_MOA(self):
        """Generates weights corresponding to adjacency matrix for non-interacting nodes AND nodes where 
        mode of action (stimulating/inhibiting) is unknown.
        
        Returns
        -------
        weights_MOA : torch.Tensor
            an adjacency matrix of all nodes in the signaling network, with activating interactions set to 1, inhibiting interactions set 
            to -1, and interactions that do not exist or have an unknown mechanism of action (stimulating/inhibiting) set to 0
        """
        self.weights_MOA = torch.zeros(self.n_network_nodes_out, self.n_network_nodes_in, dtype=torch.long, device = self.device) # adjacency matrix
        signed_MOA = self.edge_MOA[0, :].type(torch.long) - self.edge_MOA[1, :].type(torch.long) #1=activation -1=inhibition, 0=unknown
        self.weights_MOA[self.edge_list] = signed_MOA

    def initialize_weights(self):
        """Initializes weights and masks for interacting nodes and mechanism of action.

        Returns
        -------
        weights : torch.Tensor
            a torch.Tensor adjacency matrix with randomly initialized values for each signaling network interaction
        bias : torch.Tensor
            a torch.Tensor with randomly initialized values for each signaling network node
        """

        weight_values = self.initialize_weight_values()
        
        # adjaceny matrix: rows are targets, columns are sources
        weights = torch.zeros(self.n_network_nodes, self.n_network_nodes, dtype=self.dtype, device = self.device)
        weights[self.edge_list] = weight_values
        self.weights = nn.Parameter(weights)
        
        self.initialize_MOA()

    def prescale_weights(self, target_radius: float = 0.8):
        """Scale weights according to spectral radius
    
        Parameters
        ----------
        target_radius : float, optional
            _description_, by default 0.8
        """

        A = scipy.sparse.csr_matrix(self.weights.detach().cpu().numpy())
        np.random.seed(self.seed)
        eigen_value, _ = eigs(A, k = 1, v0 = np.random.rand(A.shape[0])) # first eigen value
        spectral_radius = np.abs(eigen_value)
        
        factor = target_radius/spectral_radius.item()
        self.weights.data = self.weights.data * factor
        
        self._prescaled_weights = True

    def get_sign_mistmatch(self):
        """Identifies edge weights in network that have a sign that does not agree
        with the known mode of action.
    
        Mode of action: stimulating interactions are expected to have positive weights and inhibiting interactions
        are expected to have negative weights.
        
        Returns
        -------
        sign_mismatch : torch.Tensor
            a binary adjacency matrix of all nodes in the signaling network, where values are 1 if they do not 
            match the mode of action and 0 if they match the mode of action or have an unknown mode of action
        """
        sign_mismatch = torch.ne(torch.sign(self.weights), self.weights_MOA).type(self.dtype) 
        sign_mismatch = sign_mismatch.masked_fill(self.mask_MOA, 0) # do not penalize sign mismatches of unknown interactions
    
        return sign_mismatch

    def count_sign_mismatch(self):
        """Counts total sign mismatches identified in `get_sign_mistmatch`
        
        Returns
        -------
        n_sign_mismatches : float
            the total number of sign mismatches at `iter`
        """
        n_sign_mismatches = torch.sum(self.get_sign_mistmatch()).item()
        return n_sign_mismatches

    def sign_regularization(self, lambda_L1: Annotated[float, Ge(0)] = 0):
        """Get the L1 regularization term for the neural network parameters that 
        do not fit the mechanism of action (i.e., negative weights for stimulating interactions or positive weights for inhibiting interactions).
        Only penalizes sign mismatches of known MOA.
    
        Parameters
        ----------
        lambda_L1 : Annotated[float, Ge(0)]
            the regularization parameter, by default 0 (no penalty) 
    
        Returns
        -------
        loss : torch.Tensor
            the regularization term
        """
        lambda_L1 = torch.tensor(lambda_L1, dtype = self.dtype, device = self.device)
        sign_mismatch = self.get_sign_mistmatch() # will not penalize sign mismatches of unknown interactions

        loss = lambda_L1 * torch.sum(torch.abs(self.weights * sign_mismatch))
        return loss

    def get_SS_loss(self, Y_full: torch.Tensor, spectral_loss_factor: float, subset_n: int = 5, **kwargs):
        """_summary_
    
        Parameters
        ----------
        Y_full : torch.Tensor
            output of the forward pass
            ensure to run `torch.Tensor.detach` method prior to inputting so that gradient calculations are not effected
        spectral_loss_factor : float
            _description_
        subset_n : int, optional
            _description_, by default 5
            default was 10 when using `_depr_get_SS_deviation`
    
        Returns
        -------
        _type_
            _description_
        """
        spectral_loss_factor = torch.tensor(spectral_loss_factor, dtype=Y_full.dtype, device=Y_full.device)
        exp_factor = torch.tensor(self.bionet_params['exp_factor'], dtype=Y_full.dtype, device=Y_full.device)
    
        if self.seed:
            np.random.seed(self.seed + self._ss_seed_counter)
        selected_values = np.random.permutation(Y_full.shape[0])[:subset_n]
    
        SS_deviation, aprox_spectral_radius = self._get_SS_deviation(Y_full[selected_values,:], **kwargs)        
        spectral_radius_factor = torch.exp(exp_factor*(aprox_spectral_radius-self.bionet_params['spectral_target']))
        
        loss = spectral_radius_factor * SS_deviation/torch.sum(SS_deviation.detach())
        loss = spectral_loss_factor * torch.sum(loss)
        aprox_spectral_radius = torch.mean(aprox_spectral_radius).item()
    
        self._ss_seed_counter += 1 # new seed each time this (and _get_SS_deviation) is called
    
        return loss, aprox_spectral_radius
    
    def _get_SS_deviation(self, Y_full_sub, n_probes: int = 5, power_steps: int = 5):
        """Quicker version of spectral radius implemented by Olof Nordenstorm."""
        x_prime = self.onestepdelta_activation_factor(Y_full_sub, self.bionet_params['leak'])     
        x_prime = x_prime.unsqueeze(2)
        
        T = x_prime * self.weights
        if self.seed:
            set_seeds(self.seed + self._ss_seed_counter)
        delta = torch.randn((Y_full_sub.shape[0], Y_full_sub.shape[1], n_probes), dtype=Y_full_sub.dtype, device=Y_full_sub.device)
        for i in range(power_steps):
            new = delta / torch.norm(delta,dim=1).unsqueeze(1)
            delta = torch.matmul(T, new)

        new_delta = torch.matmul(T, delta)
        batch_eigen_not_norm=torch.einsum('ijk,ijk->ik',new_delta,delta)
        normalize=torch.einsum('ijk,ijk->ik',delta,delta)
        batch_SR_values,_=torch.max(torch.abs(batch_eigen_not_norm/normalize),axis=1) # spectral radius approx 

        aprox_spectral_radius = torch.mean(batch_SR_values, axis=0)      
        SS_deviation = batch_SR_values
    
        return SS_deviation, aprox_spectral_radius
    
    def _depr_get_SS_deviation(self, Y_full_sub, n_probes: int = 5, power_steps: int = 50):
        x_prime = self.onestepdelta_activation_factor(Y_full_sub, self.bionet_params['leak'])     
        x_prime = x_prime.unsqueeze(2)
        
        T = x_prime * self.weights
        if self.seed:
            set_seeds(self.seed + self._ss_seed_counter)
        delta = torch.randn((Y_full_sub.shape[0], Y_full_sub.shape[1], n_probes), dtype=Y_full_sub.dtype, device=Y_full_sub.device)
        for i in range(power_steps):
            new = delta
            delta = torch.matmul(T, new)
    
        SS_deviation = torch.max(torch.abs(delta), axis=1)[0]
        aprox_spectral_radius = torch.mean(torch.exp(torch.log(SS_deviation)/power_steps), axis=1)
        
        SS_deviation = torch.sum(torch.abs(delta), axis=1)
        SS_deviation = torch.mean(torch.exp(torch.log(SS_deviation)/power_steps), axis=1)
    
        return SS_deviation, aprox_spectral_radius
    

class BioNetSimple(BioNetBase):
    """Builds the RNN on the signaling network topology for bulk data with no categorical covariates."""

    def __init__(self, edge_list: np.array, 
                 edge_MOA: np.array, 
                 input_node_idx: torch.Tensor,
                 n_network_nodes: int, 
                 activation_function: str = 'MML', 
                 bionet_params: Optional[Dict[str, float]] = None, 
                 dtype: torch.dtype=torch.float32, 
                device: str = 'cpu', 
                seed: int = 888):
        """See BioNetBase for details."""
        super().__init__(edge_list = edge_list, edge_MOA = edge_MOA, input_node_idx = input_node_idx, 
                     n_network_nodes = n_network_nodes, activation_function = activation_function, 
                     bionet_params = bionet_params, dtype = dtype, device = device, seed = seed)
        self.make_mask()

    def initialize_weight_values(self):
        """Initialize the RNN weight_values for all interactions in the signaling network.

        Returns
        -------
        weight_values : torch.Tensor
            a torch.Tensor with randomly initialized values for each signaling network interaction
        bias : torch.Tensor
            a torch.Tensor with randomly initialized values for each signaling network node
        """
        
        network_targets = self.edge_list[0].numpy() # the target nodes receiving an edge
        n_interactions = len(network_targets)

        set_seeds(self.seed)
        weight_values = 0.1 + 0.1*torch.rand(n_interactions, dtype=self.dtype, device = self.device)
        weight_values[self.edge_MOA[1,:]] = -weight_values[self.edge_MOA[1,:]] # make those that are inhibiting negative
        
        bias_global = 1e-3*torch.ones((self.n_network_nodes_in, 1), dtype = self.dtype, device = self.device)

        for nt_idx in np.unique(network_targets):
            if torch.all(weight_values[network_targets == nt_idx]<0):
                bias_global.data[nt_idx] = 1
        self.bias_global = nn.Parameter(bias_global)

        return weight_values 

    def make_mask(self):
        """Generates a mask for adjacency matrix for non-interacting nodes, for the mechanism of action (unknown weights), 
        and ligand nodes. 

        Returns
        -------
        weights_mask : torch.Tensor
            a boolean adjacency matrix of all nodes in the signaling network, masking (True) interactions that are not present
        """

        weights_mask = torch.zeros(self.n_network_nodes, self.n_network_nodes, dtype=bool, device = self.device) # adjacency list format (targets (rows)--> sources (columns))
        weights_mask[self.edge_list] = True # if interaction is present, do not mask
        weights_mask = torch.logical_not(weights_mask) # make non-interacting edges False and vice-vesa
        self.mask = weights_mask
        
        # mask ligand bias terms -- ensures bias terms are context-specific (e.g. categorical covariates) in a 
        # manner that is independent of the ligand information
        self.bias_mask = torch.zeros((self.n_network_nodes_in, 1), dtype = bool, device = self.device)
        self.bias_mask[self.input_node_idx] = True

        # a boolean adjacency matrix of all nodes in the signaling network, with interactions that do not exist 
        # or have an unknown mechanism of action masked (True)
        self.mask_MOA = self.weights_MOA == 0

    def implement_mask(self):
        """Fill the non-interacting nodes in adj matrix and ligand inputs in bias with 0"""
        self.weights.data.masked_fill_(mask = self.mask, value = 0.0) # fill non-interacting edges with 0
        self.bias_global.data.masked_fill_(mask = self.bias_mask, value = 0.0)
        
    def forward(self, X_full: torch.Tensor, covariates_idx = None, expr = None):
        """Learn the edeg weights within the signaling network topology.

        Parameters
        ----------
        X_full : torch.Tensor
            the linearly scaled ligand inputs. Shape is (samples x network nodes). Output of ProjectInput.
        covariates_idx : None
            no value required here and it will not be used, 
            this simply serves as a placeholder to generically work with SignalingModel.forward
        Returns
        -------
        Y_full :  torch.Tensor
            the signaling network scaled by learned interaction weights. Shape is (samples x network nodes).
        """
        bias_tot = self.bias_global
        X_bias = X_full.T + bias_tot # this is the bias and ligand input combined
        X_new = torch.zeros_like(X_bias) #initialize hidden state values at 0
        
        for t in range(self.bionet_params['max_steps']): # like an RNN, updating from previous time step
            X_old = X_new
            X_new = torch.mm(self.weights, X_new) # scale matrix by edge weights
            X_new = X_new + X_bias  # add original values and bias       
            X_new = self.activation(X_new, self.bionet_params['leak'])
            
            if (t % 10 == 0) and (t > 20):
                diff = torch.max(torch.abs(X_new - X_old))    
                if diff.lt(self.bionet_params['tolerance']):
                    break

        Y_full = X_new.T
        return Y_full, None  
    
    def L1_reg_bias(self,
                    global_bias_lambda_L1: Annotated[float, Ge(0)] = 0):
        """Get the L1 regularization term for the bias parameters.
        
        Parameters
        ----------
        bias_global : 
            the global bias to be regularized
        global_bias_lambda_L1 : Annotated[float, Ge(0)]
            the regularization parameter for the global bias, by default 0 (no penalty) 
        cat_bias_lambda_L1 : Optional[Annotated[float, Ge(0)]]
            the regularizaiton parameter for the categorical bias, by default 0 (no penalty) 
        Returns
        -------
        global_bias_loss : torch.tensor
            the regularization term
        """
        # will not use biass loss since implementing KL divergence, however keep the input for consistency with other code
        # cat embeddings in the cat one are already normalized
        global_bias_loss = L1_reg(global_bias_lambda_L1, self.bias_global)

        return OrderedDict({'global_bias_L1_loss': global_bias_loss})
    
    def L2_reg(self, 
               bias_global = None, 
               weights_lambda_L2: Annotated[float, Ge(0)] = 0, 
              global_bias_lambda_L2: Annotated[float, Ge(0)] = 0, 
              cat_bias_lambda_L2: Optional[Annotated[float, Ge(0)]] = None
              ):
        """Get the L2 regularization term for the neural network parameters.
        
        Parameters
        ----------
        bias_global : 
            placeholder for `BioNetSC.L2_reg`
        weights_lambda_L2 : Annotated[float, Ge(0)]
            the regularization parameter for the weights, by default 0 (no penalty) 
        global_bias_lambda_L2 : Annotated[float, Ge(0)]
            the regularization parameter for the bias, by default 0 (no penalty) 
        cat_bias_lambda_L2 : Optional[Annotated[float, Ge(0)]]
            placeholder for `BioNetSC.Cat` and `BioNetSC.L2_reg`, by default 0 (no penalty) 
        Returns
        -------
        bionet_L2 : torch.Tensor
            the regularization term
        """
        # cat embeddings in the cat one are already normalized
        global_bias_loss = global_bias_lambda_L2 * torch.sum(torch.square(self.bias_global))
        weight_loss = weights_lambda_L2 * torch.sum(torch.square(self.weights))

#         bionet_L2 = bias_loss + weight_loss
        return OrderedDict({'weight_L2_loss': weight_loss, 'global_bias_L2_loss': global_bias_loss})
    
class BioNetCat(BioNetBase):
    """Builds the RNN on the signaling network topology, accounting for categorical covariates of the samples (e.g. cell line, genetic background, etc.)."""
    
    DEFAULT_PARAMETERS = {**BioNetBase.DEFAULT_PARAMETERS, 
                          **{'cat_max_norm': 100}}

    def __init__(self, edge_list: np.array, 
                 edge_MOA: np.array, 
                 input_node_idx: torch.Tensor,
                 n_network_nodes: int, 
                 covariates: pd.DataFrame,
                 categorical_covariate_keys: List[str],
                 activation_function: str = 'MML', 
                 bionet_params: Optional[Dict[str, float]] = None, 
                 dtype: torch.dtype=torch.float32, 
                device: str = 'cpu', 
                seed: int = 888):    
        
        # embed covariates needed for some methods called in super().__init__
        super().__init__(edge_list = edge_list, edge_MOA = edge_MOA, input_node_idx = input_node_idx, 
                         n_network_nodes = n_network_nodes, activation_function = activation_function, 
                         bionet_params = bionet_params, dtype = dtype, device = device, seed = seed)
        self.covariates = covariates[categorical_covariate_keys]
        self.embed_covariates()
        self.make_mask()


    def initialize_weight_values(self):
        """Initialize the RNN weight_values for all interactions in the signaling network. 
        
        For categorical covariates in bulk, there will be no basal bias, but rather only covariate biases. Setting the attribute to zero makes the forward method code more consistent across the classes. 

        Returns
        -------
        weight_values : torch.Tensor
            a torch.Tensor with randomly initialized values for each signaling network interaction
        bias : torch.Tensor
            a torch.Tensor with randomly initialized values for each signaling network node
        """
        
        network_targets = self.edge_list[0].numpy() # the target nodes receiving an edge
        n_interactions = len(network_targets)

        set_seeds(self.seed)
        weight_values = 0.1 + 0.1*torch.rand(n_interactions, dtype=self.dtype, device = self.device)
        weight_values[self.edge_MOA[1,:]] = -weight_values[self.edge_MOA[1,:]] # make those that are inhibiting negative
        
        self.bias_global = torch.zeros((self.n_network_nodes_in, 1), dtype = self.dtype, device = self.device, requires_grad = False) 

        return weight_values

    def make_mask(self):
        """Generates a mask for adjacency matrix for non-interacting nodes, for the mechanism of action (unknown weights), 
        and ligand nodes. 

        Returns
        -------
        weights_mask : torch.Tensor
            a boolean adjacency matrix of all nodes in the signaling network, masking (True) interactions that are not present
        """

        weights_mask = torch.zeros(self.n_network_nodes, self.n_network_nodes, dtype=bool, device = self.device) # adjacency list format (targets (rows)--> sources (columns))
        weights_mask[self.edge_list] = True # if interaction is present, do not mask
        weights_mask = torch.logical_not(weights_mask) # make non-interacting edges False and vice-vesa
        self.mask = weights_mask
        
        # mask ligand bias terms -- ensures bias terms are context-specific (e.g. categorical covariates) in a 
        # manner that is independent of the ligand information
        self.bias_mask = torch.zeros((self.n_network_nodes_in, 1), dtype = bool, device = self.device)
        self.bias_mask[self.input_node_idx] = True
        self.cat_embeddings_mask = {covariate_cat: torch.cat([self.bias_mask.T] * embedding.weight.shape[0], dim=0)
                            for covariate_cat, embedding in self.cat_embeddings.items()}

        # a boolean adjacency matrix of all nodes in the signaling network, with interactions that do not exist 
        # or have an unknown mechanism of action masked (True)
        self.mask_MOA = self.weights_MOA == 0
        
        
    def embed_covariates(self):
        """
        Creates a dictionary mapping each categorical covariate's values to a numerical value (index), corresponding
        to the rows of the embedding in `self.cat_embeddings`. 
        """
        
        # map each category's labels to a respective numerical index
        self.cat_mapper = OrderedDict({})
        self.one_hot = {}
        for cvk in self.covariates.columns:
            if self.covariates[cvk].dtype.name == 'category' and self.covariates[cvk].dtype.ordered:
                labels = self.covariates[cvk].cat.categories
            else:
                labels = sorted(set(self.covariates[cvk]))
        
            n_labels = len(labels)
            label_idx = list(range(n_labels))
            self.cat_mapper[cvk] = dict(zip(labels, label_idx)) # index
#             self.one_hot[cvk] = F.one_hot(torch.tensor(label_idx), n_labels).to(self.get_device()) # each row is the index of the cat_mapper
        
        ############################
        # necessary for forward pass: working with category group and category group label indices rather than strings:
        # map the covariates dataframe from labels to its respective index
        self.covariates_idx = self.covariates.copy()
        for cat in self.covariates_idx.columns:
            self.covariates_idx[cat] = self.covariates[cat].map(self.cat_mapper[cat])
        # store the category group orders (1st column/category in `covariates` is 0, 2nd is 1, etc)
        self._cat_group_idx = dict(enumerate(self.cat_mapper))
        ############################
        
        # embed the categorical data
        set_seeds(self.seed)
        self.cat_embeddings = nn.ModuleDict(
                        {
                            covariate_cat: nn.Embedding(num_embeddings = len(covariate_cat_map), 
                                                        embedding_dim = self.n_network_nodes_in,
                                                        max_norm = self.bionet_params['cat_max_norm'], norm_type = 2, 
                                                       device = self.device) 
                            for covariate_cat, covariate_cat_map in self.cat_mapper.items()}
                    )
    
    def covariates_to_tensor(self, sample_ids):
        """Returns the covariates by index in torch.Tensor format for a specified list of samples.""" 
        return torch.tensor(self.covariates_idx.loc[sample_ids, :].values, device = self.device, dtype = torch.int64)
   
    def implement_mask(self):
        """Fill the non-interacting nodes in adj matrix and ligand inputs in bias with 0"""
        self.weights.data.masked_fill_(mask = self.mask, value = 0.0) # fill non-interacting edges with 0
#         self.bias_global.data.masked_fill_(mask = self.bias_mask, value = 0.0)

        for idx, cat_group in enumerate(self.cat_embeddings.keys()):
            self.cat_embeddings[cat_group].weight.data.masked_fill_(mask = self.cat_embeddings_mask[cat_group], 
                                                                  value = 0.0)
    

    def forward(self, X_full: torch.Tensor, covariates_idx: torch.Tensor, expr = None):
        """Learn the edeg weights within the signaling network topology.

        Parameters
        ----------
        X_full : torch.Tensor
            the linearly scaled ligand inputs. Shape is (samples x network nodes). Output of ProjectInput.
        covariates_idx : torch.Tensor
            rows correspond to samples as in X_full. Each column represents one categorical covariate group. Values
            in the columns represent the index mapping of the category label. This should be a row-wise subset of `self.covariates_idx`, which can also be obtained from `self.covariates_to_tensor()`
        expr : None
            no value required here and it will not be used, 
            this simply serves as a placeholder to generically work with SignalingModel.forward

        Returns
        -------
        Y_full :  torch.Tensor
            the signaling network scaled by learned interaction weights. Shape is (samples x network nodes).
        """
        
        # add categorical covariates
        bias_cats = torch.zeros_like(X_full.T, device = self.device, dtype = self.dtype)
        for cat_group_idx in range(covariates_idx.shape[1]):
            cat_group = self._cat_group_idx[cat_group_idx]
            bias_cats += self.cat_embeddings[cat_group](covariates_idx[:,cat_group_idx]).T
    #             # the indexing above would be the equivalent of this:
    #             embedding = self.cat_embeddings[cat_group].weight.clone()
    #             one_hot = self.one_hot[cat_group][labels_idx].to(embedding.dtype)
    #             bias_tot += torch.matmul(one_hot, embedding)
        
        bias_tot = self.bias_global + bias_cats
        X_bias = X_full.T + bias_tot # this is the bias and ligand input combined
        X_new = torch.zeros_like(X_bias) #initialize hidden state values at 0
        
        for t in range(self.bionet_params['max_steps']): # like an RNN, updating from previous time step
            X_old = X_new
            X_new = torch.mm(self.weights, X_new) # scale matrix by edge weights
            X_new = X_new + X_bias  # add original values and bias       
            X_new = self.activation(X_new, self.bionet_params['leak'])
            
            if (t % 10 == 0) and (t > 20):
                diff = torch.max(torch.abs(X_new - X_old))    
                if diff.lt(self.bionet_params['tolerance']):
                    break

        Y_full = X_new.T
        return Y_full, None

    def L1_reg_bias(self, 
                    cat_bias_lambda_L1: Annotated[float, Ge(0)] = 0):
        """Get the L1 regularization term for the bias parameters.
        
        Parameters
        ----------
        cat_bias_lambda_L1 : Optional[Annotated[float, Ge(0)]]
            the regularizaiton parameter for the categorical bias, by default 0 (no penalty) 
        Returns
        -------
        cat_bias_loss : torch.tensor
            the regularization term
        """
        # will not use biass loss since implementing KL divergence, however keep the input for consistency with other code
        # cat embeddings in the cat one are already normalized

        cat_bias_loss = 0
        for cat_embedding in self.cat_embeddings.values():
            cat_bias_loss += torch.sum(torch.abs(cat_embedding.weight))
        cat_bias_loss *= torch.tensor(cat_bias_lambda_L1, device = self.device, dtype = self.dtype)
        
        return OrderedDict({'cat_bias_L1_loss': cat_bias_loss})

    def L2_reg(self, 
               bias_global = None, 
               weights_lambda_L2: Annotated[float, Ge(0)] = 0, 
              global_bias_lambda_L2 = None, 
              cat_bias_lambda_L2: Annotated[float, Ge(0)] = 0
              ):
        """Get the L2 regularization term for the neural network parameters.
        
        Parameters
        ----------
        bias_global : 
            placeholder for `BioNetSC.L2_reg`
        weights_lambda_L2 : Annotated[float, Ge(0)]
            the regularization parameter for the weights, by default 0 (no penalty) 
        bias_lambda_L2 : Annotated[float, Ge(0)]
            placeholder for `BioNetSimple.L2_reg` and `BioNetSC.L2_reg`
        cat_bias_lambda_L2 : Optional[Annotated[float, Ge(0)]]
            the regularizaiton parameter for the categorical bias, by default 0 (no penalty) 
        Returns
        -------
        bionet_L2 : torch.Tensor
            the regularization term
        """
        # cat embeddings in the cat one are already normalized
        # global_bias_loss = bias_lambda_L2 * torch.sum(torch.square(self.bias_global)) # this will always be 0, uneccessary computation
        weight_loss =  L2_reg(weights_lambda_L2, self.weights)

        cat_bias_loss = 0
        for cat_embedding in self.cat_embeddings.values():
            cat_bias_loss += torch.sum(torch.square(cat_embedding.weight))
        cat_bias_loss *= torch.tensor(cat_bias_lambda_L2, device = self.device, dtype = self.dtype)

#         bionet_L2 = weight_loss + cat_bias_loss
        return OrderedDict({'weight_L2_loss': weight_loss, 'cat_bias_L2_loss': cat_bias_loss})
    
class BioNetSC(BioNetCat):
    """Builds the RNN on the signaling network topology, accounting for single-cell inputs."""
    
    DEFAULT_PARAMETERS = {**BioNetCat.DEFAULT_PARAMETERS, 
                          **GaussianVariationalEncoder.DEFAULT_HYPER_PARAMS}                          

    def __init__(self, edge_list: np.array, 
                 edge_MOA: np.array, 
                 input_node_idx: torch.Tensor,
                 n_network_nodes: int,
                 n_genes: int,
                 covariates: pd.DataFrame,
                 categorical_covariate_keys: List[str],
                 activation_function: str = 'MML', 
                 bionet_params: Optional[Dict[str, float]] = None, 
                 dtype: torch.dtype=torch.float32, 
                device: str = 'cpu', 
                seed: int = 888):    
        
        # embed covariates needed for some methods called in super().__init__
        super().__init__(edge_list = edge_list, edge_MOA = edge_MOA, input_node_idx = input_node_idx, 
                         n_network_nodes = n_network_nodes, covariates = covariates, 
                         categorical_covariate_keys = categorical_covariate_keys, activation_function = activation_function, 
                         bionet_params = bionet_params, dtype = dtype, device = device, seed = seed)

        self.vae = GaussianVariationalEncoder(n_features = n_genes, 
                                   n_latent = self.n_network_nodes_in, 
                                   decode = False, 
                                   var_min = self.bionet_params['vae_var_min'],
                                   n_hidden_nodes = self.bionet_params['vae_n_hidden_nodes'],
                                   batch_momentum = self.bionet_params['vae_batch_momentum'],
                                   layer_norm = self.bionet_params['vae_layer_norm'], 
                                   dropout_rate = self.bionet_params['vae_dropout_rate'], 
                                   activation_fn = self.bionet_params['vae_activation_fn'], 
                                              device = self.device, dtype = self.dtype, 
                                              seed = seed
                                  )

    def initialize_weight_values(self):
        """Initialize the RNN weight_values for all interactions in the signaling network. 
        
        For single-cell, the basal bias will be calculated from the gene expression during the forward pass.

        Returns
        -------
        weight_values : torch.Tensor
            a torch.Tensor with randomly initialized values for each signaling network interaction
        bias : torch.Tensor
            a torch.Tensor with randomly initialized values for each signaling network node
        """
        
        network_targets = self.edge_list[0].numpy() # the target nodes receiving an edge
        n_interactions = len(network_targets)

        set_seeds(self.seed)
        weight_values = 0.1 + 0.1*torch.rand(n_interactions, dtype=self.dtype, device = self.device)
        weight_values[self.edge_MOA[1,:]] = -weight_values[self.edge_MOA[1,:]] # make those that are inhibiting negative

        return weight_values
    
    def implement_mask(self):
        """Fill the non-interacting nodes in adj matrix and ligand inputs in bias with 0"""
        self.weights.data.masked_fill_(mask = self.mask, value = 0.0) # fill non-interacting edges with 0

        for idx, cat_group in enumerate(self.cat_embeddings.keys()):
            self.cat_embeddings[cat_group].weight.data.masked_fill_(mask = self.cat_embeddings_mask[cat_group], 
                                                                  value = 0.0)
            
        # bias global is still in the forward pass because it is generated during teh forward pass

    def forward(self, X_full: torch.Tensor, covariates_idx: torch.Tensor, expr: torch.Tensor):
        """Learn the edeg weights within the signaling network topology.

        Parameters
        ----------
        X_full : torch.Tensor
            the linearly scaled ligand inputs. Shape is (samples x network nodes). Output of ProjectInput.
        covariates_idx : torch.Tensor
            rows correspond to samples as in X_full. Each column represents one categorical covariate group. Values
            in the columns represent the index mapping of the category label. Basically a rowsubset of `self.covariates_idx`.
        expr : torch.Tensor
            the expression matrix. rows correspond to samples as in X_full. Columns are genes

        Returns
        -------
        Y_full :  torch.Tensor
            the signaling network scaled by learned interaction weights. Shape is (samples x network nodes).
        bias_global : torch.Tensor
            the context-independent bias term output by the VAE. Shape is (samples x network nodes).
        """
        bias_cats = torch.zeros_like(X_full.T, device = self.device, dtype = self.dtype)
        # add categorical covariates
        for cat_group_idx in range(covariates_idx.shape[1]):
            cat_group = self._cat_group_idx[cat_group_idx]
            bias_cats += self.cat_embeddings[cat_group](covariates_idx[:,cat_group_idx]).T
    #             # the indexing above would be the equivalent of this:
    #             embedding = self.cat_embeddings[cat_group].weight.clone()
    #             one_hot = self.one_hot[cat_group][labels_idx].to(embedding.dtype)
    #             bias_tot += torch.matmul(one_hot, embedding)
        
        bias_mu, bias_log_sigma_squared, bias_global = self.vae(expr)
        bias_global.data.masked_fill_(mask = self.bias_mask.T.expand(bias_global.shape[0], -1), value = 0.0) # apply bias mask
        
        bias_tot = bias_global.T + bias_cats
        X_bias = X_full.T + bias_tot # this is the bias and ligand input combined
        X_new = torch.zeros_like(X_bias) #initialize hidden state values at 0
        
        for t in range(self.bionet_params['max_steps']): # like an RNN, updating from previous time step
            X_old = X_new
            X_new = torch.mm(self.weights, X_new) # scale matrix by edge weights
            
            X_new = X_new + X_bias  # add original values and bias       
            X_new = self.activation(X_new, self.bionet_params['leak'])
            
            if (t % 10 == 0) and (t > 20):
                diff = torch.max(torch.abs(X_new - X_old))    
                if diff.lt(self.bionet_params['tolerance']):
                    break

        Y_full = X_new.T
        return Y_full, (bias_global, bias_mu, bias_log_sigma_squared)
    
    def L1_reg_bias(self, 
               bias_global, 
              global_bias_lambda_L1: Annotated[float, Ge(0)] = 0, 
              cat_bias_lambda_L1: Annotated[float, Ge(0)] = 0):
        """Get the L1 regularization term for the bias parameters.
        
        Parameters
        ----------
        bias_global : 
            the global bias to be regularized
        global_bias_lambda_L1 : Annotated[float, Ge(0)]
            the regularization parameter for the global bias, by default 0 (no penalty) 
        cat_bias_lambda_L1 : Optional[Annotated[float, Ge(0)]]
            the regularizaiton parameter for the categorical bias, by default 0 (no penalty) 
        Returns
        -------
         : OrderedDict
            the regularization term
        """
        # will not use biass loss since implementing KL divergence, however keep the input for consistency with other code
        # cat embeddings in the cat one are already normalized
        global_bias_loss = L1_reg(global_bias_lambda_L1, bias_global)

        cat_bias_loss = 0
        for cat_embedding in self.cat_embeddings.values():
            cat_bias_loss += torch.sum(torch.abs(cat_embedding.weight))
        cat_bias_loss *= torch.tensor(cat_bias_lambda_L1, device = self.device, dtype = self.dtype)
        
        return OrderedDict({'global_bias_L1_loss': global_bias_loss, 'cat_bias_L1_loss': cat_bias_loss})

    def L2_reg(self, 
               bias_global, 
               weights_lambda_L2: Annotated[float, Ge(0)] = 0, 
              global_bias_lambda_L2: Annotated[float, Ge(0)] = 0, 
              cat_bias_lambda_L2: Annotated[float, Ge(0)] = 0):
        """Get the L2 regularization term for the neural network parameters.
        
        Parameters
        ----------
        bias_global : 
            the global bias to be regularized
        weights_lambda_L2 : Annotated[float, Ge(0)]
            the regularization parameter for the weights, by default 0 (no penalty) 
        global_bias_lambda_L2 : Annotated[float, Ge(0)]
            the regularization parameter for the global bias, by default 0 (no penalty) 
        cat_bias_lambda_L2 : Optional[Annotated[float, Ge(0)]]
            the regularizaiton parameter for the categorical bias, by default 0 (no penalty) 
        Returns
        -------
         : OrderedDict
            the regularization terms
        """
        global_bias_loss = L2_reg(global_bias_lambda_L2, bias_global)
        weight_loss = L2_reg(weights_lambda_L2, self.weights)

        cat_bias_loss = 0
        for cat_embedding in self.cat_embeddings.values():
            cat_bias_loss += torch.sum(torch.square(cat_embedding.weight))
        cat_bias_loss *= torch.tensor(cat_bias_lambda_L2, device = self.device, dtype = self.dtype)
        
        return OrderedDict({'weight_L2_loss': weight_loss, 'global_bias_L2_loss': global_bias_loss, 'cat_bias_L2_loss': cat_bias_loss})

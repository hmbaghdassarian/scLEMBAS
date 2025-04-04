"""
Constructs the full LEMBAS model.
"""
from typing import List, Dict, Union, Annotated, Optional, Any
from annotated_types import Ge
import itertools

import pandas as pd
import numpy as np
import scipy
import torch
from torch import nn

from ..utilities import set_seeds
from .model_components import ProjectInput, ProjectOutput
from .bionetwork import BioNetSimple, BioNetCat, BioNetSC

class SignalingModel(torch.nn.Module):
    """Constructs the signaling network based RNN."""
    
    def __init__(self, net: pd.DataFrame, X_in: pd.DataFrame, y_out: pd.DataFrame,
                 expr: pd.DataFrame = None,
                 covariates: Optional[pd.DataFrame] = None, categorical_covariate_keys: Optional[List[str]] = None,
                 projection_amplitude_in: Union[int, float] = 1, projection_amplitude_out: float = 1,
                 ban_list: List[str] = None, weight_label: str = 'mode_of_action', 
                 source_label: str = 'source', target_label: str = 'target',
                 bionet_params: Dict[str, float] = None, 
                 activation_function: str='MML',                 
                 dtype: torch.dtype=torch.float32, device: str = 'cpu', seed: int = 888,
                rand_y: bool = False):
        """Parse the signaling network and build the model layers.

        Parameters
        ----------
        net: pd.DataFrame
            signaling network adjacency list with the following columns:
                - `weight_label`: whether the interaction is stimulating (1) or inhibiting (-1). Exclude non-interacting (0) nodes. 
                - `source_label`: source node column name
                - `target_label`: target node column name
        X_in : pd.DataFrame
            input ligand concentrations. Index represents samples and columns represent a ligand. Values represent amount of ligand introduced (e.g., concentration). 
        y_out : pd.DataFrame
            output TF activities. Index represents samples and columns represent TFs. Values represent activity of the TF. 
        expr : pd.DataFrame, optional
            The expression matrix with index as sample IDs, columns as gene IDs, and values as expression counts
            can be optained using `tf_adata.to_df()`
        covariates : pd.DataFrame, optional
            metadata with index as sample IDs and columns containing various metadata values/mappings, by default None
            If None, will run the original LEMBAS model that does not distinguish between categorical covariates
        categorical_covariate_keys : List[str], optional
            the columns in the `covariates` representing categorical/discrete variables, by default None
        ban_list : List[str], optional
            a list of signaling network nodes to disregard, by default None
        projection_amplitude_in : Union[int, float]
            value with which to scale ligand inputs by, by default 1 (see `ProjectInput` for details, can also be tuned as a learned parameter in the model)
        projection_amplitude_out : float
             value with which to scale TF activity outputs by, by default 1 (see `ProjectOutput` for details, can also be tuned as a learned parameter in the model)
        bionet_params : Dict[str, float], optional
            hyper parameters for the model, by default None
            Key values include:
                - 'max_steps': maximum number of time steps of the RNN, by default 300
                - 'tolerance': threshold at which to break RNN; based on magnitude of change of updated edge weight values, by default 1e-5
                - 'leak': parameter to tune extent of leaking, analogous to leaky ReLU, by default 0.01
                - 'spectral_target': _description_, by default np.exp(np.log(params['tolerance'])/params['target_steps'])
                - 'exp_factor': _description_, by default 20
                - 'cat_max_norm' : passed to `max_norm` argument of nn.Embedding when generating categorical embeddings (only if covariates is not None); a value of 100 keeps the categorical embeddings close to the standard normal distribution, like the global bias
            vae hyper params. This is only used for single-cell. Keyword arguments to pass to the encoder. Keys include:
                n_hidden_nodes : List[int], optional
                    number of hidden nodes per hidden layer, by default [64]
                    each element in the list corresponds to one hidden layer (i.e., no. of hidden layers = length of list)
                batch_momentum : float, optional
                    `momentum` parameter for `BatchNorm` layer, by default .01
                    If None, a `BatchNorm` is not added
                layer_norm : bool, optional
                    whether to have `LayerNorm` layers or not, by default False
                dropout_rate : int | float, optional
                    dropout rate to apply to each of the hidden layers, by default 0.1
                    If None, dropout is not added
                activation_fn : nn.Module | None, optional
                    non-linear Pytorch activation function, by default nn.ReLU. No activation if set to None
                linear_output : bool, optional
                    whether the final layer in the encoder should have a linear activation function (True) or the specified `activation_fn` (False)
                var_eps : float, optional
                    Minimum value for the variance, by default 1e-4. Used for numerical stability
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
        rand_y : bool
            if True, won't reorder the output feature labels alphabetically as is standardly done, allows for testing against baseline random models when True, by default False
        """
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.seed = seed
        self._gradient_seed_counter = 0
        self.projection_amplitude_out = projection_amplitude_out
        
        input_combs = itertools.combinations([X_in, y_out, expr], 2)
        for comb in input_combs:
            if (comb[0].index != comb[1].index).any():
                raise ValueError('The X, y, and expr inputs do not have the same samples')
        del input_combs

        # creates self.node_idx_map
        edge_list, edge_MOA = self.parse_network(net, ban_list, weight_label, source_label, target_label)

        ## filter for nodes in the network, sorting in alphabetical order (as node_labels and node_idx_map is)
#         self.X_in = X_in.loc[:, np.intersect1d(X_in.columns.values, node_labels)]
#         self.y_out = y_out.loc[:, np.intersect1d(y_out.columns.values, node_labels)]
        # filter for nodes in the network, sorting in order of the node_idx_map
        self.X_in = X_in[sorted([col for col in self.node_idx_map if col in X_in.columns], key=self.node_idx_map.get)]
        if not rand_y: 
            self.y_out = y_out[sorted([col for col in self.node_idx_map if col in y_out.columns], key=self.node_idx_map.get)]
        else:
            raise ValueError('make sure this is functioning as expected')
            self.y_out = y_out
        self.expr = expr

        # define model layers
        self.input_layer = ProjectInput(node_idx_map = self.node_idx_map, 
                                        input_labels = self.X_in.columns.values, 
                                        projection_amplitude = projection_amplitude_in, 
                                        dtype = self.dtype, 
                                        device = self.device)
        if covariates is None:
            self.signaling_network = BioNetSimple(edge_list = edge_list, 
                                                edge_MOA = edge_MOA,
                                                input_node_idx = self.input_layer.input_node_idx,
                                                n_network_nodes = len(self.node_idx_map), 
                                                bionet_params = bionet_params, 
                                                activation_function = activation_function, 
                                                dtype = self.dtype, device = self.device, seed = self.seed)
        else:
            if self.expr is None:
                self.signaling_network = BioNetCat(edge_list = edge_list, 
                                                   edge_MOA = edge_MOA,
                                                   input_node_idx = self.input_layer.input_node_idx,
                                                   n_network_nodes = len(self.node_idx_map), 
                                                   bionet_params = bionet_params, 
                                                   activation_function = activation_function, 
                                                   covariates = covariates, 
                                                   categorical_covariate_keys = categorical_covariate_keys, 
                                                   dtype = self.dtype, device = self.device, seed = self.seed)
            else:
                self.signaling_network = BioNetSC(edge_list = edge_list, 
                                                  edge_MOA = edge_MOA,
                                                  input_node_idx = self.input_layer.input_node_idx,
                                                  n_network_nodes = len(self.node_idx_map),
                                                  n_genes = self.expr.shape[1],
                                                  bionet_params = bionet_params, 
                                                  activation_function = activation_function, 
                                                  covariates = covariates, 
                                                  categorical_covariate_keys = categorical_covariate_keys, 
                                                  dtype = self.dtype, device = self.device, seed = self.seed)
                
        self.output_layer = ProjectOutput(node_idx_map = self.node_idx_map, 
                                          output_labels = self.y_out.columns.values, 
                                          projection_amplitude = self.projection_amplitude_out, 
                                          dtype = self.dtype, device = self.device)
    def get_device(self):
        if self.device == 'cuda':
            device = next(self.parameters()).device
            return device.type + ':{}'.format(device.index)
        else:
            return self.device
       
    def parse_network(self, net: pd.DataFrame, ban_list: List[str] = None, 
                 weight_label: str = 'mode_of_action', source_label: str = 'source', target_label: str = 'target'):
        """Parse adjacency network.
    
        Parameters
        ----------
        net: pd.DataFrame
            signaling network adjacency list with the following columns:
                - `weight_label`: whether the interaction is stimulating (1) or inhibiting (-1) or unknown (0). Exclude non-interacting (0)
                nodes. 
                - `source_label`: source node column name
                - `target_label`: target node column name
        ban_list : List[str], optional
            a list of signaling network nodes to disregard, by default None
    
        Returns
        -------
        edge_list : np.array
            a (2, net.shape[0]) array where the first row represents the indices for the target node and the 
            second row represents the indices for the source node. net.shape[0] is the total # of interactions
        node_labels : list
            a list of the network nodes in the same order as the indices
        edge_MOA : np.array
            a (2, net.shape[0]) array where the first row is a boolean of whether the interactions are stimulating and the 
            second row is a boolean of whether the interactions are inhibiting. 
            
            Note: Edge_list includes interactions that are not delineated as activating OR inhibiting, s.t. edge_MOA records this 
            as [False, False].
        """
        if not ban_list:
            ban_list = []
        if sorted(net[weight_label].unique()) != [-1, 0.1, 1]:
            raise ValueError(weight_label + ' values must be 1 or -1')
        
        net = net[~ net[source_label].isin(ban_list)]
        net = net[~ net[target_label].isin(ban_list)]
    
        # create an edge list with node incides
        node_labels = sorted(pd.concat([net[source_label], net[target_label]]).unique())
        self.node_idx_map = {node_name: idx for idx, node_name in enumerate(node_labels)}
        
        source_indices = net[source_label].map(self.node_idx_map).values
        target_indices = net[target_label].map(self.node_idx_map).values

        # # get edge list
        # edge_list = np.array((target_indices, source_indices))
        # edge_MOA = net[weight_label].values
        # get edge list *ordered by source-target node index*
        n_nodes = len(node_labels)
        A = scipy.sparse.csr_matrix((net[weight_label].values, (source_indices, target_indices)), shape=(n_nodes, n_nodes)) # calculate adjacency matrix
        source_indices, target_indices, edge_MOA = scipy.sparse.find(A) # re-orders adjacency list by index
        edge_list = np.array((target_indices, source_indices)) 
        edge_MOA = np.array([[edge_MOA==1],[edge_MOA==-1]]).squeeze() # convert to boolean

        return edge_list, edge_MOA

    def df_to_tensor(self, df: pd.DataFrame):
        """Converts a pandas dataframe to the appropriate torch.tensor"""
        return torch.tensor(df.values.copy(), dtype=self.dtype, device = self.device)

    def forward(self, X_in, covariates_idx: torch.Tensor, expr: torch.Tensor):
        """Forward pass of the model.Linearly scales ligand inputs, learns weights for signaling network interactions, 
        and transforms this to TF activity. See `forward` methods of each layer for details.

        Parameters
        ----------
        X_in : torch.tensor
            input ligand values 
        covariates_idx : torch.Tensor
            rows correspond to samples as in X_full. Each column represents one categorical covariate group. Values
            in the columns represent the index mapping of the category label. This should be a row-wise subset of `signaling_network.covariates_idx`, which can also be obtained from `signaling_network.covariates_to_tensor()`
            only relevant for categorical data, otherwise None
        biases : tuple
            tuple of bias_global, bias_mu, bias_log_sigma_squared
            only relevant for single-cell, otherwise None
        """
        X_full = self.input_layer(X_in) # input ligands to signaling network
        Y_full, biases = self.signaling_network(X_full, covariates_idx, expr) # RNN of full signaling network
        Y_hat = self.output_layer(Y_full)

        return Y_hat, Y_full, biases

    def L2_reg(self,
               input_lambda_L2: Annotated[float, Ge(0)] = 0, 
              bn_weights_lambda_L2: Annotated[float, Ge(0)] = 0, 
              global_bias_lambda_L2: Optional[Annotated[float, Ge(0)]] = None, 
              cat_bias_lambda_L2: Optional[Annotated[float, Ge(0)]] = None,
               bias_global = None,
              output_weights_lambda_L2: Annotated[float, Ge(0)] = 0, 
              output_bias_lambda_L2: Annotated[float, Ge(0)] = 0):
        """Get the L2 regularization term for the neural network parameters.
        
        Parameters
        ----------
        input_lambda_L2 : Annotated[float, Ge(0)]
            the regularization parameter for the ProjectInput layer weights, by default 0 (no penalty)  
        bn_weights_lambda_l2 : Annotated[float, Ge(0)]
            the regularization parameter for the bionetwork layer weights, by default 0 (no penalty) 
        global_bias_lambda_L2 : Annotated[float, Ge(0)]
            tthe regularization parameter for the bionetwork layer bias (global if separated into global and categoriacl), by default 0 (no penalty)
        bias_global : 
            the global bias vector, only to be used with BioNetSC as it is not a stored parameter
        output_weights_lambda_L2 : Annotated[float, Ge(0)]
            the regularization parameter for the ProjectOutput layer weights, by default 0 (no penalty) 
        output_bias_lambda_L2 : Annotated[float, Ge(0)]
            the regularization parameter for the ProjectOutput layer bias, by default 0 (no penalty) 

        Returns
        -------
         : torch.Tensor
            the regularization term (as the sum of the regularization terms for each layer)
        """
        input_loss = self.input_layer.L2_reg(input_lambda_L2) 
        sn_loss = self.signaling_network.L2_reg(bias_global = bias_global, 
                                                weights_lambda_L2 = bn_weights_lambda_L2, 
                                                global_bias_lambda_L2 = global_bias_lambda_L2, 
                                                cat_bias_lambda_L2 = cat_bias_lambda_L2) 
        output_loss = self.output_layer.L2_reg(weights_lambda_L2 = output_weights_lambda_L2, 
                                               bias_lambda_L2 = output_bias_lambda_L2)
        return input_loss, sn_loss, output_loss

    def ligand_regularization(self, lambda_L2: Annotated[float, Ge(0)] = 0):
        """DEPRECATED: now setting bias term to 0 and masking directly during forward pass        

        Get the L2 regularization term for the ligand biases. Intuitively, extracellular ligands should not contribute to 
        "baseline activity" affecting intracellular signaling nodes.
        
        Parameters
        ----------
        lambda_L2 : Annotated[float, Ge(0)]
            the regularization parameter, by default 0 (no penalty) 
        
        Returns
        -------
        loss : torch.Tensor
            the regularization term
        """
        raise ValueError('This is deprecated since bias terms are now masked')
        loss = lambda_L2 * torch.sum(torch.square(self.signaling_network.bias_basal[self.input_layer.input_node_idx]))
        return loss

    def uniform_regularization(self, lambda_L2: torch.tensor, Y_full: torch.Tensor, 
                     target_min: float = 0.0, target_max: float = None):
        """Get the L2 regularization term for deviations of the nodes in Y_full from that of a uniform distribution between 
        `target_min` and `target_max`. 
        Note, this penalizes both deviations from the uniform distribution AND values that are out of range (like a double penalty).
    
        Parameters
        ----------
        lambda_L2 : torch.tensor
            scaling factor for state loss
        Y_full : torch.Tensor
            the signaling network scaled by learned interaction weights. Shape is (samples x network nodes). 
            Output of BioNet.
        target_min : float, optional
            minimum values for nodes in Y_full to take on, by default 0.0
        target_max : float, optional
            maximum values for nodes in Y_full to take on, by default 1/`self.projection_amplitude_out`
    
        Returns
        -------
        loss : torch.Tensor
            the regularization term
        """
        if not target_max:
            target_max = 1/self.projection_amplitude_out
        
        sorted_Y_full, _ = torch.sort(Y_full, axis=0) # sorts each column (signaling network node) in ascending order
        target_distribution = torch.linspace(target_min, target_max, Y_full.shape[0], dtype=Y_full.dtype, device=Y_full.device).reshape(-1, 1)
        
        dist_loss = torch.sum(torch.square(sorted_Y_full - target_distribution)) # difference in distribution
        below_loss = torch.sum(Y_full.lt(target_min) * torch.square(Y_full-target_min)) # those that are below the minimum value
        above_loss = torch.sum(Y_full.gt(target_max) * torch.square(Y_full-target_max)) # those that are above the maximum value
        loss = lambda_L2*(dist_loss + below_loss + above_loss)
        return loss

    def add_gradient_noise(self, noise_level: Union[float, int]):
        """Adds noise to backwards pass gradient calculations. Use during training to make model more robust. 
    
        Parameters
        ----------
        noise_level : Union[float, int]
            scaling factor for amount of noise to add 
        """
        all_params = list(self.parameters())
        if self.seed:
            set_seeds(self.seed + self._gradient_seed_counter)
        for i in range(len(all_params)):
            if all_params[i].requires_grad:
                all_noise = torch.randn(all_params[i].grad.shape, dtype=all_params[i].dtype, device=all_params[i].device)
                all_params[i].grad += (noise_level * all_noise)
    
        self._gradient_seed_counter += 1 # new random noise each time function is called

#     def copy(self):
#         return copy.deepcopy(self)
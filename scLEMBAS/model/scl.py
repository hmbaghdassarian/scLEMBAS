"""
Constructs the full LEMBAS model.
"""
from typing import List, Dict, Union, Annotated
from annotated_types import Ge

import pandas as pd
import numpy as np
import scipy
import torch
from torch import nn

from .model_utilities import update_with_defaults
from .bionetwork import ProjectInput, BioNet, ProjectOutput
from .tfa import TFA

class SignalingModel(torch.nn.Module):
    """Constructs the signaling network based RNN."""
    DEFAULT_TRAINING_PARAMETERS = {'target_steps': 100, 'max_steps': 300, 'exp_factor': 20, 'leak': 0.01, 'tolerance': 1e-5}
    
    def __init__(self, net: pd.DataFrame, X_in: pd.DataFrame, y_out: pd.DataFrame,
                 projection_amplitude_in: Union[int, float] = 1, projection_amplitude_out: float = 1,
                 ban_list: List[str] = None, weight_label: str = 'mode_of_action', 
                 source_label: str = 'source', target_label: str = 'target', 
                bionet_params: Dict[str, float] = None , 
                 activation_function: str='MML', dtype: torch.dtype=torch.float32, device: str = 'cpu', seed: int = 888):
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
        ban_list : List[str], optional
            a list of signaling network nodes to disregard, by default None
        projection_amplitude_in : Union[int, float]
            value with which to scale ligand inputs by, by default 1 (see `ProjectInput` for details, can also be tuned as a learned parameter in the model)
        projection_amplitude_out : float
             value with which to scale TF activity outputs by, by default 1 (see `ProjectOutput` for details, can also be tuned as a learned parameter in the model)
        bionet_params : Dict[str, float], optional
            training parameters for the model, by default None
            Key values include:
                - 'max_steps': maximum number of time steps of the RNN, by default 300
                - 'tolerance': threshold at which to break RNN; based on magnitude of change of updated edge weight values, by default 1e-5
                - 'leak': parameter to tune extent of leaking, analogous to leaky ReLU, by default 0.01
                - 'spectral_target': _description_, by default np.exp(np.log(params['tolerance'])/params['target_steps'])
                - 'exp_factor': _description_, by default 20
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
        self.dtype = dtype
        self.device = device
        self.seed = seed
        self._gradient_seed_counter = 0
        self.projection_amplitude_out = projection_amplitude_out

        edge_list, node_labels, edge_MOA = self.parse_network(net, ban_list, weight_label, source_label, target_label)
        if not bionet_params:
            bionet_params = self.DEFAULT_TRAINING_PARAMETERS.copy()
        else:
            bionet_params = self.set_training_parameters(**bionet_params)

        # filter for nodes in the network, sorting by node_labels order
        self.X_in = X_in.loc[:, np.intersect1d(X_in.columns.values, node_labels)]
        self.y_out = y_out.loc[:, np.intersect1d(y_out.columns.values, node_labels)]

        # define model layers
        self.input_layer = ProjectInput(node_idx_map = self.node_idx_map, 
                                        input_labels = self.X_in.columns.values, 
                                        projection_amplitude = projection_amplitude_in, 
                                        dtype = self.dtype, 
                                        device = self.device)
        self.signaling_network = BioNet(edge_list = edge_list, 
                                        edge_MOA = edge_MOA, 
                                        n_network_nodes = len(node_labels), 
                                        bionet_params = bionet_params, 
                                        activation_function = activation_function, 
                                        dtype = self.dtype, device = self.device, seed = self.seed)
        self.output_layer = ProjectOutput(node_idx_map = self.node_idx_map, 
                                          output_labels = self.y_out.columns.values, 
                                          projection_amplitude = self.projection_amplitude_out, 
                                          dtype = self.dtype, device = device)

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
        self.node_idx_map = {idx: node_name for node_name, idx in enumerate(node_labels)}
        
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

        return edge_list, node_labels, edge_MOA

    def df_to_tensor(self, df: pd.DataFrame):
        """Converts a pandas dataframe to the appropriate torch.tensor"""
        return torch.tensor(df.values.copy(), dtype=self.dtype, device = self.device)

    def set_training_parameters(self, **attributes):
        """Set the parameters for training the model. Overrides default parameters with attributes if specified.
        Adapted from LEMBAS `trainingParameters`
    
        Parameters
        ----------
        attributes : dict
            keys are parameter names and values are parameter value
        """
        # #set defaults
        # default_parameters = self.DEFAULT_TRAINING_PARAMETERS.copy()
        # allowed_params = list(default_parameters.keys()) + ['spectral_target']
    
        # params = {**default_parameters, **attributes}
        # if 'spectral_target' not in params.keys():
        #     params['spectral_target'] = np.exp(np.log(params['tolerance'])/params['target_steps'])
    
        # params = {k: v for k,v in params.items() if k in allowed_params}

        params = update_with_defaults(default_params = self.DEFAULT_TRAINING_PARAMETERS, 
                                      user_parameters = attributes, 
                                      additional_parameters = ['spectral_target'])
        if 'spectral_target' not in params.keys():
            params['spectral_target'] = np.exp(np.log(params['tolerance'])/params['target_steps'])
    
        return params

    def forward(self, X_in):
        """Linearly scales ligand inputs, learns weights for signaling network interactions, and transforms this to TF activity. See
        `forward` methods of each layer for details."""
        X_full = self.input_layer(X_in) # input ligands to signaling network
        Y_full = self.signaling_network(X_full) # RNN of full signaling network
        Y_hat = self.output_layer(Y_full) # TF outputs of signaling network
        return Y_hat, Y_full

    def L2_reg(self, lambda_L2: Annotated[float, Ge(0)] = 0):
        """Get the L2 regularization term for the neural network parameters.
        
        Parameters
        ----------
        lambda_L2 : Annotated[float, Ge(0)]
            the regularization parameter, by default 0 (no penalty) 
        
        Returns
        -------
         : torch.Tensor
            the regularization term (as the sum of the regularization terms for each layer)
        """
        return self.input_layer.L2_reg(lambda_L2) + self.signaling_network.L2_reg(lambda_L2) + self.output_layer.L2_reg(lambda_L2)

    def ligand_regularization(self, lambda_L2: Annotated[float, Ge(0)] = 0):
        """Get the L2 regularization term for the ligand biases. Intuitively, extracellular ligands should not contribute to 
        "baseline (i.e., unstimulated) activity" affecting intrecllular signaling nodes and thus TF outputs.
        
        Parameters
        ----------
        lambda_L2 : Annotated[float, Ge(0)]
            the regularization parameter, by default 0 (no penalty) 
        
        Returns
        -------
        loss : torch.Tensor
            the regularization term
        """
        loss = lambda_L2 * torch.sum(torch.square(self.signaling_network.bias[self.input_layer.input_node_order]))
        return loss

    def uniform_regularization(self, lambda_L2: float, Y_full: torch.Tensor, 
                     target_min: float = 0.0, target_max: float = None):
        """Get the L2 regularization term for deviations of the nodes in Y_full from that of a uniform distribution between 
        `target_min` and `target_max`. 
        Note, this penalizes both deviations from the uniform distribution AND values that are out of range (like a double penalty).
    
        Parameters
        ----------
        lambda_L2 : float
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
        lambda_L2 = torch.tensor(lambda_L2, dtype = Y_full.dtype, device = Y_full.device)
        # loss = lambda_L2 * expected_uniform_distribution(Y_full, target_max = 1/self.projectionAmplitude)
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

    def copy(self):
        return copy.deepcopy(self)
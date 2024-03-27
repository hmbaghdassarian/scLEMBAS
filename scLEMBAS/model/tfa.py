"""
Construct the TF activity autoencoder.
"""

from collections import OrderedDict
from typing import List, Any, Dict, Union, Literal
import warnings

import pandas as pd
import torch
from torch import nn

from ..utilities import set_seeds
from .model_utilities import update_with_defaults

class Encoder(nn.Module):
    """
    Project input TF activity to a latent space.
    Adapted from scVI's `FCLayers`.
    """

    DEFAULT_HYPER_PARAMS = {'n_latent': 32, 'n_hidden_layers': 1, 'n_hidden_nodes': [64], 
                            'batch_momentum': 0.01, 'layer_norm': False, 'dropout_rate': 0.1,
                            'activation_fn': nn.ReLU # can make as None to have purely linear
                           }
    
    def __init__(self, n_features: int, 
                 n_latent: int = 32, 
                 n_hidden_layers: int = 1,
                 n_hidden_nodes: List[int] | int = [64],
                 batch_norm: bool = True, 
                 batch_momentum: float = 0.01,
                 layer_norm: bool = False,
                 dropout_rate: int | float = 0.1,
                 activation_fn: nn.Module | None = nn.ReLU):
        """Initialize encoder.

        Parameters
        ----------
        n_features : int
            the full number of features input to the encoder
        n_latent : int, optional
            dimension (no. of features) of the latent space, by default 32
        n_hidden_layers : int, optional
            the number of fully-connected hidden layers, by default 1
        n_hidden_nodes : int, optional
            number of hidden nodes per layer, by default 64
            if n_hidden_layers > 1, can specify a list of hidden nodes corresponding to number of nodes per layer
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
        """
        super().__init__()

        # set up params
        self.batch_momentum = batch_momentum
        self.layer_norm = layer_norm
        self.dropout_rate = dropout_rate
        self.activation_fn = activation_fn

        if self.batch_momentum and self.layer_norm:
            warnings.warn('You have applied both a batch- and layer-normalization. Recommended to choose one of the two.')
        
        # set up the layer sizes
        if isinstance(n_hidden_nodes, list):
            if len(n_hidden_nodes) != n_hidden_layers:
                raise ValueError('The number of elements in n_hidden_nodes must match the number of hidden layers.')
        else:
            n_hidden_nodes = [n_hidden_nodes]*n_hidden_layers

        self.layers_dim = [n_features] + n_hidden_nodes + [n_latent]

        self.encoder = nn.Sequential(
           OrderedDict(
               [('Encoder Layer {}'.format(i), self._single_layer(n_in,n_out)) 
                for i, (n_in, n_out) in enumerate(zip(self.layers_dim[:-1], self.layers_dim[1:]))]
               )
           ) 

    def _single_layer(self, n_in, n_out):
        """Creates a single [set of] layer[s] in the encoder."""

        all_layers = OrderedDict([('linear', nn.Linear(in_features = n_in, out_features = n_out, bias = True)),
                                  ('batch normalization', nn.BatchNorm1d(n_out, momentum=self.batch_momentum) if self.batch_momentum else None),
                                  ('layer normalization', nn.LayerNorm(n_out, elementwise_affine=False) if self.layer_norm else None),
                                  ('activation', self.activation_fn() if self.activation_fn else None),
                                  ('dropout', nn.Dropout(p=self.dropout_rate) if (self.dropout_rate and self.dropout_rate > 0) else None)])
        return nn.Sequential(OrderedDict((k, v) for k, v in all_layers.items() if v is not None))

    def forward(self, x):
        return self.encoder(x)

class TFA(nn.Module):
    """Decompose TF activity into basal effects and covariate-specific effects. 
    
    Adapted from Compositional Perturbation Autoencoder (https://doi.org/10.15252/msb.202211517)."""

    DEFAULT_HYPER_PARAMS = {'n_latent': 32, 'cat_max_norm': 1, 'recon_loss': 'gauss'}

    def __init__(self, covariates: pd.DataFrame, 
                 categorical_covariate_keys: List[str],
                 n_features_in: int,
                 n_latent: int = 32, 
                 cat_max_norm: int | float | None = 1, 
                 recon_loss : Literal['gauss', 'nb'] = 'gauss',
                encoder_hyper_params: Dict[str, Any] = Encoder.DEFAULT_HYPER_PARAMS, 
                device: str = 'cpu'):
        """Initializes model layers.
    
        Parameters
        ----------
        covariates : pd.DataFrame
            metadata with index as sample ids and columns containing various metadata values/mappings
        categorical_covariate_keys : List[str]
            the columns in the dataframe representing categorical/discrete variables
        n_features_in : int
            the number of input features to the autoencoder (either # of TFs or # of nodes in network)
        n_latent: int, optional
            dimension (no. of featuers) of the latent space, by default 32
        cat_max_norm : int | float | None, optional
            passed to `max_norm` argument of nn.Embedding when generating categorical covariate embeddings, by default 1
        recon_loss : Literal['gauss', 'nb'], optional
            Autoencoder loss (either "gauss" or "nb")
            Currently can only handle "guass"
        encoder_hyper_params : Dict[str, Any]
            Keyword arguments to pass to `Encoder`. Keys include:
                n_hidden_layers : int, optional
                    the number of fully-connected hidden layers, by default 1
                n_hidden_nodes : int, optional
                    number of hidden nodes per layer, by default 64
                    if n_hidden_layers > 1, can specify a list of hidden nodes corresponding to number of nodes per layer
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
        device : str
            whether to use gpu ("cuda") or cpu ("cpu"), by default "cpu"
        """
        super().__init__()

        self.device = device

        # encoder
        encoder_hyper_params = update_with_defaults(default_parameters=Encoder.DEFAULT_HYPER_PARAMS, 
                                                    user_parameters = encoder_hyper_params)
        encoder_hyper_params['n_latent'] = n_latent
        self.encoder = Encoder(n_features = n_features_in, **encoder_hyper_params)

        # categorical embeddings
        # create an embedding for each discrete covariate
        self.map_cat_covariates(covariates, categorical_covariate_keys)
        self.cat_embeddings = nn.ModuleDict(
            {
                covariate_cat: nn.Embedding(num_embeddings = len(covariate_cat_map), embedding_dim = n_latent, 
                                           max_norm = cat_max_norm, norm_type = 2) 
                for covariate_cat, covariate_cat_map in self.cat_mapper.items()}
        )
    
        if recon_loss == 'gauss':
            pass
        else:
            raise ValueError('Currently, TPA can only handle a guassian loss.')

    def map_cat_covariates(self, covariates: pd.DataFrame, categorical_covariate_keys: List[str]):
        """Creates a dictionary mapping each categorical covariate's values to a numerical value (index). 
    
        Parameters
        ----------
        covariates : pd.DataFrame
            metadata with index as sample ids and columns containing various metadata values/mappings
        categorical_covariate_keys : List[str]
            the columns in the dataframe representing categorical/discrete variables
        """

        self.cat_mapper = {}
        for cvk in categorical_covariate_keys:
            if covariates[cvk].dtype.name == 'category' and covariates[cvk].dtype.ordered:
                labels = covariates[cvk].cat.categories
            else:
                labels = sorted(set(covariates[cvk]))
            
            self.cat_mapper[cvk] = {k: idx for idx, k in enumerate(labels)}

    def forward(self, x):
        z_basal = self.encoder(x)
        return z_basal
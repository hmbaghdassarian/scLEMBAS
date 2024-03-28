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
    Project input TF activity to a latent space (encoder) or vice-versa (decoder).
    Adapted from scVI's `FCLayers`.
    """

    DEFAULT_HYPER_PARAMS = {'n_hidden_nodes': [64],
                            'batch_momentum': 0.01, 'layer_norm': False, 'dropout_rate': 0.1,
                            'activation_fn': nn.ReLU # can make as None to have purely linear
                            }
    
    def __init__(self, n_features: int, n_latent: int,
                 decode: bool = False,
                 n_hidden_layers: int = 1,
                 n_hidden_nodes: List[int] = [64],
                 batch_norm: bool = True, 
                 batch_momentum: float = 0.01,
                 layer_norm: bool = False,
                 dropout_rate: int | float = 0.1,
                 activation_fn: nn.Module | None = nn.ReLU,
                ):
        """Initialize encoder.

        Parameters
        ----------
        n_features : int
            the full number of features input to the encoder
        n_latent : int, optional
            dimension (no. of features) of the latent space, by default 32
        decode : bool, optional
            whether to encode into latent space (False) or decode into full feature space (True)
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
        """
        super().__init__()

        # set up params
        self.batch_momentum = batch_momentum
        self.layer_norm = layer_norm
        self.dropout_rate = dropout_rate
        self.activation_fn = activation_fn

        if self.batch_momentum and self.layer_norm:
            warnings.warn('You have applied both a batch- and layer-normalization. Recommended to choose one of the two.')
        if n_hidden_nodes != sorted(n_hidden_nodes)[::-1]:
            warnings.warn('You have specified an encoder/decoder layer that does not have a consistent change in # of nodes per hidden layer')
        
        self.layers_dim = [n_features] + n_hidden_nodes + [n_latent]
        if decode: 
            self.layers_dim = self.layers_dim[::-1]

        enc_name = 'Encoder' if not decode else 'Decoder'
        self.encoder = nn.Sequential(
           OrderedDict(
               [(enc_name + ' Layer {}'.format(i), self._single_layer(n_in,n_out)) 
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

    DEFAULT_HYPER_PARAMS = {'n_latent': 32, 'cat_max_norm': 1}

    def __init__(self, covariates: pd.DataFrame, 
                 categorical_covariate_keys: List[str],
                 n_features_in: int,
                 n_latent: int = 32, 
                 cat_max_norm: int | float | None = 1, 
                encoder_hyper_params: Dict[str, Any] = Encoder.DEFAULT_HYPER_PARAMS, 
                 decoder_hyper_params: Dict[str, Any] = Encoder.DEFAULT_HYPER_PARAMS, 
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
        encoder_hyper_params : Dict[str, Any]
            Keyword arguments to pass to `Encoder`. Keys include:
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
        decoder_hyper_params : Dict[str, Any]
            same as `encoder_hyper_params`, but projects back from latent space to full feature space
            note, layer order is reversed so must list `n_hidden_nodes` as you would in encoder (from larger to bigger)
        device : str
            whether to use gpu ("cuda") or cpu ("cpu"), by default "cpu"
        """
        super().__init__()

        self.device = device

        # encoder
        encoder_hyper_params = update_with_defaults(default_parameters=Encoder.DEFAULT_HYPER_PARAMS, 
                                                    user_parameters = encoder_hyper_params)
        self.encoder = Encoder(n_features = n_features_in, n_latent = n_latent, **encoder_hyper_params)

        # decoder
        decoder_hyper_params = update_with_defaults(default_parameters=Encoder.DEFAULT_HYPER_PARAMS, 
                                                    user_parameters = decoder_hyper_params)
        self.decoder = Encoder(n_features = n_features_in, n_latent = n_latent, decode = True, **decoder_hyper_params)

        # categorical embeddings
        # create an embedding for each discrete covariate
        self.map_cat_covariates(covariates, categorical_covariate_keys)
        self.cat_embeddings = nn.ModuleDict(
            {
                covariate_cat: nn.Embedding(num_embeddings = len(covariate_cat_map), embedding_dim = n_latent, 
                                           max_norm = cat_max_norm, norm_type = 2) 
                for covariate_cat, covariate_cat_map in self.cat_mapper.items()}
        )

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
        z_full = self.decoder(z_basal)
        return z_basal, z_full
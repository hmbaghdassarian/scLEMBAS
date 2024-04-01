"""
Construct the TF activity autoencoder (TFA).
"""

from collections import OrderedDict
from typing import List, Any, Dict, Union, Literal, Optional
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
                            'activation_fn': nn.ReLU, # can make as None to have purely linear
                            'linear_output': True
                            }
    
    def __init__(self, n_features: int, n_latent: int,
                 decode: bool = False,
                 n_hidden_nodes: List[int] = [64],
                 batch_momentum: float = 0.01,
                 layer_norm: bool = False,
                 dropout_rate: int | float = 0.1,
                 activation_fn: nn.Module | None = nn.ReLU,
                 linear_output : bool = True,
                 dtype: torch.dtype=torch.float32,
                 device: str = 'cpu', 
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
            note, these are the layers between the input layer and the bottleneck (i.e., does not include the bottleneck)
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
            whether the final layer in the encoder should only be linear (True) or incorporate the specified `activation_fn` (False)
        device : str
            whether to use gpu ("cuda") or cpu ("cpu"), by default "cpu"
        dtype : torch.dtype, optional
            datatype to store values in torch, by default torch.float32
        """
        super().__init__()

        # set up params
        self.batch_momentum = batch_momentum
        self.layer_norm = layer_norm
        self.dropout_rate = dropout_rate
        self.activation_fn = activation_fn
        self.dtype = dtype
        self.device = device

        if self.batch_momentum and self.layer_norm:
            warnings.warn('You have applied both a batch- and layer-normalization. Recommended to choose one of the two.')
        if n_hidden_nodes != sorted(n_hidden_nodes)[::-1]:
            warnings.warn('You have specified an encoder/decoder layer that does not have a consistent change in # of nodes per hidden layer')
        
        self.layers_dim = [n_features] + n_hidden_nodes + [n_latent]
        if decode: 
            self.layers_dim = self.layers_dim[::-1]

        enc_name = 'Encoder' if not decode else 'Decoder'
        all_layers = OrderedDict()
        for i, (n_in, n_out) in enumerate(zip(self.layers_dim[:-1], self.layers_dim[1:])):
            if i != len(n_hidden_nodes):
                all_layers[enc_name + ' Layer {}'.format(i)] = self._single_layer(n_in,n_out, drop_keys = None)
            else:
                drop_keys = ['batch normalization', 'layer normalization', 'dropout']
                if linear_output:
                    drop_keys += ['activation']
                all_layers[enc_name + ' Layer {}'.format(i)] = self._single_layer(n_in,n_out, drop_keys = drop_keys)
        self.encoder = nn.Sequential(all_layers)

    def _single_layer(self, n_in: int, n_out: int, drop_keys: Optional[List[str]] = None):
        """Creates a single [set of] layer[s] in the encoder."""

        all_layers = OrderedDict([('linear', nn.Linear(in_features = n_in, out_features = n_out, bias = True, 
                                                       device = self.device, dtype = self.dtype)),
                                  ('batch normalization', nn.BatchNorm1d(n_out, momentum=self.batch_momentum, 
                                                                         device = self.device, dtype = self.dtype) if self.batch_momentum else None),
                                  ('layer normalization', nn.LayerNorm(n_out, elementwise_affine=False, 
                                                                       device = self.device, dtype = self.dtype) if self.layer_norm else None),
                                  ('activation', self.activation_fn() if self.activation_fn else None),
                                  ('dropout', nn.Dropout(p=self.dropout_rate) if (self.dropout_rate and self.dropout_rate > 0) else None)])
        if drop_keys:
            for dk in drop_keys:
                if dk in all_layers:
                    del all_layers[dk]
        return nn.Sequential(OrderedDict((k, v) for k, v in all_layers.items() if v is not None))

    def forward(self, x):
        return self.encoder(x)


def _identity(x):
    return x

class VariationalEncoder(nn.Module):
    """
    Generative (Gaussian) projection from input TF activity to a latent space (encoder) or vice-versa (decoder).
    Adapted from scVI's `Encoder`.
    """

    DEFAULT_HYPER_PARAMS = {'var_min': 1e-4, 
                            'n_hidden_nodes': [64],
                            'batch_momentum': 0.01, 'layer_norm': False, 'dropout_rate': 0.1,
                            'activation_fn': nn.ReLU, # can make as None to have purely linear
                            'linear_output': True,}
    
    def __init__(self, n_features: int, n_latent: int,
                 var_min: float = 1e-4,
                 decode: bool = False,
                 n_hidden_nodes: List[int] = [64],
                 batch_norm: bool = True, 
                 batch_momentum: float = 0.01,
                 layer_norm: bool = False,
                 dropout_rate: int | float = 0.1,
                 activation_fn: nn.Module | None = nn.ReLU,
                 linear_output : bool = True,
                 dtype: torch.dtype=torch.float32,
                 device: str = 'cpu', 
                ):
        """Initialize variational decoder.

        Parameters
        ----------
        n_features : int
            the full number of features input to the encoder
        n_latent : int, optional
            dimension (no. of features) of the latent space, by default 32
        var_min : float, optional
            Minimum value for the variance, by default 1e-4. Used for numerical stability
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
        dtype : torch.dtype, optional
            datatype to store values in torch, by default torch.float32
        device : str, optional
            whether to use gpu ("cuda") or cpu ("cpu"), by default "cpu"
        """
        super().__init__()
        
        self.z_transformation = _identity # z has gaussian distribution
        self.var_activation = torch.exp # ensure positivity of variance
        self.var_min = var_min
    
        n_hidden_final = n_hidden_nodes[0]
        n_hidden_nodes = n_hidden_nodes[1:]
        self.decoder = Encoder(n_features = n_hidden_final, n_latent = n_latent, 
                               decode = decode, linear_output = linear_output,
                               n_hidden_nodes = n_hidden_nodes, batch_norm = batch_norm, 
                              batch_momentum = batch_momentum, layer_norm = layer_norm, dropout_rate = dropout_rate, 
                              activation_fn = activation_fn, dtype = dtype, device = device)
        self.mean_decoder = nn.Linear(n_hidden_final, n_features, device = device)
        self.var_decoder = nn.Linear(n_hidden_final, n_features, device = device, dtype = dtype)
    
    def forward(self, x):
        z = self.decoder(x)
        z_m = self.mean_decoder(z)
        z_v = self.var_activation(self.var_decoder(z)) + self.var_min # log-var

        dist = torch.distributions.Normal(z_m, z_v.sqrt())
        latent = self.z_transformation(dist.rsample())

        return z_m, z_v, latent


class TFA(nn.Module):
    """Decompose TF activity into basal effects and covariate-specific effects. 
    
    Adapted from Compositional Perturbation Autoencoder (https://doi.org/10.15252/msb.202211517)."""

    DEFAULT_HYPER_PARAMS = {'n_latent': 32, 'cat_max_norm': 1, 'generative_decoder': True, 'recon_loss': 'gauss'}

    def __init__(self, covariates: pd.DataFrame, 
                 categorical_covariate_keys: List[str],
                 n_features_in: int,
                 n_latent: int = 32, 
                 cat_max_norm: int | float | None = 1, 
                 generative_decoder : bool = True,
                 recon_loss : Literal['gauss', 'nb'] = 'gauss',
                encoder_hyper_params: Dict[str, Any] = Encoder.DEFAULT_HYPER_PARAMS, 
                 decoder_hyper_params: Dict[str, Any] = Encoder.DEFAULT_HYPER_PARAMS, 
                 dtype: torch.dtype=torch.float32,
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
        generative_decoder : bool, optional
            whether to make the decoder layer variational/generative (True) or not (False)
        recon_loss : Literal['gauss', 'nb'], optional
            Autoencoder loss (either "gauss" or "nb"), by default 'gauss'
            Currently can only handle "guass"
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
            Additional key words when using the generative/variational decoder:
                var_eps : float, optional
                    Minimum value for the variance, by default 1e-4. Used for numerical stability
        dtype : torch.dtype, optional
            datatype to store values in torch, by default torch.float32
        device : str
            whether to use gpu ("cuda") or cpu ("cpu"), by default "cpu"
        """
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.generative_decoder = generative_decoder
        self.recon_loss = recon_loss

        # encoder
        encoder_hyper_params = update_with_defaults(default_parameters=Encoder.DEFAULT_HYPER_PARAMS, 
                                                    user_parameters = encoder_hyper_params)
        self.encoder = Encoder(n_features = n_features_in, n_latent = n_latent, 
                               device = self.device, dtype = self.dtype,
                               **encoder_hyper_params)

        # decoder
        if not self.generative_decoder: 
            decoder_hyper_params = update_with_defaults(default_parameters=Encoder.DEFAULT_HYPER_PARAMS,
                                                        user_parameters = decoder_hyper_params)
            self.decoder = Encoder(n_features = n_features_in, n_latent = n_latent, decode = True, 
                                   device = self.device, dtype = self.dtype,
                                   **decoder_hyper_params)
        else:
            if self.recon_loss == 'gauss':
                decoder_hyper_params = update_with_defaults(default_parameters=VariationalEncoder.DEFAULT_HYPER_PARAMS,
                                                            user_parameters = decoder_hyper_params)
                self.decoder = VariationalDecoder(n_features = n_features_in, n_latent = n_latent, 
                                                  decode = True, device = self.device, dtype = self.dtype,
                                                  **decoder_hyper_params)
            else:
                raise ValueError('Currently, TPA can only handle a guassian loss in its generative process.')

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
        if not self.generative_decoder:
            z_full = self.decoder(z_basal)
            px_mean, px_var,= None, None
        else: 
            px_mean, px_var, z_full = self.decoder(z_basal)
        return z_basal, z_full, px_mean, px_var
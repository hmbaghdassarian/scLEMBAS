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

class FCLayers(nn.Module):
    """
    Generates standard, fully-connected neural-network.
    Adapted from scVI's `FCLayers`.
    """
    DEFAULT_HYPER_PARAMS = {'batch_momentum': 0.01, 'layer_norm': False, 'dropout_rate': 0.1,
                        'activation_fn': nn.ReLU, # can make as None to have purely linear
                        }
    def __init__(self, layers: List[int],
                 batch_momentum: float = 0.01,
                 layer_norm: bool = False,
                 dropout_rate: int | float = 0.1,
                 activation_fn: nn.Module | None = nn.ReLU,
                 dtype: torch.dtype=torch.float32,
                 device: str = 'cpu', 
                ):
        """Initialize encoder.

        Parameters
        ----------
        layers : List[int]
            the size of each layer (including inputs and outputs)
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

        fc_layers = OrderedDict()
        for i, (n_in, n_out) in enumerate(zip(layers[:-1], layers[1:])):
                fc_layers['FC Layer {}'.format(i)] = self._single_layer(n_in,n_out)#, drop_keys = None)
        self.fc_layers = nn.Sequential(fc_layers)

    def _single_layer(self, n_in: int, n_out: int):#, drop_keys: Optional[List[str]] = None):
        """Creates a single [set of] layer[s] in the encoder."""

        all_layers = OrderedDict([('linear', nn.Linear(in_features = n_in, out_features = n_out, bias = True, 
                                                       device = self.device, dtype = self.dtype)),
                                  ('batch normalization', nn.BatchNorm1d(n_out, momentum=self.batch_momentum, 
                                                                         device = self.device, dtype = self.dtype) if self.batch_momentum else None),
                                  ('layer normalization', nn.LayerNorm(n_out, elementwise_affine=False, 
                                                                       device = self.device, dtype = self.dtype) if self.layer_norm else None),
                                  ('activation', self.activation_fn() if self.activation_fn else None),
                                  ('dropout', nn.Dropout(p=self.dropout_rate) if (self.dropout_rate and self.dropout_rate > 0) else None)])

        return nn.Sequential(OrderedDict((k, v) for k, v in all_layers.items() if v is not None))

    def forward(self, x):
        return self.fc_layers(x)

ENCODER_HYPER_PARAMS = {**FCLayers.DEFAULT_HYPER_PARAMS, **{'n_hidden_nodes': [64]}}
class Encoder(nn.Module):
    """
    Project input TF activity to a latent space (encoder) or vice-versa (decoder).
    Adapted from CPA's `VanillaEncoder` (https://github.com/theislab/cpa/blob/main/cpa/_utils.py).
    """

    DEFAULT_HYPER_PARAMS = {**ENCODER_HYPER_PARAMS, **{'linear_output': True}}
    
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
        self.dtype = dtype
        self.device = device

        if n_hidden_nodes != sorted(n_hidden_nodes)[::-1]:
            warnings.warn('You have specified an encoder/decoder layer that does not have a consistent change in # of nodes per hidden layer')
        
        if decode:
            layers_dim = [n_latent] + n_hidden_nodes[::-1]
            n_encode_in, n_encode_out = n_hidden_nodes[0], n_features
            enc_name = 'Decoder'
        else:
            layers_dim = [n_features] + n_hidden_nodes
            n_encode_in, n_encode_out = n_hidden_nodes[-1], n_latent
            enc_name = 'Encoder'

        self.hidden_layers = FCLayers(layers = layers_dim,
                                 batch_momentum = batch_momentum, 
                                 layer_norm = layer_norm, 
                                 dropout_rate = dropout_rate, 
                                 activation_fn = activation_fn, 
                                 dtype = dtype, device = device)
        self.z = FCLayers(layers = [n_encode_in, n_encode_out],
                          batch_momentum = None, layer_norm = None, dropout_rate = None, 
                          activation_fn = activation_fn if not linear_output else None,
                          device = device, dtype = dtype)

    def forward(self, x):
        return self.z(self.hidden_layers(x))


# def _identity(x):
#     return x
    
class GaussianVariationalEncoder(nn.Module):
    """
    Generative (Gaussian) projection from input TF activity to a latent space (encoder) or vice-versa (decoder).
    Adapted from scVI's `Encoder`.
    """

    DEFAULT_HYPER_PARAMS = {**ENCODER_HYPER_PARAMS, **{'var_min': 1e-4}}
    
    def __init__(self, n_features: int, n_latent: int,
                 var_min: float = 1e-4,
                 decode: bool = False,
                 n_hidden_nodes: List[int] = [64],
                 batch_momentum: float = 0.01,
                 layer_norm: bool = False,
                 dropout_rate: int | float = 0.1,
                 activation_fn: nn.Module | None = nn.ReLU,
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

        self.dtype = dtype
        self.device = device
        
        # self.z_transformation = _identity # z has gaussian distribution
        # self.var_activation = torch.exp # ensure positivity of variance
        self.var_min = var_min
    
        if n_hidden_nodes != sorted(n_hidden_nodes)[::-1]:
            warnings.warn('You have specified an encoder/decoder layer that does not have a consistent change in # of nodes per hidden layer')
        
        if decode:
            layers_dim = [n_latent] + n_hidden_nodes[::-1]
            n_encode_in, n_encode_out = n_hidden_nodes[0], n_features
        else:
            layers_dim = [n_features] + n_hidden_nodes
            n_encode_in, n_encode_out = n_hidden_nodes[-1], n_latent

        self.hidden_layers = FCLayers(layers = layers_dim,
                                      batch_momentum = batch_momentum,
                                      layer_norm = layer_norm,
                                      dropout_rate = dropout_rate,
                                      activation_fn = activation_fn,
                                      dtype = self.dtype, device = self.device)
        self.z_mean = nn.Linear(n_encode_in, n_encode_out, device = self.device, dtype = self.dtype)
        self.z_log_var = nn.Linear(n_encode_in, n_encode_out, device = self.device, dtype = self.dtype) # log of the standard deviation
    
    def forward(self, x):
        """Calculates the latent space distribution and samples from it.
    
        Returns
        -------
        z_m : torch.Tensor
            the mean (mu) parameter for each feature in latent space
        z_v : torch.Tensor
            the log-variance parameter for each feature in latent space
        z : torch.Tensor
            the sample drawn from N(z_m, z_v)
        """
        h = self.hidden_layers(x)
        z_mu = self.z_mean(h)
        z_sigma = torch.exp(self.z_log_var(h)/2.) + self.var_min # log-var trick

        # # scVI sampling
        # dist = torch.distributions.Normal(z_m, z_v.sqrt())
        # z = self.z_transformation(dist.rsample())
        
        # reparameterization 
        epsilon = torch.randn_like(z_m, dtype = self.dtype, device = self.device)
        z = z_m + z_sigma*epsilon #self.z_transformation(z_m + z_sigma*epsilon)

        return z_m, z_v, z

dist_encoders = {'vanilla': Encoder, 'gauss': GaussianVariationalEncoder}
class TFA(nn.Module):
    """Decompose TF activity into basal effects and covariate-specific effects. 
    
    Adapted from Compositional Perturbation Autoencoder (https://doi.org/10.15252/msb.202211517)."""

    DEFAULT_HYPER_PARAMS = {'n_latent': 32, 'cat_max_norm': 1, 'encoder_dist': None, 'decoder_dist': None}

    def __init__(self, covariates: pd.DataFrame, 
                 categorical_covariate_keys: List[str],
                 n_features_in: int,
                 n_latent: int = 32, 
                 cat_max_norm: int | float | None = 1, 
                 encoder_dist: Optional[Literal['gauss', 'nb']] = None,
                 decoder_dist: Optional[Literal['gauss', 'nb']] = None,
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
        encoder_dist : Optional[Literal['gauss', 'nb'], optional
            the latent space distribution to model in the encoder block, by default None
            if None, uses a vanilla encoder that directly projects to the latent space
        decoder_dist : Optional[Literal['gauss', 'nb'], optional
            same as `encoder_dist` (i.e., a stochastic decoder), by default None
        encoder_hyper_params : Dict[str, Any]
            Keyword arguments to pass to `Encoder` or `VariationalEncoder`. Keys include:
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
            Keywords specific to `Encoder`: 
                linear_output : bool, optional
                    whether the final layer in the encoder should only be linear (True) or incorporate the specified `activation_fn` (False)
            Keywords specific to `GaussianVariationalEncoder`: 
                var_min : float, optional
                    Minimum value for the variance, by default 1e-4. Used for numerical stability
        decoder_hyper_params : Dict[str, Any]
            same as `encoder_hyper_params`
        dtype : torch.dtype, optional
            datatype to store values in torch, by default torch.float32
        device : str
            whether to use gpu ("cuda") or cpu ("cpu"), by default "cpu"
        """
        super().__init__()

        self.device = device
        self.dtype = dtype

        if encoder_dist == 'nb' or decoder_dist == 'nb':
            raise ValueError('Negative binomial distributions are not currently implemented')
        self.encoder_dist = 'vanilla' if not encoder_dist else encoder_dist
        self.decoder_dist = 'vanilla' if not decoder_dist else decoder_dist

        tfa_encoder = dist_encoders[self.encoder_dist]
        tfa_decoder = dist_encoders[self.decoder_dist]

        # encoder
        encoder_hyper_params = update_with_defaults(default_parameters=tfa_encoder.DEFAULT_HYPER_PARAMS, 
                                                    user_parameters = encoder_hyper_params)
        self.encoder = tfa_encoder(n_features = n_features_in, n_latent = n_latent, 
                               device = self.device, dtype = self.dtype,
                               **encoder_hyper_params)

        # decoder
        decoder_hyper_params = update_with_defaults(default_parameters=tfa_decoder.DEFAULT_HYPER_PARAMS, 
                                                    user_parameters = encoder_hyper_params)
        self.decoder = tfa_decoder(n_features = n_features_in, n_latent = n_latent, 
                               device = self.device, dtype = self.dtype,
                               **decoder_hyper_params)

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
        if self.encoder_dist == 'vanilla':
            z_m, z_v, z_basal = None, None, self.encoder(x)
        elif self.encoder_dist == 'gauss':
            z_m, z_v, z_basal = self.encoder(x)

        if self.decoder_dist == 'vanilla':
            x_m, x_v, x_ = None, None, self.encoder(z)
        elif self.decoder_dist == 'gauss':
            x_m, x_v, x_ = self.encoder(z)

        return z_m, z_v, z_basal
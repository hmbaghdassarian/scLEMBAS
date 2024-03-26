class Encoder(nn.Module):
    """Build a fully connected encoder that projects input TF activity to a latent space.
    
    Adapted from scVI's `FCLayers`."""

    DEFAULT_HYPER_PARAMS = {'n_latent': 64, 'batch_norm': True, 'layer_norm': False, 'dropout_rate': 0.1,
                            'activation_fn': nn.ReLU # can make as None to have purely linear
               }
    
    def __init__(self, n_features: int, 
                 n_latent: int = 64, 
                 batch_norm: bool = True, 
                 layer_norm: bool = False,
                 dropout_rate: int | float = 0.1,
                 activation_fn: nn.Module | None = nn.ReLU):
        """Initialize encoder.

        Parameters
        ----------
        n_features : int
            the full number of features input to the encoder
        n_latent : int, optional
            dimension (no. of features) of the latent space, by default 64
        batch_norm : bool, optional
            whether to have `BatchNorm` layers or not, by default True
        layer_norm : bool, optional
            whether to have `LayerNorm` layers or not, by default False
        dropout_rate : int | float, optional
            dropout rate to apply to each of the hidden layers, by default 0.1
        activation_fn : nn.Module | None, optional
            non-linear Pytorch activation function, by default nn.ReLU. No activation if set to None
        """
        super().__init__()
        n_in, n_out = n_features, n_latent
        self.encoder = nn.Sequential(
            nn.Linear(in_features = n_in, out_features = n_out, bias = True),
            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001) if batch_norm else None,
            nn.LayerNorm(n_out, elementwise_affine=False) if layer_norm else None,
            activation_fn() if activation_fn else None,
            nn.Dropout(p=dropout_rate) if (dropout_rate and dropout_rate > 0) else None
        )

    def forward(self, x):
        return self.encoder(x)

import pandas as pd
from typing import List, Any, Dict, Union, Literal
import torch.nn as nn

class TFA(nn.Module):
    """Decompose TF activity into basal effects and covariate-specific effects. 
    
    Adapted from Compositional Perturbation Autoencoder (https://doi.org/10.15252/msb.202211517)."""
    def __init__(self, cat_covariates: pd.DataFrame, 
                 categorical_covariate_keys: List[str],
                 n_latent: int = 64, 
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
        n_latent: int, optional
            dimension (no. of featuers) of the latent space, by default 64
        cat_max_norm : int | float | None, optional
            passed to `max_norm` argument of nn.Embedding when generating categorical covariate embeddings, by default 1
        recon_loss : Literal['gauss', 'nb'], optional
            Autoencoder loss (either "gauss" or "nb")
            Currently can only handle "guass"
        encoder_hyper_params : Dict[str, Any]
            Key word arguments to pass to `Encoder`. Keys include:
                batch_norm : bool, optional
                    whether to have `BatchNorm` layers or not, by default True
                layer_norm : bool, optional
                    whether to have `LayerNorm` layers or not, by default False
                dropout_rate : int | float, optional
                    dropout rate to apply to each of the hidden layers, by default 0.1
                activation_fn : nn.Module | None,optional
                    non-linear Pytorch activation function, by default nn.ReLU. No activation if set to None
        device : str
            whether to use gpu ("cuda") or cpu ("cpu"), by default "cpu"
        """
        super().__init__()

        self.device = device

        # encoder
        #n_features = ??
        encoder_hyper_params['n_latent'] = n_latent
        self.encoder = Encoder(n_features = n_features, **encoder_hyper_params)

        # categorical embeddings
        # create an embedding for each discrete covariate
        self.map_cat_covariates(cat_covariates, categorical_covariate_keys)
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
        cat_covariates : pd.DataFrame
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
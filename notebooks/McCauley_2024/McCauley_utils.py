import os
import json
from typing import List, Literal

import numpy as np
import pandas as pd

import scanpy as sc

import torch
import torch.nn as nn

from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA



import sys
from pathlib import Path
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))
from notebook_utils import get_split

sclembas = '/home/hmbaghda/Projects/scLEMBAS'
sys.path.insert(1, os.path.join(sclembas))
from scLEMBAS import io
from scLEMBAS import preprocess as pp 
from scLEMBAS.model.train import TrainSC
from scLEMBAS.model.scl import SignalingModel

data_path = '/home/hmbaghda/orcd/pool/scLEMBAS/analysis'
author = 'McCauley'

def generate_lr_params(n_epochs, 
                       max_lr, 
                       lr_scaling_factor=10, 
                       lr_decay=0.75,
                       n_restarts=4,
                       n_adversarial_start=0,
                       role='scl'):
    """
    Generate LR scheduler params for WarmupCosineAnnealingWarmRestarts
    that ensures discriminator and generator follow the same curve in real (epoch) time.

    For 'generator', it updates when either discriminator is active.

    Returns:
        Dict of scheduler parameters
    """

    total_active_epochs = n_epochs
    n_steps = n_epochs  # default for SCL

    if role in ['generator', 'cat_discriminator', 'pert_discriminator']:
        if n_adversarial_start >= n_epochs: # won't be used anyways
            n_adversarial_start = 0
        total_active_epochs = n_epochs - n_adversarial_start
        n_steps = total_active_epochs // 1

    T_0 = max(1, n_steps // n_restarts)
    warmup_epochs = max(1, n_steps // 10)

    if warmup_epochs >= T_0:
        warmup_epochs = 0

    return {
        'max_epochs': n_epochs,
        'maximum_learning_rate': max_lr,
        'minimum_learning_rate': max_lr / lr_scaling_factor,
        'lr_restart_epoch': T_0,
        'n_optimizer_resets': 0,
        'lr_decay': lr_decay,
        'lr_restart_factor': 1,
        'warmup_epochs': warmup_epochs
    }

def load_data():
    sn_ppis = pd.read_csv(os.path.join(data_path, 'processed', author + '_sn_ppis.csv'), 
                     index_col = 0)

    tf_adata = io.read_tfad(os.path.join(data_path, 'processed', author + '_consensus_tf_activity.h5ad'))

    adata = sc.read_h5ad(os.path.join(data_path, 'processed', author + '_normalized_counts.h5ad'))
    adata = adata[tf_adata.obs_names, adata.var['highly_variable']].copy() # filter for HVGs and filtered perts
    
    
    expr = adata.to_df().copy()

    source_label = 'source_genesymbol'
    target_label = 'target_genesymbol'
    weight_label = 'mode_of_action'
    stimulation_label = 'consensus_stimulation'
    inhibition_label = 'consensus_inhibition'

    cat_col = 'cell_type'
    pert_col = 'ligand'
    ctrl_pert = 'CTRL'
    
    # basic formatting checks
    if not np.all(tf_adata.var_names == sorted(tf_adata.var_names)):
        raise ValueError('Ensure TF adata features are sorted on input')

    if not np.all(adata.obs_names == tf_adata.obs_names):
        raise ValueError('Ensure gene expression and TF activity sample features are orderd the same')

    if len(set(tf_adata.obs[pert_col])) != len(tf_adata.obs[pert_col].cat.categories):
        raise ValueError('Make sure only present perturbations are in the categorical columns')

    if len(set(tf_adata.obs[cat_col])) != len(tf_adata.obs[cat_col].cat.categories):
        raise ValueError('Make sure only present cell lines are in the categorical columns')
    
    all_data = (sn_ppis, tf_adata, adata, expr, source_label, target_label, weight_label, stimulation_label, inhibition_label, cat_col, pert_col, ctrl_pert)
    
    return all_data


all_data = load_data()
sn_ppis, tf_adata, adata, expr, source_label, target_label, weight_label, stimulation_label, inhibition_label, cat_col, pert_col, ctrl_pert = all_data



def initialize_mod_and_trainer(
    fold: int, 
    adversarial_penalty: bool = True, 
    randomize: bool = False, 
    num_stochastic_edges: int = None,
    bn_weights_lambda_L2: float = 1e-7,
    bn_weights_lambda_L1: float = 0,
    seed: int = 888):
    """_summary_

    Parameters
    ----------
    fold : int
        which previously generated split to use
    adversarial_penalty : bool, optional
        whether to penalize the generator by the discriminator performance, by default True
        if False, will not use the discriminators, and generator should not have perturbation and categorical information removed
    randomize : bool, optional
        whether to permute the features, by default False
        if True, creates a random baseline model
    num_stochastic_edges : int
        whether to add stochastic edges to the network
    bn_weights_lambda_L2 : float, optional
        L2 regularization strength for the signaling network weights, by default 1e-7
        only alter for exploring self pruning behavior
    bn_weights_lambda_L1 : float, optional
        same as L2 but for L1 regularization
    seed : int, optional
        the seed to use for initialization and training, by default 888
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------- BATCHES -------------------------------------
    
    # batches
    split = get_split(fold = fold, author = author)
    
    n_train_cells = len(split['train_barcodes'])
    n_test_cells = len(split['test_barcodes'])
    
    n_batches = 15
    batch_params = {
        'train_batch_size': int(np.round(n_train_cells/n_batches)), 
        'test_batch_size': int(np.round(n_test_cells/n_batches)), 
        'validation_batch_size': np.nan, 
        'drop_last_batch': True
    }
    
    # ------------------------------------- PRESETS -------------------------------------
    
    # direct into init but used in other parts
    projection_amplitude_out = 1
    
    # lr presets
    max_epochs = 600
    n_adversarial_start = 200
    lr_scaling_factor = 10
    lr_decay = 0.9
    n_restarts = 4
    n_restarts_adversarial = 4
    
    # ------------------------------------- BULK LEMBAS -------------------------------------
    bionet_params = {'target_steps': 100, 
                     'max_steps': 120, 
                     'exp_factor': 50, 
                     'tolerance': 1e-5, 
                     'leak': 1e-2}

    spectral_radius_params = {'n_probes_spectral': 5, 
                              'power_steps_spectral': 5, 
                              'subset_n_spectral': 5, 
                             'track_spectral_radius': True}
    target_spectral_radius = 0.9

    noise_params = {
        'network_noise_scale': 0.01, 
        'min_network_noise': 0.0025, 
        'gradient_noise_scale': 1e-9, 
        'include_gradient_noise_vae': True, 
        'include_gradient_noise_embedding': True, 
        'constant_gradient_noise': True

    }
    
    # RNN LR
    prediction_loss_fn = torch.nn.MSELoss(reduction='mean')
    lr_params = generate_lr_params(n_epochs = max_epochs, 
                                   max_lr = 2e-3, 
                                   lr_scaling_factor = lr_scaling_factor, 
                                   lr_decay = lr_decay,
                                   n_restarts = n_restarts,
                                   n_adversarial_start = np.nan, 
                                   role = 'scl')
    
    # ------------------------------------- REGULARIZERS -------------------------------------
    
    bionet_params['cat_max_norm'] = 100
    regularization_params = {
        'input_lambda_L2': 0, # irrelevant because setting the requires grad to False

        'bn_weights_lambda_L2': bn_weights_lambda_L2, #1e-7,
        'bn_weights_lambda_L1': bn_weights_lambda_L1, #0,
        'moa_lambda_L1': 1e2,
        'uniform_lambda_L2': 0, #1e-7, 
        'uniform_min': -1/projection_amplitude_out,
        'uniform_max': 1/projection_amplitude_out,
        'adj_scaling_KL': 0,  # using uniform/bn_weights already
        'adj_prior_mu': 0, # irrelevant because adj_scaling_KL is 0
        'adj_prior_sigma': 0.2, # irrelevant because adj_scaling_KL is 0

        'output_weights_lambda_L2': 1e-7,
        'output_bias_lambda_L2': 1e-7,

        'spectral_loss_factor': 0,


        'global_bias_lambda_L2': 0, # using KL divergence instead
        'global_bias_lambda_L1': 0, # using KL divergence instead
        'cat_bias_lambda_L2': 5e-4,  # allow for generalization (not collapsing on perturbation)
        'cat_bias_lambda_L1': 0, # using cat max norm
    }



    contrastive_loss_params = {
        'methods': [], 
        'lambda_scalers': [], 
        'understimate_only': np.nan, # only for _bulk_actual
        'min_percentile': np.nan, # only for _sc
        'triplet_margin_frac': np.nan, # for sc only
    }


    cat_pert_params = {
        'regularization_scaler': 100, 
        'method': 'orthogonality', 
        'per_label': False, 
        'include_adjacency': False, 
        'temperature': 0.1
                          }

    # aggregate
    training_params = {
        **lr_params, 
        **batch_params, 
        **regularization_params, 
        **spectral_radius_params,
        **noise_params
    }

    training_params['prediction_loss_fn_scaler'] = 100
    
    # ------------------------------------- VAE -------------------------------------
    
    # building
    n_layers_vae = 3
    n_nodes = len(set(sn_ppis[source_label].tolist() + sn_ppis[target_label].tolist()))
    vae_n_hidden_nodes = list(np.round(np.linspace(adata.shape[1], n_nodes, n_layers_vae + 2)).astype(int)[1:-1])

    vae_mod_params = {
        'vae_batch_momentum': 0.01, 
        'vae_layer_norm': False, 
        'vae_dropout_rate': 0.7,
        'vae_activation_fn': nn.LeakyReLU,
        'vae_n_hidden_nodes': vae_n_hidden_nodes, 
        'vae_var_min': 1e-4

    } 
    bionet_params = {**bionet_params, **vae_mod_params}

    # training
    vae_params = {
        'prior_mu': 0, 
        'prior_sigma': 1,
        'scaling_KL': 5e-3,  
        'lambda_l2': 1e-7, 
        'optimizer': torch.optim.Adam
    }


    vae_lr_params = generate_lr_params(n_epochs = max_epochs,
                                       max_lr = 5e-4, #max_lr,
                                       lr_scaling_factor = lr_scaling_factor, 
                                       lr_decay = lr_decay,
                                       n_restarts = n_restarts_adversarial,
                                       n_adversarial_start = n_adversarial_start,
                                       role = 'generator')

    vae_params = {**vae_params, **vae_lr_params}
    del vae_params['max_epochs']
    
    # ------------------------------------- DISCRIMINATORS -------------------------------------
    
    discriminator_params = {
        'batch_momentum': None,
        'layer_norm': False,
        'spectral_norm': np.nan, #False,
        'dropout_rate': np.nan,
        'activation_fn': nn.LeakyReLU,
        'n_hidden_nodes': np.nan,
        'lr_restart_factor': 1,
        'optimizer': torch.optim.Adam,
        'discriminator_lambda_L2': 1e-3,
        'discriminator_penalty_weight': np.nan, 
        'bionet_activation': False,
        'smooth_labels': True, 
        'epsilon_smooth': np.nan
    }

    
    cat_n_layers_disc = 3
    cat_disc_n_hidden_nodes = list(np.round(np.linspace(n_nodes, 
                                                        tf_adata.obs[cat_col].nunique(),
                                                        cat_n_layers_disc + 2)).astype(int)[1:-1])
    cat_discriminator_params = discriminator_params.copy()
    cat_discriminator_params['n_hidden_nodes'] = cat_disc_n_hidden_nodes
    cat_discriminator_params['dropout_rate'] = 0.1

    cat_discriminator_params['spectral_norm'] = True
#     if cat_spectral_norm:
    cat_discriminator_params['discriminator_lambda_L2'] = 0


    pert_n_layers_disc = 3
    pert_disc_n_hidden_nodes = list(np.round(np.linspace(n_nodes, 
                                                        tf_adata.obs[pert_col].nunique(),
                                                        pert_n_layers_disc + 2)).astype(int)[1:-1])

    # add 3 additional "starting" layers since classifying perturbation is difficult
    pert_disc_n_hidden_nodes = [pert_disc_n_hidden_nodes[0]]*3 + pert_disc_n_hidden_nodes

    pert_discriminator_params = discriminator_params.copy()
    pert_discriminator_params['n_hidden_nodes'] = pert_disc_n_hidden_nodes
    pert_discriminator_params['dropout_rate'] = 0.3

    pert_discriminator_params['spectral_norm'] = True
#     if pert_spectral_norm:
    pert_discriminator_params['discriminator_lambda_L2'] = 0

    cat_discriminator_params['epsilon_smooth'] = 0.1 #min(0.1, 1/tf_adata.obs[cat_col].nunique())
    pert_discriminator_params['epsilon_smooth'] = 0.1 #min(0.1, 1/tf_adata.obs[pert_col].nunique())
    
    # Adversarial 
    if adversarial_penalty:
        cat_discriminator_params['discriminator_penalty_weight'] = pp.discriminator_weight_curve(
            n_epochs = max_epochs - n_adversarial_start,
            min_penalty_weight = 0.001,
            max_penalty_weight = 11,
            a = 1,
            b = 2.5, 
            curve_type = 'power')

        pert_discriminator_params['discriminator_penalty_weight'] = pp.discriminator_weight_curve(
            n_epochs = max_epochs - n_adversarial_start,
            min_penalty_weight = 0.001,
            max_penalty_weight = 10,
            a = 1,
            b = 2.75, 
            curve_type = 'power')
        
        n_cat_discriminator_train = 10
        n_pert_discriminator_train = 10
    else: # no adversarial training, handled internally by the trainer
        cat_discriminator_params['discriminator_penalty_weight'] = [0]*(max_epochs - n_adversarial_start)
        pert_discriminator_params['discriminator_penalty_weight'] = [0]*(max_epochs - n_adversarial_start)
        
        n_cat_discriminator_train = 0
        n_pert_discriminator_train = 0
        
    # LRs
    discriminator_lr_params = generate_lr_params(
        n_epochs = max_epochs,
        max_lr = 1e-3,
        lr_scaling_factor = lr_scaling_factor, 
        lr_decay = lr_decay,
        n_restarts = n_restarts_adversarial,
        n_adversarial_start = n_adversarial_start, 
        role = 'cat_discriminator')
    del discriminator_lr_params['max_epochs']

    # perturbation
    discriminator_lr_params = generate_lr_params(
        n_epochs = max_epochs,
        max_lr = 1e-3,
        lr_scaling_factor = lr_scaling_factor, 
        lr_decay = lr_decay,
        n_restarts = n_restarts_adversarial,
        n_adversarial_start = n_adversarial_start, 
        role = 'pert_discriminator')
    del discriminator_lr_params['max_epochs']

    
    # Aggregate
    cat_discriminator_params = {**cat_discriminator_params, **discriminator_lr_params}
    pert_discriminator_params = {**pert_discriminator_params, **discriminator_lr_params}
    
    
    # ------------------------------------- INITIALIZATION -------------------------------------
    
    # input stimulation
    X_in = pd.get_dummies(tf_adata.obs[pert_col]).astype(int)
    X_in.drop(columns = ctrl_pert, inplace = True) # all 0s
    
    y_out = tf_adata.to_df().copy()
    if randomize:
        np.random.seed(seed)
        permuted_tfs = np.random.permutation(y_out.columns)
        y_out = y_out[permuted_tfs].copy()

    
    mod = SignalingModel(
        net = sn_ppis,
        X_in = X_in,
        y_out = y_out, 
        rand_y_features = randomize,
        expr = expr, 
        covariates = tf_adata.obs.copy(),
        categorical_covariate_keys = [cat_col],
        projection_amplitude_in = 10, 
        projection_amplitude_out = projection_amplitude_out,
        weight_label = weight_label, source_label = source_label, target_label = target_label,
        bionet_params = bionet_params, 
        _num_self_prune_edges = num_stochastic_edges,
        dtype = torch.float32, device = device, seed = seed)

    mod.input_layer.weights.requires_grad = False # don't learn scaling factors for the ligand input concentrations
    mod.signaling_network.prescale_weights(target_radius = target_spectral_radius) # spectral radius
    
    
    trainer = TrainSC(
        mod = mod,
        prediction_optimizer = torch.optim.Adam,
        prediction_loss_fn = prediction_loss_fn, 
        per_condition_loss = True,
        n_adversarial_start = n_adversarial_start, 
        n_cat_discriminator_train = n_cat_discriminator_train,
        n_pert_discriminator_train = n_pert_discriminator_train,
        gradient_ascent = True,
        cat_discriminator_params = cat_discriminator_params,
        pert_discriminator_params = pert_discriminator_params,
        vae_params = vae_params,
        hyper_params = training_params,
        contrastive_loss_params = contrastive_loss_params,
        cat_pert_params = cat_pert_params,
        train_split = {'train': split['train_barcodes'], 'test': split['test_barcodes'], 'validation': None}, 
        train_seed = seed, 
        n_track_test = 20,
        n_track_validation = None, 
        n_eval_cells = np.nan, 
        n_eval_bootstrap = np.nan
    )
    
    return mod, trainer



reduction_type_map = {'pca': 'pc', 
                      'pls': 'pls', 
                      'umap': 'umap', 
                      'umap_pls': 'umap (pls)'}
def adata_dimviz(
        adata, 
        reduction_type: Literal['pca', 'pls', 'umap', 'umap_pls'], 
        cats: List[str], 
        subset_size: int = int(1e4), 
        seed: int = 888):
    """Formats for visualization of embedding

    Parameters
    ----------
    adata : _type_
        AnnData object
    reduction_type : Literal['pca', 'pls', 'umap', 'umap_pls']
        embedding to visualize
    cats : List[str]
        columns in adata.obs to retain
    subset_size : int, optional
        proportionally subsets across all cats, by default int(1e4)
    seed : int, optional
        random state, by default 888
    """


    reduction_type_ = reduction_type_map[reduction_type]

    if type(cats) != list:
        cats = [cats]

    viz_df = pd.DataFrame(adata.obsm['X_' + reduction_type])
    viz_df = pd.concat([viz_df, pd.DataFrame(adata.obs[cats]).reset_index(drop = True)], ignore_index = True, axis = 1)
    viz_df.columns = [reduction_type_.upper() + str(i+1) for i in range(viz_df.shape[1])][:-len(cats)] + cats


#     nmi = normalized_mutual_info_score(adata.obs.leiden, adata.obs[cat])

    if subset_size is not None and subset_size < viz_df.shape[0]:
        grouped = viz_df.groupby(cats, observed=False)
        cell_prop = viz_df[cats].value_counts(normalize = True)
        index_to_keep = []
        np.random.seed(seed)
        for cat_type, cat_df in grouped:
            mask = (viz_df[cats].values == cat_type)
            if len(cats) > 1:
                mask = mask.all(axis=1)
            all_barcodes = viz_df[mask].index
            
            cat_subset_size = np.round(max(1, np.round(cell_prop.loc[cat_type]*subset_size)))
            cat_subset_size = min(cat_subset_size, len(all_barcodes))
            cat_subset_size = int(cat_subset_size)

            
            index_to_keep += np.random.choice(all_barcodes, 
                         size = cat_subset_size,
                         replace = False).tolist()

        viz_df = viz_df.loc[index_to_keep, :]

    # shuffle
    viz_df = viz_df.sample(frac=1, random_state = seed).reset_index(drop=True)
    
    return viz_df



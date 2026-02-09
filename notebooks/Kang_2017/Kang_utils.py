"""Common functions used across multiple scripts"""

import os
import json
from typing import List, Literal

import numpy as np
import pandas as pd

import scanpy as sc

import torch
import torch.nn as nn

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
author = 'Kang'

rev_stim = {'STIM': 'CTRL', 'CTRL': 'STIM'}
stim_map = {'STIM': 1, 'CTRL': 0}
rev_stim_map = {v:k for k,v in stim_map.items()}

def load_data():
    """Drops the lowest 3 cell types"""

    cat_col = 'seurat_annotations'
    pert_col = 'stim'

    tf_adata = io.read_tfad(os.path.join(data_path, 'processed', '{}_tf_activity.h5ad').format(author))
    tf_adata.obs['condition'] = tf_adata.obs[cat_col].astype(str) + '^' + tf_adata.obs[pert_col].astype(str)
    tf_adata = tf_adata[:, sorted(tf_adata.var_names)] 

    adata = sc.read_h5ad(os.path.join(data_path, 'processed', '{}_expr_scored.h5ad').format(author))
    adata = adata[tf_adata.obs_names, adata.var['highly_variable']].copy() # filter for HVGs and filtered perts
    expr = adata.to_df().copy()

    sn_ppis = pd.read_csv(os.path.join(data_path, 'processed', '{}_sn_ppis.csv').format(author), index_col = 0)
    source_label = 'source_genesymbol'
    target_label = 'target_genesymbol'
    weight_label = 'mode_of_action'
    stimulation_label = 'consensus_stimulation'
    inhibition_label = 'consensus_inhibition'

    # basic formatting checks
    if not np.all(tf_adata.var_names == sorted(tf_adata.var_names)):
        raise ValueError('Ensure TF adata features are sorted on input')

    if not np.all(adata.obs_names == tf_adata.obs_names):
        raise ValueError('Ensure gene expression and TF activity sample features are orderd the same')

    if len(set(tf_adata.obs[pert_col])) != len(tf_adata.obs[pert_col].cat.categories):
        raise ValueError('Make sure only present perturbations are in the categorical columns')

    if len(set(tf_adata.obs[cat_col])) != len(tf_adata.obs[cat_col].cat.categories):
        raise ValueError('Make sure only present cell lines are in the categorical columns')

    all_data = (sn_ppis, tf_adata, adata, expr, source_label, target_label, weight_label, stimulation_label, inhibition_label, cat_col, pert_col)

    return all_data

def generate_lr_params(n_epochs, max_lr, lr_scaling_factor=10, lr_decay=0.75, 
                       n_adversarial_start = 200, 
                       role='scl'):
    """
    Generate LR scheduler params for WarmupCosineAnnealingWarmRestarts
    that ensures discriminator and generator follow the same curve in real (epoch) time.

    Parameters:
        - n_epochs: total training epochs
        - max_lr: peak learning rate
        - lr_scaling_factor: factor to determine min LR
        - lr_decay: gamma decay per restart
        - n_adversarial_start: epoch when adversarial training begins
        - n_discriminator_train: frequency of discriminator training relative to generator
        - n_restarts: desired number of cosine peaks (n_restarts)
        - role: 'scl' or discriminator' or 'generator'

    Returns:
        Dict of scheduler parameters
    """

    total_active_epochs = n_epochs
    n_discriminator_train_ = 1
    if role in ['discriminator', 'generator']:
        total_active_epochs = n_epochs - n_adversarial_start
        if role == 'generator':
            n_discriminator_train_ = n_discriminator_train
            
    n_restarts = 4 if total_active_epochs // n_discriminator_train_ > 500 else 2


    T_0 = max(1, (total_active_epochs // n_discriminator_train_) // n_restarts)
    warmup_epochs = max(1, (total_active_epochs // n_discriminator_train_) // 10)

#     if reset_state:
#         if total_active_epochs // n_discriminator_train_ > 400:
#             n_optimizer_resets = 2
#         elif total_active_epochs // n_discriminator_train_ < 100:
#             n_optimizer_resets = 0
#         else:
#             n_optimizer_resets = 1
#     else:
    n_optimizer_resets = 0
        
    if warmup_epochs >= T_0:
        warmup_epochs = 0

    return {
        'max_epochs': n_epochs,
        'maximum_learning_rate': max_lr,
        'minimum_learning_rate': max_lr / lr_scaling_factor,
        'lr_restart_epoch': T_0,  
        'n_optimizer_resets': n_optimizer_resets,  
        'lr_decay': lr_decay,
        'lr_restart_factor': 1,
        'warmup_epochs': warmup_epochs
    }

def generate_discriminator_params(n_epochs, max_lr, discriminator_penalty_weight, 
                                  discriminator_dropout_rate,
                                  lr_scaling_factor,
                                  lr_decay):
    general_params = generate_lr_params(n_epochs, 
                                        max_lr, 
                                        lr_scaling_factor = lr_scaling_factor, 
                                        lr_decay = lr_decay,
                                       role = 'discriminator')
    
    keys_to_keep = ['maximum_learning_rate', 'minimum_learning_rate', 'lr_restart_epoch', 
                   'warmup_epochs', 'lr_decay', 'n_optimizer_resets']
    discriminator_params = {'batch_momentum': 0,
                            'layer_norm': False,
                            'spectral_norm': False,
                            'dropout_rate': discriminator_dropout_rate,
                            'activation_fn': nn.LeakyReLU,
                            'n_hidden_nodes': [768, 512, 256],
                            'lr_restart_factor': 1,
                            'optimizer': torch.optim.Adam,
                            'discriminator_lambda_L2': 1e-3,
                            'discriminator_penalty_weight': discriminator_penalty_weight, 
                            'bionet_activation': False,
                           'smooth_labels': True, 
                           'epsilon_smoothing': 0.1}
    discriminator_params = {**discriminator_params, 
                           **{k:v for k,v in general_params.items() if k in keys_to_keep}}
    
    return discriminator_params

all_data = load_data()
sn_ppis, tf_adata, adata, expr, source_label, target_label, weight_label, stimulation_label, inhibition_label, cat_col, pert_col = all_data



def initialize_mod_and_trainer(
    fold: int, 
    adversarial_penalty: bool = True, 
    randomize: bool = False, 
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
    seed : int, optional
        the seed to use for initialization and training, by default 888
    """


    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------- BATCHES -------------------------------------

    # batches
    split = get_split(fold = fold, author = author)

    n_train_cells = len(split['train_barcodes'])
    n_test_cells = len(split['test_barcodes'])

    n_batches = 22
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

    # ------------------------------------- BULK LEMBAS -------------------------------------
    bionet_params = {'target_steps': 100, 
                     'max_steps': 120, 
                     'exp_factor':50, 
                     'tolerance': 1e-5, 
                     'leak':1e-2}

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

    prediction_loss_fn = torch.nn.MSELoss(reduction='mean')
    lr_params = generate_lr_params(
        n_epochs = max_epochs,
        max_lr = 2e-3, 
        lr_scaling_factor = lr_scaling_factor, 
        lr_decay = lr_decay, 
        role = 'scl'
    )

    # ------------------------------------- REGULARIZERS -------------------------------------
    bionet_params['cat_max_norm'] = 100

    regularization_params = {
        'input_lambda_L2': 0, # irrelevant because setting the requires grad to False

        'bn_weights_lambda_L2': 1e-7,
        'moa_lambda_L1': 1e2,
        'uniform_lambda_L2': 0, #1e-7, 
        'uniform_min': 0, #-1/projection_amplitude_out,
        'uniform_max': 1, #1/projection_amplitude_out,
        'adj_scaling_KL': 0,  # using uniform/bn_weights already
        'adj_prior_mu': 0, # irrelevant because adj_scaling_KL is 0
        'adj_prior_sigma': 0.2, # irrelevant because adj_scaling_KL is 0

        'output_weights_lambda_L2': 1e-7,
        'output_bias_lambda_L2': 1e-7,

        'spectral_loss_factor': 0,


        'global_bias_lambda_L2': 0, # using KL divergence instead
        'global_bias_lambda_L1': 0, # using KL divergence instead
        'cat_bias_lambda_L2': 1e-4,  # allow for generalization (not collapsing on perturbation)
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
    n_layers_vae = 2
    n_nodes = len(set(sn_ppis[source_label].tolist() + sn_ppis[target_label].tolist()))
    vae_n_hidden_nodes = list(np.round(np.linspace(adata.shape[1], n_nodes, n_layers_vae + 2)).astype(int)[1:-1])


    vae_mod_params = {'vae_batch_momentum': 0.01, 
                  'vae_layer_norm': False, 
                  'vae_dropout_rate': 0.7,
                  'vae_activation_fn': nn.LeakyReLU,
                  'vae_n_hidden_nodes': vae_n_hidden_nodes, 
                  'vae_var_min': 1e-4, 
                     }

    vae_params = {'prior_mu': 0,
           'prior_sigma': 1,
           'lambda_l2': 1e-5,
           'scaling_KL': 1e-3, 
           'optimizer': torch.optim.Adam}
    vae_lr_params = generate_lr_params(
            n_epochs = max_epochs, 
            max_lr = 5e-4,
            lr_scaling_factor = lr_scaling_factor, 
            lr_decay = lr_decay,
            role = 'discriminator'
        ) 

    vae_params = {**vae_params, **vae_lr_params}
    del vae_params['max_epochs']

    # ------------------------------------- DISCRIMINATORS -------------------------------------
    if adversarial_penalty:
        cat_discriminator_penalty_weight = pp.discriminator_weight_curve(
            n_epochs = max_epochs - n_adversarial_start,
            min_penalty_weight = 0.1,
            max_penalty_weight = 12,
            a = 1,
            b = 2, 
            curve_type = 'power')

        pert_discriminator_penalty_weight = pp.discriminator_weight_curve(
            n_epochs = max_epochs - n_adversarial_start,
            min_penalty_weight = 0.1,
            max_penalty_weight = 8,
            a = 1,
            b = 3.5, 
            curve_type = 'power')

        n_cat_discriminator_train = 5
        n_pert_discriminator_train = 5

    else: # no adversarial training, handled internally by the trainer
        cat_discriminator_penalty_weight= [0]*(max_epochs - n_adversarial_start)
        pert_discriminator_penalty_weight = [0]*(max_epochs - n_adversarial_start)

        n_cat_discriminator_train = 0
        n_pert_discriminator_train = 0


    cat_discriminator_params = generate_discriminator_params(
        n_epochs = max_epochs,                              
        max_lr = 1e-3,                                          
        discriminator_dropout_rate = 0.1,                                                 
        discriminator_penalty_weight = cat_discriminator_penalty_weight,                                               
        lr_scaling_factor = lr_scaling_factor, lr_decay = lr_decay)
    pert_discriminator_params = generate_discriminator_params(
        n_epochs = max_epochs,                        
        max_lr = 1e-3,                                                    
        discriminator_dropout_rate = 0.1,                                                
        discriminator_penalty_weight = pert_discriminator_penalty_weight,                                                    
        lr_scaling_factor = lr_scaling_factor, lr_decay = lr_decay)

    
    # ------------------------------------- INITIALIZATION -------------------------------------
    # input stimulation
    X_in = pd.DataFrame(tf_adata.obs.stim.cat.codes, columns = ['STIM'])
    
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





# def setup_prediction(mod, 
#                      train_cells,
#                      tf_adata, 
#                      train_mode, 
#                      counterfactual,
#                      ):

#     cov_idx_map = dict(zip(mod.signaling_network.covariates['seurat_annotations'], 
#                         mod.signaling_network.covariates_idx['seurat_annotations']))
#     cov_rev_map = {v:k for k,v in cov_idx_map.items()}
#     full_expr, full_X, full_covariates = None, None, None

#     train_conds = tf_adata.obs.loc[train_cells, 'condition'].unique()
#     test_conds = tf_adata[~tf_adata.obs.condition.isin(train_conds), ].obs.condition.unique()

#     if not counterfactual and not train_mode:
#         raise ValueError('Trying to predict test cells without a counterfactual')

#     iterable_conds = train_conds if train_mode else test_conds
#     for cond in sorted(iterable_conds): 
#         stim, ct = cond.split('^') # stim and ct of the prediction
        
#         if counterfactual: 
#             predict_cells_from = tf_adata.obs[(tf_adata.obs['condition'] == rev_stim[stim] + '^' + ct)].index.tolist()
#         else:
#             predict_cells_from = tf_adata.obs[tf_adata.obs.condition == cond].index.tolist()
#         n_predictions = len(predict_cells_from)
            
#         # generate model inputs
#         # input gene expression: counterfactual or not
#         expr_in = mod.df_to_tensor(mod.expr.loc[predict_cells_from, :])
        
#         # input stimulation
#         X_in = pd.DataFrame(data = {'IFNB1': [stim_map[stim]]*n_predictions})
#         X_in = mod.df_to_tensor(X_in)
        
#         # input ct
#         covariates_in = torch.tensor([cov_idx_map[ct]]*n_predictions,
#                                         device = mod.device, dtype = torch.int64).view(-1,1)
        
#         full_expr = expr_in if full_expr is None else torch.cat((full_expr, expr_in), dim = 0)
#         full_X = X_in if full_X is None else torch.cat((full_X, X_in), dim = 0)
#         full_covariates = covariates_in if full_covariates is None else torch.cat((full_covariates, covariates_in), dim = 0)
        
#         clear_memory()
    
#     # metadata setup
#     obs = pd.DataFrame(full_covariates.detach().cpu().numpy())
#     obs.columns = ['seurat_annotations']
#     obs.seurat_annotations = obs.seurat_annotations.map(cov_rev_map)
#     obs['stim'] = pd.Series(full_X.detach().cpu().numpy().reshape(-1)).map(rev_stim_map)
#     obs['condition'] = obs['stim'].astype(str) + '^' + obs['seurat_annotations'].astype(str)

#     return full_expr, full_X, full_covariates, obs


# def run_prediction(mod, 
#                    remove_type, 
#                    return_bias, 
#                    X_in, 
#                    covariates_idx, 
#                    expr, 
#                   obs, 
#                   return_full):

#     # ----------------CHECKS----------------
#     if type(remove_type) != list:
#         remove_type = [remove_type]
#     if len(remove_type) not in [1,2]:
#         raise ValueError('Cannot remove more than two components at once')
#     if len(set(remove_type).difference(['none', 'global_bias', 'categorical_bias', 'total_bias', 'adj'])) > 0:
#         raise ValeuError('Incorrect remove_type specified')
#     if len(remove_type) == 2:
#         if sorted(remove_type) not in [['adj', 'categorical_bias'], ['adj', 'global_bias']]:
#             raise ValueError('Can only specify multiple remove types when ')
#     if remove_type != ['none'] and return_bias:
#         raise ValueError('Have not considered looking at the bias components without the full forward pass')
    
#     # ----------------FORWARD PASS----------------
#     mod.eval()
#     with torch.inference_mode():
#         X_full = mod.input_layer(X_in) # input ligands to signaling network

#         bias_cats = torch.zeros_like(X_full.T, device = mod.signaling_network.device, dtype = mod.signaling_network.dtype)
#         # add categorical covariates
#         for cat_group_idx in range(covariates_idx.shape[1]):
#             cat_group = mod.signaling_network._cat_group_idx[cat_group_idx]
#             bias_cats += mod.signaling_network.cat_embeddings[cat_group](covariates_idx[:,cat_group_idx]).T

#         bias_mu, bias_log_sigma_squared, bias_global = mod.signaling_network.vae(expr)
#         bias_global.data.masked_fill_(mask = mod.signaling_network.bias_mask.T.expand(bias_global.shape[0], -1), value = 0.0) # apply bias mask

# #         if 'bias_global_scaler' in mod.signaling_network.bionet_params:
# #             bias_global /= mod.signaling_network.bionet_params['bias_global_scaler']

# #         # this is equivalent to dividing bias_tot by the scalers, but since we get out individual components 
# #         # we should scale each individual component
# #         elif 'bias_tot_scaler' in mod.signaling_network.bionet_params:
# #             bias_global /= mod.signaling_network.bionet_params['bias_tot_scaler']
# #             bias_cats /= mod.signaling_network.bionet_params['bias_tot_scaler']

#         if return_bias:
#             bias_tot = bias_global.T + bias_cats
#             bias_sigma = torch.exp(bias_log_sigma_squared/2.) + mod.signaling_network.vae.var_min
#             clear_memory()
#             return bias_global, bias_mu, bias_sigma, bias_cats, bias_tot, obs

#         if remove_type == ['none'] or remove_type == ['adj']:
#             bias_tot = bias_global.T + bias_cats # include all biases
#         elif 'categorical_bias' in remove_type:
#             bias_tot = bias_global.T # don't include categorical bias
#         elif 'global_bias' in remove_type:
#             bias_tot = bias_cats # don't include global bias
#         elif remove_type == ['total_bias']:
#             bias_tot = torch.zeros_like(X_full.T, device = mod.signaling_network.device, dtype = mod.signaling_network.dtype) # don't include bias    
#         else:
#             raise ValueError('Incorrect remove_type specified')

#         X_bias = X_full.T + bias_tot # this is the bias with the projection_amplitude included
#         X_new = torch.zeros_like(X_bias) #initialize hidden state values at 0

#         if 'adj' in remove_type: 
#             X_new = mod.signaling_network.activation(X_bias,
#                                                      mod.signaling_network.bionet_params['leak'])
#             # this is the equivalen of setting the signaling network weights to 0 in the 
#             # iteration below because this makes X_new = 0 at every element in the forward pass

#             # see commented out remove_type == 'adj' below for the equivalent
#         else:
#             for t in range(mod.signaling_network.bionet_params['max_steps']): # like an RNN, updating from previous time step
#                 X_old = X_new

#     #             if remove_type == 'adj':
#     #                 X_new = torch.mm(torch.zeros(mod.signaling_network.weights.shape,
#     #                             device = mod.signaling_network.device, 
#     #                             requires_grad=False), X_new)
#                 if 'signaling_weights_scaler' in mod.signaling_network.bionet_params: #DEV
#                      X_new = torch.mm(mod.signaling_network.weights*mod.signaling_network.bionet_params['signaling_weights_scaler'],
#                                       X_new) # scale matrix by edge weights
#                 else:
#                     X_new = torch.mm(mod.signaling_network.weights, X_new) # scale matrix by edge weights

#                 X_new = X_new + X_bias  # add original values and bias       
#                 X_new = mod.signaling_network.activation(X_new, mod.signaling_network.bionet_params['leak'])

#                 if (t % 10 == 0) and (t > 20):
#                     diff = torch.max(torch.abs(X_new - X_old))    
#                     if diff.lt(mod.signaling_network.bionet_params['tolerance']):
#                         break
        
#         Y_full = X_new.T
        
#         if return_full:
#             return sc.AnnData(X = Y_full.detach().cpu().numpy(), obs = obs)

#         y_predicted = mod.output_layer(Y_full)
        
#         clear_memory()
#     if remove_type == ['none']:
#         consistent_forward = True
#         y_predicted_, Y_full_, biases_ = mod(X_in, covariates_idx, expr)
#         for tt_1, tt_2 in zip([y_predicted, Y_full, bias_global, bias_mu, bias_log_sigma_squared], 
#                               [y_predicted_, Y_full_, *biases_]):
#             if not torch.equal(tt_1, tt_2):
#                 consistent_forward = False
#         if not consistent_forward:
#             raise ValueError('Prediction here does not match forward pass')
#         del y_predicted_, Y_full_, biases_

#     y_predicted = pd.DataFrame(y_predicted.detach().cpu().numpy())
#     y_predicted.columns = mod.y_out.columns
#     tf_adata_predicted = sc.AnnData(X = y_predicted, obs = obs)

#     del X_full, X_in, covariates_idx, expr
#     del bias_mu, bias_log_sigma_squared, bias_global
#     del X_bias, X_new, Y_full
#     if 'adj' not in remove_type:
#         del X_old

#     clear_memory()

#     return tf_adata_predicted

# def get_prediction(mod,
#                    train_cells,
#                    tf_adata,
#                    train_mode: bool = False,
#                    counterfactual: bool = True,
#                    remove_type: Literal['none', 'global_bias', 'categorical_bias', 'total_bias', 'adj'] = 'none',
#                    return_bias: bool = False, 
#                   max_cells = None, 
#                   return_full: bool = False):
#     """Get prediction from a model given a counterfactual

#     Parameters
#     ----------
#     mod : _type_
#         _description_
#     tf_adata : _type_
#         all the actual data
#     train_cells : List[str]
#         all cells the model was trained on 
#     train_mode : bool
#         predicts the training conditions, as a negative control to see how model predictions perform on training data
#     return_bias : bool, optional
#         whether to return bias terms (True) or prediction (False), by default False
#     counterfactual : bool, optional
#         whether to calculate from opposite stimulation condition (True) or same one (False)
#     remove_type : Literal['none', 'global_bias', 'categorical_bias', 'total_bias', 'adj'], optional
#         can be a string or a list of strings
#         which components of bias/adj matrix to include in the prediction, by default 'all_bias'; 
#         only incorporated if return_bias = False
#         any bias component includes the full adjacency matrix
#         - 'none': includes all components in the prediction
#         - 'categorical_bias': includes global but excludes categorical bias in the prediction
#         - 'global_bias': includes categorical but excludes global bias in the prediction
#         - 'total_bias': does not include bias in the prediction (just input and signaling weights)
#         - 'adj': includes all bias but sets signaling weights to 0
#         the only list of strings are combining either categorical or global bias with adj, since in these cases just removing one of the two bias components still leaves two components in the model, making it hard to decouple effects. 
#     test_cells : 
#         the list of test cell barcodes, necessary if return_loss = True
#     max_cells : int
#         the max cells in a forward pass; for cuda memory, will break up into chunks
#     return_full : bool, optional
#         whether to return model output prior to ProjectOutput transformation (True) or after (False), by default False
#     """
    
#     max_cells = np.inf if max_cells is None else max_cells
    
#     expr, X_in, covariates_idx, obs = setup_prediction(mod, 
#                                                        train_cells,
#                                                        tf_adata, 
#                                                        train_mode, 
#                                                        counterfactual
#                                                               )

#     if expr.shape[0] < max_cells:
#         res = run_prediction(mod, 
#                              remove_type, 
#                              return_bias, 
#                              X_in, 
#                              covariates_idx, 
#                              expr, 
#                             obs, 
#                             return_full)

#     else:
#         res = []

#         # split into chunks for cuda memory
#         expr_chunks = torch.split(expr, max_cells)
#         X_in_chunks = torch.split(X_in, max_cells)
#         covariates_idx_chunks = torch.split(covariates_idx, max_cells)
#         obs = obs.copy()
#         obs.index = obs.index.astype(str)
#         obs_index = obs.index
#         obs_chunks = [obs_index[i:i+max_cells] for i in range(0, len(obs_index), max_cells)]

#         for chunk_idx in range(len(expr_chunks)):
#             res_ = run_prediction(mod, 
#                                  remove_type, 
#                                  return_bias, 
#                                  X_in_chunks[chunk_idx], 
#                                  covariates_idx_chunks[chunk_idx], 
#                                  expr_chunks[chunk_idx], 
#                                 obs.loc[obs_chunks[chunk_idx], :], 
#                                  return_full)
#             res.append(res_)

#         if not return_bias:
#             res = sc.concat(res)
#             res.obs_names_make_unique()
#         else:
#             bias_global_chunks, bias_mu_chunks, bias_sigma_chunks, bias_cats_chunks, bias_tot_chunks, obs_chunks = zip(*res)

#             obs = pd.concat(obs_chunks, axis=0)
#             bias_global = torch.cat(bias_global_chunks, dim=0)
#             bias_mu = torch.cat(bias_mu_chunks, dim=0)
#             bias_sigma = torch.cat(bias_sigma_chunks, dim=0)
#             bias_cats = torch.cat(bias_cats_chunks, dim=1)
#             bias_tot = torch.cat(bias_tot_chunks, dim=1)

#             res = (bias_global, bias_mu, bias_sigma, bias_cats, bias_tot, obs)
#     return res



# def get_loss(tf_adata, tf_adata_predicted, device = device):
#     """Calculates the loss between predicted and actual data per condition. 
#     geom_loss by default normalizes to sample size so that doesn't need to be done. 
#     Does take average across conditions for the total loss (rather than simply summing), so that it does not change with 
#     the number of conditions 
#     """
#     loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05).to(device)
#     loss = {}
#     conds = tf_adata_predicted.obs.condition.unique()
#     for cond in conds:
#         y_predicted = torch.tensor(tf_adata_predicted[tf_adata_predicted.obs.condition == cond, ].to_df().values).to(device)
#         y_actual = torch.tensor(tf_adata[tf_adata.obs.condition == cond, ].to_df().values).to(device)
#         loss[cond] = loss_fn(y_predicted, y_actual)
#         clear_memory()
#     loss['EMD Loss'] = sum(loss.values())/len(loss) # averaged across condition to not scale with n_conditions
    
    
#     return {k: float(v.cpu().numpy()) for k,v in loss.items()}



# def get_best_hyperparams(res, 
#                          iter_cols = ['max_epochs', 'max_lr', 'train_batch_size'], 
#                         loss_col = 'emd_loss_total'):
    
#     res_nans = res.groupby(iter_cols)[loss_col].apply(lambda x: x.isna().any()).reset_index()
#     res_nans = res_nans[res_nans[loss_col]]
#     res_nans.set_index(iter_cols, inplace = True)

#     mean_emd = res.groupby(iter_cols).emd_loss_total.mean().reset_index()
#     mean_emd = mean_emd.set_index(iter_cols).drop(res_nans.index)
#     best_emd_loss = mean_emd.emd_loss_total.min()
#     best_emd = mean_emd[mean_emd.emd_loss_total == best_emd_loss]
#     best_emd_mean = best_emd.copy()
    
#     best_hyperparams = dict(zip(best_emd.index.names, best_emd.index[0]))
#     best_emd = res[(res.max_epochs == best_hyperparams['max_epochs']) & 
#                    (res.max_lr == best_hyperparams['max_lr']) & 
#                    (res.train_batch_size == best_hyperparams['train_batch_size'])]
    
#     return best_emd_mean, best_hyperparams, best_emd


# def adata_dimviz_bias(adata, reduction_type, cat, subset_size = None, seed = 888):
#     viz_df = pd.DataFrame(adata.obsm['X_' + reduction_type])
#     viz_df = pd.concat([viz_df, pd.DataFrame(adata.obs[cat]).reset_index(drop = True)], ignore_index = True, axis = 1)

#     reduction_type_ = 'pc' if reduction_type == 'pca' else reduction_type
#     viz_df.columns = [reduction_type_.upper() + str(i+1) for i in range(viz_df.shape[1])]
#     viz_df.columns = viz_df.columns[:-1].tolist() + [cat]
    
#     nmi = normalized_mutual_info_score(adata.obs.leiden, adata.obs[cat])
    
#     if subset_size is not None:
#         cell_prop = viz_df[cat].value_counts()/viz_df.shape[0]
#         index_to_keep = []
#         for cell_type in viz_df[cat].unique():
#             np.random.seed(seed)
#             index_to_keep += np.random.choice(viz_df[viz_df[cat] == cell_type].index, 
#                                               size = int(np.round(cell_prop.loc[cell_type]*subset_size)), 
#                                               replace = False).tolist()
#         viz_df = viz_df.loc[index_to_keep, :]
    
#     # shuffle
#     viz_df = viz_df.sample(frac=1, random_state = seed).reset_index(drop=True)
#     return viz_df,nmi

# def adata_dimviz_prediction(adata, reduction_type, cats, max_condition_size = None, seed = 888):
#     viz_df = pd.DataFrame(adata.obsm['X_' + reduction_type])
#     for cat in cats:
#         viz_df = pd.concat([viz_df, pd.DataFrame(adata.obs[cat]).reset_index(drop = True)], ignore_index = True, axis = 1)
#     if reduction_type=='umap':
#         viz_df.columns = [reduction_type.upper() + str(i+1) for i in range(viz_df.shape[1])]
#     elif reduction_type=='pca':
#         viz_df.columns = [reduction_type.upper()[:-1] + str(i+1) for i in range(viz_df.shape[1])]
#     viz_df.columns = viz_df.columns[:-len(cats)].tolist() + cats
    
#     if max_condition_size is not None:
# #         cell_prop = viz_df.condition.value_counts()/viz_df.shape[0]
# #         index_to_keep = []
# #         for cond in viz_df.condition.unique():
# #             np.random.seed(seed)
# #             index_to_keep += np.random.choice(viz_df[viz_df.condition == cond].index, 
# #                                               size = int(np.round(cell_prop.loc[cond]*subset_size)), 
# #                                               replace = False).tolist()
# #         viz_df = viz_df.loc[index_to_keep, :]
        
        
#         vc = viz_df.condition.value_counts()
#         subset_conds = vc[vc > max_condition_size].index.tolist()

#         drop_idx = []
#         for subset_cond in subset_conds: 
#             np.random.seed(seed)
#             drop_idx += list(np.random.choice(viz_df[viz_df.condition == subset_cond].index, 
#                              vc[subset_cond] - subset_size, 
#                             replace = False))
#         viz_df.drop(index = drop_idx, inplace = True)

#     # shuffle
    
#     viz_df = viz_df.sample(frac=1, random_state = seed).reset_index(drop=True)
#     return viz_df


# def prepare_for_metrics(tf_adata, 
#                 tf_adata_predicted, 
#                 resolution,
#                 calculation_type: Literal['embed', 'project'] = 'project',
#                 n_neighbors: int = 15, 
#                 run_umap = True, 
#                        ):
#     """Combine predictions with actual values and then recalculate neighbors/clusters. 
#     Project will project predictions to space calculated on actual values
#     Embed will jointly embed the actual and predicted values."""
    

#     tf_adata_actual = tf_adata.copy()
#     tf_adata_actual.obs['batch'] = 'actual'

#     tf_adata_predicted = tf_adata_predicted.copy()
#     tf_adata_predicted.obs['batch'] = 'predicted'
    
#     tf_adata_ = sc.concat([tf_adata_actual, tf_adata_predicted])
#     tf_adata_.obs['barcode'] = tf_adata_.obs.index.tolist()

#     if len(set(tf_adata_.obs_names)) < len(tf_adata_.obs_names):
#         tf_adata_.obs_names_make_unique()
    
#     if calculation_type == 'project': # project the predicted data into the actual data space
#         # project new data into PCA space
#         pc_rank = tf_adata.uns["pca"]['pca_rank']
#         # pca_mod = tf_adata.uns['pca']['pca_mod']
#         tf_adata_.obsm['X_pca'] = project_to_pca(X_new=tf_adata_.X, adata=tf_adata)
#         tf_adata_.uns['pca'] = tf_adata.uns['pca'].copy()
        
#         # neihgbors/clustering
#         embed_adata(adata = tf_adata_,
#                           cluster_col_name = 'new_TF_clusters',
#                           n_components = np.nan,
#                           pc_rank = np.nan,
#                           resolution = resolution,
#                           n_neighbors = n_neighbors,
#                           nmi_label = None,
#                           run_pca = False, 
#                           run_umap = False, 
#                           cluster_data = True)
        
#         # project from PCA space into UMAP space
#         if run_umap:
#             from scanpy.tools._utils import _choose_representation
#             from scanpy._utils import NeighborsView
            
#             neighbors = NeighborsView(tf_adata_, 'neighbors')
#             X_pca = _choose_representation(
#                 tf_adata_,
#                 use_rep=neighbors['params'].get("use_rep", None),
#                 n_pcs=neighbors['params'].get("n_pcs", None),
#                 silent=True,
#             )

#             tf_adata_.obsm['X_umap'] = tf_adata_actual.uns['umap']['umap_mod'].transform(X_pca)
#             tf_adata_.uns['umap'] = tf_adata_actual.uns['umap'].copy()
        

#     elif calculation_type == 'embed': # embed the combined data again
#         embed_adata(adata = tf_adata_,
#                           cluster_col_name = 'new_TF_clusters',
#                           n_components = 50,
#                           pc_rank = 'automate',
#                           resolution = resolution,
#                           n_neighbors = n_neighbors,
#                           nmi_label = None,
#                           run_pca = True, 
#                           run_umap = run_umap, 
#                          cluster_data = True)
#     return tf_adata_

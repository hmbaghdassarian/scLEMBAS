#!/usr/bin/env python
# coding: utf-8

# On the validation data, we identify the best set of hyperparameters (epochs, train batch size, and learning rate) that minimize the loss (EMD) of the validation data predictions. 
# 
# During training, the each cell in the generated data is drawn from the same distribution as each cell in the actual data. Thus, the EMD can just be calculated across all cells. For validation, we will generate different numbers of cells for each condition (stimulation + cell type combination) as compared to the actual data. Thus, we will calculate the EMD separately on each condition and add them together. 
# 
# We run the EMD loss on in-distribution gene expression (asking the counter-factual of predicting out of distribtion conditions from in-distribution gene expression inputs). What would the TF activity of a measured cell look like in an unmeasured condition?

# In[1]:


import os
import glob

import torch
import torch.nn as nn
from geomloss import SamplesLoss

import pandas as pd
import scanpy as sc
import numpy as np


# In[2]:


import sys
sclembas = '/home/hmbaghda/Projects/scLEMBAS'
sys.path.insert(1, os.path.join(sclembas))
from scLEMBAS import io
from scLEMBAS.model.scl import SignalingModel
from scLEMBAS.model.train import TrainSC
from scLEMBAS.preprocess import exponential_discriminator_weight


# In[3]:


n_cores = 20
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)

seed = 888
device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = '/nobackup/users/hmbaghda/scLEMBAS/analysis'
models_path = os.path.join(data_path, 'processed', 'models')


# Load the data:

# In[4]:


adata = sc.read_h5ad(os.path.join(data_path, 'processed', 'kang_expr_scored.h5ad'))
sn_ppis = pd.read_csv(os.path.join(data_path, 'processed', 'Kang_sn_ppis.csv'), index_col = 0)

tf_adata = io.read_tfad(os.path.join(data_path, 'processed', 'Kang_tf_activity.h5ad'))
tf_adata.obs['condition'] = tf_adata.obs['stim'].astype(str) + '^' + tf_adata.obs['seurat_annotations'].astype(str)


# In[5]:


data_split_path = os.path.join(os.path.join(data_path, 'processed', 'data_split_barcodes'))
fold_keys = ['train_cells', 'val_cells', 'train_cond', 'val_cond']
k_fold_cells = {}
for k in range(5):
    k_fold_cells[k] = {fk: open(os.path.join(data_split_path, 'kang_' + str(k) + '_' + fk + '.txt')).read().splitlines()
                      for fk in fold_keys}


# In[6]:


source_label = 'source_genesymbol'
target_label = 'target_genesymbol'
weight_label = 'mode_of_action'
stimulation_label = 'consensus_stimulation'
inhibition_label = 'consensus_inhibition'


# Setup the parameters:

# In[7]:


def generate_lr_params(n_epochs, max_lr, lr_scaling_factor = 10, lr_decay = 0.9):
    lr_period = 3 if n_epochs < 500 else 4
    lr_params = {'max_epochs': n_epochs, 
                 'maximum_learning_rate': max_lr, 
                 'minimum_learning_rate': max_lr/lr_scaling_factor,
                 'lr_restart_epoch': int(n_epochs/lr_period), 
                 'reset_optimizer_epoch': int(n_epochs/3), 
                'lr_decay': lr_decay, 
                 'lr_restart_factor': 1, 
                 'warmup_epochs': int(n_epochs/10)}
    return lr_params

def generate_discriminator_params(n_epochs, max_lr, discriminator_penalty_weight, 
                                  lr_scaling_factor = 10, lr_decay = 0.9):
    general_params = generate_lr_params(n_epochs, max_lr, lr_scaling_factor = lr_scaling_factor, 
                                        lr_decay = lr_decay)
    
    keys_to_keep = ['maximum_learning_rate', 'minimum_learning_rate', 'lr_restart_epoch', 
                   'warmup_epochs', 'lr_decay', 'reset_optimizer_epoch']
    discriminator_params = {'batch_momentum': 0.01,
                            'layer_norm': False,
                            'dropout_rate': 0.1,
                            'activation_fn': nn.LeakyReLU,
                            'n_hidden_nodes': [768, 512, 256],
                            'lr_restart_factor': 1,
                            'optimizer': torch.optim.Adam,
                            'discriminator_lambda_L2': 1e-3,
                            'discriminator_penalty_weight': discriminator_penalty_weight}
    discriminator_params = {**discriminator_params, 
                           **{k:v for k,v in general_params.items() if k in keys_to_keep}}
    
    return discriminator_params


# In[8]:


# vae
n_layers_vae = 2
n_nodes = len(set(sn_ppis[source_label].tolist() + sn_ppis[target_label].tolist()))
vae_n_hidden_nodes = list(np.round(np.linspace(adata.shape[1], n_nodes, n_layers_vae + 2)).astype(int)[1:-1])

# linear scaling of inputs/outputs
projection_amplitude_in = 1
projection_amplitude_out = 1

# other parameters
bionet_params = {'target_steps': 100, 
                 'max_steps': 120, 
                 'exp_factor':50, 
                 'tolerance': 1e-5, 
                 'leak':1e-2, 
                'cat_max_norm': 1} 
vae_params = {'vae_batch_momentum': 0.01, 'vae_layer_norm': False, 'vae_dropout_rate': 0.1,
              'vae_activation_fn': nn.LeakyReLU,
              'vae_n_hidden_nodes': vae_n_hidden_nodes, 
              'vae_var_min': 1e-4}
bionet_params = {**bionet_params, **vae_params}

# training parameters
other_params_default = {'network_noise_scale': 10, 'gradient_noise_scale': 1e-9, 
               'test_batch_size': np.nan}
spectral_radius_params = {'n_probes_spectral': 5, 
                          'power_steps_spectral': 50, 
                          'subset_n_spectral': 10}
target_spectral_radius = 0.9

regularization_params_default = {'input_lambda_L2': 0, # doesn't matter if setting the requires grad to False
                         'bn_weights_lambda_l2': 1e-7, 
                         'bn_bias_lambda_L2': 0, # don't incorporate because of KL divergence
                         'output_weights_lambda_L2': 1e-7,
                         'output_bias_lambda_L2': 1e-7,
                         'moa_lambda_L1': 1e2,  
                         'uniform_lambda_L2': 1e-7,#, 1e-5,
                         'uniform_min': 0,
                         'uniform_max': 1, 
                         'spectral_loss_factor': 1e-6,
                        'vae_lambda_l2': 1e-7, 
                        'vae_scaling_KL': 1e-2}


# Iterate:

# In[9]:


max_epochs_iter = [100, 300, 600, 900, 1500]
max_lr_iter = [1e-3, 1e-4]
train_batch_iter = [256, 512, 1024, 2048]


# In[10]:


if os.path.isfile(os.path.join(data_path, 'processed', 'Kang_k_fold_validation_results.csv')):
    res = pd.read_csv(os.path.join(data_path, 'processed', 'Kang_k_fold_validation_results.csv'), index_col = 0)
else:
    res = pd.DataFrame(columns = ['max_epochs', 'max_lr', 'train_batch_size', 'k', 'emd_loss_total', 
                                 'KL_regularization'])

if (res.shape[0] == 0) or (not np.any(res.k == max(k_fold_cells))):
    best_emd_loss = np.inf
else:
    
    best_emd_loss = res.loc[:res[res.k == max(k_fold_cells)].index.max(),:].groupby(['max_epochs', 'max_lr', 'train_batch_size']).emd_loss_total.mean().min()

stim_map = {'STIM': 1, 'CTRL': 0}
max_err_iter = 5


# In[52]:


for max_epochs in max_epochs_iter:
    
    discriminator_penalty_weight, _ = exponential_discriminator_weight(n_epochs = max_epochs,
                                                                   min_penalty_weight = 0.1,
                                                                   max_penalty_weight = 4, 
                                                                  a = 1, 
                                                                  b = 0.007)
    for max_lr in max_lr_iter:
        
        lr_params = generate_lr_params(n_epochs = max_epochs, max_lr = max_lr, lr_scaling_factor = 10, lr_decay = 0.9)
        discriminator_params = generate_discriminator_params(n_epochs = max_epochs, max_lr = max_lr, 
                              discriminator_penalty_weight = discriminator_penalty_weight, 
                              lr_scaling_factor = 10, lr_decay = 0.9)
        
        for train_batch in train_batch_iter:
            if res[(res.max_epochs == max_epochs) & (res.max_lr == max_lr) & 
                    (res.train_batch_size == train_batch)].shape[0] == 0: # no k's run
                emd_loss_k = []
                trainers_k = {}
            else:
                emd_loss_k = res[(res.max_epochs == max_epochs) & (res.max_lr == max_lr) &
                                 (res.train_batch_size == train_batch)].emd_loss_total.tolist()
                trainers_k = io.read_pickled_object(os.path.join(data_path, 'processed', 'Kang_trainers_k.pickle'))
                
            for k in k_fold_cells:
                if res[(res.max_epochs == max_epochs) & (res.max_lr == max_lr) & 
                    (res.train_batch_size == train_batch) & (res.k == k)].shape[0] == 1:
                    run = False
                    pass
                else:
                    run = True
                    # sanity checks
                    if k in trainers_k:
                        raise ValueError('Something went wrong in loading new files')
                    elif len(trainers_k) > 0:
                        if (max(trainers_k) != k - 1):
                            raise ValueError('Something went wrong in loading new files')
                        for _k in trainers_k:
                            if not isinstance(trainers_k[_k], float): # not nan
                                a = trainers_k[_k].hyper_params['max_epochs'] == max_epochs
                                b = trainers_k[_k].hyper_params['maximum_learning_rate'] == max_lr
                                c = trainers_k[_k].hyper_params['train_batch_size'] == train_batch
                                if not (a and b and c):
                                    raise ValueError('Something went wrong in loading new files')    
                        
                        
                    print('epochs: {}, lr: {:.4f}, train batch size: {}, k: {}'.format(max_epochs, max_lr, train_batch, k))
                    print('------')
                    
                    train_cells = k_fold_cells[k]['train_cells']
                    val_cells = k_fold_cells[k]['val_cells']

                    # tune other parameters as a function of those being adjusted
                    regularization_params = regularization_params_default.copy()
                    other_params = {**other_params_default,
                                    **{'train_batch_size': train_batch,
                                       'validation_batch_size': len(val_cells)}}
                    training_params = {**lr_params, **other_params, **regularization_params, **spectral_radius_params}


                    mod = SignalingModel(net = sn_ppis,
                                         X_in = pd.DataFrame(tf_adata.obs.stim.cat.codes, columns = ['IFNB1']),
                                         y_out = tf_adata.to_df().copy(), 
                                         expr = adata.to_df().copy(), 
                                         covariates = tf_adata.obs.copy(),
                                         categorical_covariate_keys = ['seurat_annotations'],
                                         projection_amplitude_in = projection_amplitude_in, 
                                         projection_amplitude_out = projection_amplitude_out,
                                         weight_label = weight_label, source_label = source_label, target_label = target_label,
                                         bionet_params = bionet_params, 
                                         dtype = torch.float32, device = device, seed = seed)
                    # model setup
                    mod.input_layer.weights.requires_grad = False # don't learn scaling factors for the ligand input concentrations
                    mod.signaling_network.prescale_weights(target_radius = target_spectral_radius) # spectral radius

                    # training loop
                    trainer = TrainSC(mod = mod,
                                       prediction_optimizer = torch.optim.Adam,
                                       prediction_loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05).to(device), #torch.nn.MSELoss(reduction='mean'),
                                      discriminator_params = discriminator_params,
                                       hyper_params = training_params,
                                       train_split = {'train': train_cells, 'test': None, 'validation': val_cells}, 
                                       train_seed = seed, 
                                       track_test = False,
                                       track_validation = False)
                    try:
                        mod = trainer.train_model(verbose = False)
                        err = False
                    except:
                        err = True
                        err_iter = 0
                        while err and (err_iter < max_err_iter): 
                            try: # try with higher regularization of KL which prevents NaN errors
                                if err_iter <3:
                                    kl_scaler = 2
                                else:
                                    kl_scaler = 10
                                regularization_params['vae_scaling_KL'] *= kl_scaler
                                training_params = {**lr_params, **other_params, **regularization_params, **spectral_radius_params}

                                mod = SignalingModel(net = sn_ppis,
                                                     X_in = pd.DataFrame(tf_adata.obs.stim.cat.codes, columns = ['IFNB1']),
                                                     y_out = tf_adata.to_df().copy(), 
                                                     expr = adata.to_df().copy(), 
                                                     covariates = tf_adata.obs.copy(),
                                                     categorical_covariate_keys = ['seurat_annotations'],
                                                     projection_amplitude_in = projection_amplitude_in, 
                                                     projection_amplitude_out = projection_amplitude_out,
                                                     weight_label = weight_label, source_label = source_label, target_label = target_label,
                                                     bionet_params = bionet_params, 
                                                     dtype = torch.float32, device = device, seed = seed)
                                # model setup
                                mod.input_layer.weights.requires_grad = False # don't learn scaling factors for the ligand input concentrations
                                mod.signaling_network.prescale_weights(target_radius = target_spectral_radius) # spectral radius

                                # training loop
                                trainer = TrainSC(mod = mod,
                                                   prediction_optimizer = torch.optim.Adam,
                                                   prediction_loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05).to(device), #torch.nn.MSELoss(reduction='mean'),
                                                  discriminator_params = discriminator_params,
                                                   hyper_params = training_params,
                                                   train_split = {'train': train_cells, 'test': None, 'validation': val_cells}, 
                                                   train_seed = seed, 
                                                   track_test = False,
                                                   track_validation = False)
                                mod = trainer.train_model(verbose = False)
                                err = False
                            except: 
                                err_iter += 1
                    if err:
                        emd_loss_tot = np.nan 
                        torch.cuda.empty_cache()
                    else:        
                        
                        # prediction on in-distribution gene expression inputs, per condition
                        emd_loss_tot = 0
                        for cond in k_fold_cells[k]['val_cond']:
                            loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05).to(device)
                            stim, ct = cond.split('^')

                            vall_cells_cond = tf_adata.obs[(tf_adata.obs.index.isin(val_cells)) & (tf_adata.obs['condition'] == cond)].index.tolist()
                            y_val = mod.df_to_tensor(trainer.y_validation.loc[vall_cells_cond, :])

                            # in distribution gene expression!
                            expr_val = mod.df_to_tensor(mod.expr.loc[train_cells, :])

                            # set the stimulation condition we want to predict
                            X_val = pd.DataFrame(data = {'IFNB1': [stim_map[stim]]*len(train_cells)})
                            X_val = mod.df_to_tensor(X_val)

                            # set the cell type we want to predict
                            cov_idx_map = dict(zip(mod.signaling_network.covariates['seurat_annotations'], 
                                                   mod.signaling_network.covariates_idx['seurat_annotations']))
                            covariates_idx_val = torch.tensor([cov_idx_map[ct]]*len(train_cells), device = mod.device, dtype = torch.int64).view(-1,1)


                            mod.eval()
                            with torch.inference_mode():
                                y_predicted, _, _ = mod(X_in = X_val, covariates_idx = covariates_idx_val, expr = expr_val)

                            emd_loss_tot += loss_fn(y_predicted, y_val).detach().cpu().item()
                            del expr_val, X_val, covariates_idx_val, y_predicted, loss_fn
                            torch.cuda.empty_cache()

                    emd_loss_k.append(emd_loss_tot)
                    res.loc[res.shape[0], :] = [max_epochs, max_lr, train_batch, k, emd_loss_tot, 
                                               regularization_params['vae_scaling_KL']]
                    
                    trainers_k[k] = trainer
                    io.write_pickled_object(trainers_k, os.path.join(data_path, 'processed', 'Kang_trainers_k.pickle'))
                    res.to_csv(os.path.join(data_path, 'processed', 'Kang_k_fold_validation_results.csv'))
                    
            emd_loss_mean = np.nanmean(emd_loss_k)
            if emd_loss_mean < best_emd_loss:
                best_emd_loss = emd_loss_mean
                for k, trainer in trainers_k.items():
                    io.write_pickled_object(trainer, os.path.join(models_path, 'Kang_best_trainer_' + str(k) + '.pickle'))
            if run:
                del trainers_k, trainer, mod
                torch.cuda.empty_cache()


# In[120]:


isinstance(np.nan, float)


# In[53]:


# files = [os.path.join(data_path, 'processed', 'Kang_k_fold_validation_results.csv'), 
#          os.path.join(models_path, 'Kang_best_trainer_*'), 
#          os.path.join(data_path, 'processed', 'Kang_trainers_k.pickle')]
# for i, f in enumerate(files):
#     print('rm ' + f)
#     if i == len(files) - 1:
#         print('')
#         print('')


# In[6]:


# test = io.read_pickled_object(os.path.join(data_path, 'processed', 'full_run_Kang_trainers_k.pickle'))
# res = pd.read_csv(os.path.join(data_path, 'processed', 'full_run_Kang_k_fold_validation_results.csv'), index_col = 0)


#!/usr/bin/env python
# coding: utf-8

# Now that we have shown that the k-fold models generalize better to the test data than a random baseline, we will generate a new model with the best hyperparameters identified, but trained on the training + validation data. This is simply to have one coherent model for additional downstream assessments. It also should give the model better power.

# In[1]:


import os

import pandas as pd
import scanpy as sc

import torch
import torch.nn as nn
import numpy as np
from geomloss import SamplesLoss
from sklearn.metrics import normalized_mutual_info_score
from scipy import stats
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(1, '../.')
from Kang_utils import get_best_hyperparams


# In[2]:


import sys
sclembas = '/home/hmbaghda/Projects/scLEMBAS'
sys.path.insert(1, os.path.join(sclembas))
from scLEMBAS import io
from scLEMBAS.model.scl import SignalingModel
from scLEMBAS.model.train import TrainSC
from scLEMBAS.preprocess import discriminator_weight_curve, embed_tf_activity


# In[3]:


seed = 888
device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = '/nobackup/users/hmbaghda/scLEMBAS/analysis'
models_path = os.path.join(data_path, 'processed', 'models')


# In[4]:


adata = sc.read_h5ad(os.path.join(data_path, 'processed', 'kang_expr_scored.h5ad'))
sn_ppis = pd.read_csv(os.path.join(data_path, 'processed', 'Kang_sn_ppis.csv'), index_col = 0)

tf_adata = io.read_tfad(os.path.join(data_path, 'processed', 'Kang_tf_activity.h5ad'))
tf_adata.obs['condition'] = tf_adata.obs['stim'].astype(str) + '^' + tf_adata.obs['seurat_annotations'].astype(str)

# ensures correct order of test data
# note, this already saved in order, matching the mod.y_out columns
tf_adata = tf_adata[:, sorted(tf_adata.var_names)] 


# ### Data

# In[5]:


source_label = 'source_genesymbol'
target_label = 'target_genesymbol'
weight_label = 'mode_of_action'
stimulation_label = 'consensus_stimulation'
inhibition_label = 'consensus_inhibition'


# In[6]:


data_split_path = os.path.join(os.path.join(data_path, 'processed', 'data_split_barcodes'))
fold_keys = ['train_cells', 'val_cells', 'train_cond', 'val_cond']
k_fold_cells = {}
for k in range(5):
    k_fold_cells[k] = {fk: open(os.path.join(data_split_path, 'kang_' + str(k) + '_' + fk + '.txt')).read().splitlines()
                      for fk in fold_keys}


# ### Params

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
                'cat_max_norm': 100 # 1 
                } 
vae_params = {'vae_batch_momentum': 0.01, 'vae_layer_norm': False, 'vae_dropout_rate': 0.1,
              'vae_activation_fn': nn.LeakyReLU,
              'vae_n_hidden_nodes': vae_n_hidden_nodes, 
              'vae_var_min': 1e-4}
bionet_params = {**bionet_params, **vae_params}

# training parameters
other_params_default = {'network_noise_scale': 10, 'gradient_noise_scale': 1e-9, 
               'test_batch_size': np.nan}
spectral_radius_params = {'n_probes_spectral': 5, 
                          'power_steps_spectral': 5, #50, 
                          'subset_n_spectral': 5} #10}
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


# In[9]:


# max_epochs_all = [600, 900, 1200]
# mpw_all = np.arange(5, 8.01, 0.25)
# b_all = np.arange(0.4, 1.21, 0.2)

# combs_all = list(itertools.product(max_epochs_all, mpw_all, b_all))
# max_lr = 0.0001
# train_batch = 512

# subset = 0.2 # 20% of grid

# combs = []
# counter = 0
# while ((900, 5.0, 0.4) not in combs) or ((1200, 5.0, 0.4) not in combs) or ((600, 5.0, 0.4) not in combs):
#     np.random.seed(seed + counter)
#     choice_idx = sorted(np.random.choice(range(len(combs_all)), int(np.floor(subset*len(combs_all)))))
#     combs = [comb for idx, comb in enumerate(combs_all) if idx in choice_idx]
#     counter += 1


# In[17]:


max_epochs_all = [600, 900, 1200]
mpw_all = np.arange(5, 8.01, 0.25)
b_all = np.arange(0.4, 1.21, 0.2)

combs_all = list(itertools.product(mpw_all, b_all))

subset = 0.3 # fraction of grid

combs = []
counter = 0

while (5.0, 0.4) not in combs:
    np.random.seed(seed + counter)
    choice_idx = sorted(np.random.choice(range(len(combs_all)), int(np.floor(subset*len(combs_all)))))
    combs = [comb for idx, comb in enumerate(combs_all) if idx in choice_idx]
    counter += 1
combs = [(x, y[0], y[1]) for x, y in itertools.product(max_epochs_all, combs)]

max_lr = 0.0001
train_batch = 512


# In[19]:


if os.path.isfile('/nobackup/users/hmbaghda/trash/res.csv'):
    res = pd.read_csv('/nobackup/users/hmbaghda/trash/res.csv', index_col = 0)
else:
    res = pd.DataFrame(columns = ['max_epochs', 'max_penalty_weight', 'b', 'disc_loss', 'nmi_global', 'nmi_cat'])
    


# In[ ]:


for hyperparams in combs:
    max_epochs, max_penalty_weight, b = hyperparams
    if not (res.iloc[:, :3] == list(hyperparams)).all(axis=1).any():
        print('epochs: {} | max penalty: {:.1f} | b power: {:.2f}'.format(max_epochs, max_penalty_weight, b))
        k = 'all'
        if k == 'all':
            test_cells = open(os.path.join(data_path, 'processed', 'data_split_barcodes', 'kang_test.txt')).read().splitlines()
            train_cells = [barcode for barcode in tf_adata.obs_names if barcode not in test_cells]
            val_cells = test_cells
        else:
            train_cells = k_fold_cells[k]['train_cells']
            test_cells = k_fold_cells[k]['val_cells']
            val_cells = test_cells



#         frac_no_adv = 0 # allow discriminator to fully learn the first fraction of epochs
#         n_no_adv = int(max_epochs*frac_no_adv)
        min_penalty_weight = 0.1
#         discriminator_penalty_weight = [min_penalty_weight/10]*n_no_adv
        discriminator_penalty_weight = discriminator_weight_curve(n_epochs = max_epochs,
                                                                       min_penalty_weight = min_penalty_weight,
                                                                       max_penalty_weight = max_penalty_weight, 
                                                                      a = 1, 
                                                                      b = b, 
                                                                          curve_type = 'power')

        lr_params = generate_lr_params(n_epochs = max_epochs, max_lr = max_lr, lr_scaling_factor = 10, lr_decay = 0.9)
        discriminator_params = generate_discriminator_params(n_epochs = max_epochs, max_lr = max_lr, 
                              discriminator_penalty_weight = discriminator_penalty_weight, 
                              lr_scaling_factor = 10, lr_decay = 0.9)
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
        mod = trainer.train_model(verbose = False)


        stim_map = {'STIM': 1, 'CTRL': 0}
        rev_stim = {'STIM': 'CTRL', 'CTRL': 'STIM'}
        # rev_stim_map = {k: int(not bool(v)) for k,v in stim_map.items()}

        cov_idx_map = dict(zip(mod.signaling_network.covariates['seurat_annotations'], 
                               mod.signaling_network.covariates_idx['seurat_annotations']))
        cov_rev_map = {v:k for k,v in cov_idx_map.items()}


        only_within_celltype = True # only change stim within a cell type
        full_expr, full_X, full_covariates = None, None, None

        for cond in tf_adata.obs.loc[test_cells, :].condition.unique():
            stim, ct = cond.split('^')

            if only_within_celltype:
                train_cells_cond = tf_adata.obs[(tf_adata.obs.index.isin(train_cells)) & 
                                                (tf_adata.obs['condition'] == rev_stim[stim] + '^' + ct)].index.tolist()
            else:
                train_cells_cond = train_cells

            expr_test = mod.df_to_tensor(mod.expr.loc[train_cells_cond, :])

            X_test_df = pd.DataFrame(data = {'IFNB1': [stim_map[stim]]*len(train_cells_cond)})
            X_test = mod.df_to_tensor(X_test_df)

            covariates_idx_test = torch.tensor([cov_idx_map[ct]]*len(train_cells_cond), 
                                               device = mod.device, dtype = torch.int64).view(-1,1)

            if full_expr is None:
                full_expr = expr_test
            else: 
                full_expr = torch.cat((full_expr, expr_test), dim = 0)

            if full_X is None:
                full_X = X_test
            else: 
                full_X = torch.cat((full_X, X_test), dim = 0)

            if full_covariates is None:
                full_covariates = covariates_idx_test
            else: 
                full_covariates = torch.cat((full_covariates, covariates_idx_test), dim = 0)    


        mod.eval()
        with torch.inference_mode():
            y_predicted, Y_full, biases = mod(X_in = full_X, covariates_idx = full_covariates, expr = full_expr)
            bias_global, bias_mu, bias_log_sigma_squared = biases
            bias_sigma = torch.exp(bias_log_sigma_squared/2.) + mod.signaling_network.vae.var_min

            bias_cats = torch.zeros_like(bias_global.T, device = mod.device, dtype = mod.dtype)
            for cat_group_idx in range(full_covariates.shape[1]):
                cat_group = mod.signaling_network._cat_group_idx[cat_group_idx]
                mod.signaling_network.cat_embeddings[cat_group].weight.data.masked_fill_(mask = mod.signaling_network.cat_embeddings_mask[cat_group], 
                                                                            value = 0.0)
                bias_cats += mod.signaling_network.cat_embeddings[cat_group](full_covariates[:,cat_group_idx]).T
            bias_tot = bias_global.T + bias_cats    

        discriminator = trainer.discriminator['discriminators']['seurat_annotations']
        loss_fn = nn.CrossEntropyLoss()

        discriminator.eval()
        with torch.inference_mode():
            discriminator_prediction = discriminator(bias_global)
            discriminator_loss = loss_fn(discriminator_prediction, full_covariates.view(-1)).detach().cpu().item()


        full_covariates = full_covariates.detach().cpu().numpy()
        bias_global = bias_global.detach().cpu().numpy()
        bias_tot = bias_tot.detach().cpu().numpy().T

        obs = pd.DataFrame(full_covariates)
        obs.columns = ['seurat_annotations']
        obs.seurat_annotations = obs.seurat_annotations.map(cov_rev_map)

        # full model
        bias_adata = sc.AnnData(X = bias_global, obs = obs)
        embed_tf_activity(bias_adata, scanpy_pca = False, cluster_col_name = 'leiden', resolution = 1)
        nmi_global = normalized_mutual_info_score(bias_adata.obs.leiden, bias_adata.obs.seurat_annotations)

        # full model -- categorical information added
        bias_tot = sc.AnnData(X = bias_tot, obs = obs)
        embed_tf_activity(bias_tot, scanpy_pca = False, cluster_col_name = 'leiden', resolution = 1)
        nmi_cat = normalized_mutual_info_score(bias_tot.obs.leiden, bias_tot.obs.seurat_annotations)

        res.loc[res.shape[0], :] = [max_epochs, max_penalty_weight, b, 
                                    discriminator_loss, nmi_global, nmi_cat]
        res.to_csv('/nobackup/users/hmbaghda/trash/res.csv')

        if nmi_cat/nmi_global >= 5: 
            fn = 'e{}_p{:.1f}_b{:.2f}'.format(max_epochs, max_penalty_weight, b) + '.csv'
            fn = os.path.join('/nobackup/users/hmbaghda/trash/', fn)
            trainer.stats['train'].to_csv(fn)
            


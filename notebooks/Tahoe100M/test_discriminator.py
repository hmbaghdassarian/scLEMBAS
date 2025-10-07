#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse

def str_to_bool(value):
    """Convert argument string to boolean."""
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes', 'y'):
        return True
    elif value.lower() in ('false', '0', 'no', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (true/false or 1/0).")
        
def int_or_str(val):
    try:
        return int(val)
    except ValueError:
        return val

parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int_or_str, required=True, help="Filename index")
parser.add_argument("--subset_size", type=float, required=True)

parser.add_argument("--max_epochs", type=int, required=True)

parser.add_argument("--pert_spectral_norm", type = str_to_bool, required = True)
parser.add_argument("--pert_dropout", type=float, required=True)
parser.add_argument("--pert_n_layers", type = int, default = 4)
parser.add_argument("--pert_max_lr", type=float, required=True)

parser.add_argument("--smooth_labels", type = str_to_bool, required = True)
parser.add_argument("--pert_epsilon_smooth", type = float, required = True)

# ########################################################################
args = parser.parse_args()
fn = str(args.index)
subset_size = args.subset_size
max_epochs = args.max_epochs
pert_spectral_norm = args.pert_spectral_norm
pert_dropout = args.pert_dropout
pert_n_layers = args.pert_n_layers
pert_max_lr = args.pert_max_lr
smooth_labels = args.smooth_labels
pert_epsilon_smooth = args.pert_epsilon_smooth

#python test_discriminator.py --index 1 --subset_size 0.15 --max_epochs 600 --pert_spectral_norm true --pert_n_layers 4 --pert_max_lr 1e-3 --smooth_labels True --pert_epsilon_smooth 0.033


# In[2]:


import numpy as np
import math
subset = True


# In[ ]:


# pert_dropout = 0.1
# pert_n_layers = 4
# pert_max_lr = 1e-3
# pert_spectral_norm = True
# smooth_labels = True
# pert_epsilon_smooth = 0.033

# # LR params
# pert_max_lr = 1e-3

# max_epochs = 20

# subset_size = 0.15


# In[3]:


lr_scaling_factor = 10
lr_decay = 0.9
n_restarts = 3

batch_params = {
    'train_batch_size': int(512*8), 
    'test_batch_size': int(512*8), 
    'validation_batch_size': np.nan
}


max_batch_scaler = 24

if subset:
    if subset_size < 0.02:
        batch_scaler = 2
    else:
        batch_scaler = int(8 * round(math.ceil(subset_size / 0.05) * 0.05, 2) / 0.05)
        batch_scaler = min(batch_scaler, max_batch_scaler)

    batch_params['train_batch_size'] = int(512*batch_scaler)
    batch_params['test_batch_size'] = int(512*batch_scaler)


# In[4]:


index = 'dev'
noadv = False
vae_scaling_KL = 5e-3
n_cat_discriminator_train = 5
cat_dropout = 0.1
n_adversarial_start = 200
main_max_lr = 2e-3
gen_max_lr = 2.75e-4
cat_max_lr = 1e-3
cat_max_penalty_weight = 11
generator_dropout_rate = 0.7
n_layers_vae = 3
cat_bias_pert_scaler = 0
cat_pert_method = 'orthogonality'
cat_bias_lambda_L2 = 1e-2
spectral_loss_factor = 0
uniform_lambda_L2 = 0
cat_pert_pert_label = False
n_pert_discriminator_train = 5

contrastive_loss_scaler = [1]
contrastive_loss_type = ['sc_actual']
contrastive_percentile = 0.3
contrastive_triplet_margin_frac = 0.1


# In[5]:


import os
import time
import math

from tqdm import tqdm

import pandas as pd
import numpy as np
import scanpy as sc

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn

import sys
sclembas = '/home/hmbaghda/Projects/scLEMBAS'
sys.path.insert(1, os.path.join(sclembas))
from scLEMBAS import io
from scLEMBAS import preprocess as pp 
from scLEMBAS.model.train import TrainSC
from scLEMBAS.model.scl import SignalingModel

import Tahoe_utils as Tu


# In[6]:


subset = True


# In[7]:


n_cores = 30
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)

seed = 888
mod_seed = 888
data_path = '/nobackup/users/hmbaghda/scLEMBAS/analysis'
author = 'Tahoe100M'

device = "cuda" if torch.cuda.is_available() else "cpu"


# Load data:

# In[8]:


sn_ppis = pd.read_csv(os.path.join(data_path, 'processed', author + '_sn_ppis.csv'), 
                     index_col = 0)
source_label = 'source_genesymbol'
target_label = 'target_genesymbol'
weight_label = 'mode_of_action'
stimulation_label = 'consensus_stimulation'
inhibition_label = 'consensus_inhibition'

tf_adata = io.read_tfad(os.path.join(data_path, 'processed', author + '_consensus_tf_activity.h5ad'))
adata = sc.read_h5ad(os.path.join(data_path, 'processed', author + '_expr_in.h5ad'))
expr = adata.to_df().copy()

# basic formatting checks
if not np.all(tf_adata.var_names == sorted(tf_adata.var_names)):
    raise ValueError('Ensure TF adata features are sorted on input')
    
if not np.all(adata.obs_names == tf_adata.obs_names):
    raise ValueError('Ensure gene expression and TF activity sample features are orderd the same')
    
if len(set(tf_adata.obs.drug)) != len(tf_adata.obs.drug.cat.categories):
    raise ValueError('Make sure only present perturbations are in the categorical columns')
    
if len(set(tf_adata.obs.cell_line)) != len(tf_adata.obs.cell_line.cat.categories):
    raise ValueError('Make sure only present cell lines are in the categorical columns')


# In[10]:


cat_col, pert_col = 'cell_line', 'drug'


# # Train/test split:

# In[16]:


# train_split, test_split = Tu.Tahoe100M_split(tf_adata,
#                                           train_frac = 0.9, 
#                                           min_drug_frac = 0.7, 
#                                           min_cell_line_frac = 0.7, 
#                                           exclude_control = True, 
#                                           max_attempts = 1000, 
#                                           seed = seed)

# train_cells = train_split['barcodes']
# test_cells = test_split['barcodes']

# cell_line_counts = pd.DataFrame({
#     'train': train_split['cell_line_counts'],
#     'test': test_split['cell_line_counts']
# }).sort_values(by = 'train', ascending = True)

# drug_counts = pd.DataFrame({
#     'train': train_split['drug_counts'],
#     'test': test_split['drug_counts']
# }).sort_values(by = 'train', ascending = True)


from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    tf_adata.obs,
    test_size=0.1,
    stratify=tf_adata.obs[pert_col],   # ensures balanced category proportions
    random_state=seed
)

train_cells = train_df.index.tolist()
test_cells = test_df.index.tolist()


# In[17]:


if subset:
    np.random.seed(seed)
    train_cells = list(np.random.choice(
            train_cells,
            size = int(np.round(len(train_cells) * subset_size)),
            replace = False
        ))

    tf_adata = tf_adata[tf_adata.obs_names.isin(train_cells + test_cells),:].copy()
    adata = adata[tf_adata.obs_names, :].copy()
    expr = adata.to_df().copy()


# In[18]:


cat_col = 'cell_line'
pert_col = 'drug'


# In[20]:


tf_adata[train_cells, :].obs[pert_col].value_counts().unique()


# **Note: As mentioned in Notebook 00A, when splitting the data into train-test splits, conditions will remain balanced, but drug/cell lines will not be completely balanced (due to OOD, cannot ensure the drug and cell line split is the same as the condition split).** 

# # Hyper-parameters:

# In[21]:


def generate_lr_params(n_epochs, 
                       max_lr, 
                       lr_scaling_factor=10, 
                       lr_decay=0.75,
                       n_restarts=4,
#                        n_cat_discriminator_train=5,
#                        n_pert_discriminator_train=5,
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

#    DEPRECATED: this was necessary when generator trained every n_discriminator_train epochs
#                rather than discriminator training n_discriminator_train times every epoch
#     elif role in 'generator':
#         # Generator steps on any epoch divisible by either cat or pert discriminator schedule
#         effective_epochs = range(n_adversarial_start, n_epochs)
#         n_steps = sum(
#             (e % n_cat_discriminator_train == 0) or (e % n_pert_discriminator_train == 0)
#             for e in effective_epochs
#         )

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


# In[22]:


projection_amplitude_in = 10
projection_amplitude_out = 1

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
    'min_network_noise': 0.0015, #0.0025, 
    'gradient_noise_scale': 1e-9, 
    'include_gradient_noise_vae': True, 
    'include_gradient_noise_embedding': True, 
    'constant_gradient_noise': True
    
}


# In[23]:


loss_scaler = 100
prediction_loss_fn = torch.nn.MSELoss(reduction='mean')


batch_params = {
    'train_batch_size': int(512*8), 
    'test_batch_size': int(512*8), 
    'validation_batch_size': np.nan
}


max_batch_scaler = 24

if subset:
    if subset_size < 0.02:
        batch_scaler = 2
    else:
        batch_scaler = int(8 * round(math.ceil(subset_size / 0.05) * 0.05, 2) / 0.05)
        batch_scaler = min(batch_scaler, max_batch_scaler)

    batch_params['train_batch_size'] = int(512*batch_scaler)
    batch_params['test_batch_size'] = int(512*batch_scaler)
# max_epochs = 600

lr_scaling_factor = 10
lr_decay = 0.9
n_restarts = 4
lr_params = generate_lr_params(n_epochs = max_epochs, 
                               max_lr = main_max_lr, 
                               lr_scaling_factor = lr_scaling_factor, 
                               lr_decay = lr_decay,
                               n_restarts = n_restarts,
                               n_adversarial_start = np.nan, 
#                                n_cat_discriminator_train = np.nan, 
#                                n_pert_discriminator_train = np.nan,
                               role = 'scl')

# reset_state = False # DEPRECATED
# initialize_fc = True # DEPRECATED


# In[24]:


bionet_params['cat_max_norm'] = 100
regularization_params = {
    'input_lambda_L2': 0, # irrelevant because setting the requires grad to False
    
    'bn_weights_lambda_L2': 1e-7,
    'moa_lambda_L1': 1e2,
    'uniform_lambda_L2': uniform_lambda_L2, #0, #1e-7, 
    'uniform_min': -1/projection_amplitude_out,
    'uniform_max': 1/projection_amplitude_out,
    'adj_scaling_KL': 0,  # using uniform/bn_weights already
    'adj_prior_mu': 0, # irrelevant because adj_scaling_KL is 0
    'adj_prior_sigma': 0.2, # irrelevant because adj_scaling_KL is 0
    
    'output_weights_lambda_L2': 1e-7,
    'output_bias_lambda_L2': 1e-7,
    
    'spectral_loss_factor': spectral_loss_factor,
    
    
    'global_bias_lambda_L2': 0, # using KL divergence instead
    'global_bias_lambda_L1': 0, # using KL divergence instead
    'cat_bias_lambda_L2': cat_bias_lambda_L2, # 1e-4, # allow for generalization (not collapsing on perturbation)
    'cat_bias_lambda_L1': 0, # using cat max norm
}


contrastive_loss_params = {
    'methods': contrastive_loss_type, 
    'lambda_scalers': contrastive_loss_scaler, 
    'understimate_only': True, # only for _bulk_actual
    'min_percentile': contrastive_percentile, # only for _sc
    'triplet_margin_frac': contrastive_triplet_margin_frac, # for sc only 
}

cat_pert_params = {'regularization_scaler': cat_bias_pert_scaler, 
                       'method': cat_pert_method, 
                       'per_label': cat_pert_pert_label, 
                       'include_adjacency': False, 
                       'temperature': 0.1
                      }


# In[25]:


training_params = {
    **lr_params, 
    **batch_params, 
    **regularization_params, 
    **spectral_radius_params,
    **noise_params
}

training_params['prediction_loss_fn_scaler'] = loss_scaler


# VAE:

# In[26]:


# building
# n_layers_vae = 1
n_nodes = len(set(sn_ppis[source_label].tolist() + sn_ppis[target_label].tolist()))
vae_n_hidden_nodes = list(np.round(np.linspace(adata.shape[1], n_nodes, n_layers_vae + 2)).astype(int)[1:-1])

# generator_dropout_rate = 0.7
vae_mod_params = {
    'vae_batch_momentum': 0.01, 
    'vae_layer_norm': False, 
    'vae_dropout_rate': generator_dropout_rate,
    'vae_activation_fn': nn.LeakyReLU,
    'vae_n_hidden_nodes': vae_n_hidden_nodes, 
    'vae_var_min': 1e-4

} 
bionet_params = {**bionet_params, **vae_mod_params}

# training
vae_params = {
    'prior_mu': 0, 
    'prior_sigma': 1,
    'scaling_KL': vae_scaling_KL, #1e-2, 
    'lambda_l2': 1e-5, 
    'optimizer': torch.optim.Adam
}

 
n_restarts_adversarial = 4
vae_lr_params = generate_lr_params(n_epochs = max_epochs,
                                   max_lr = gen_max_lr, #max_lr,
                                   lr_scaling_factor = lr_scaling_factor, 
                                   lr_decay = lr_decay,
                                   n_restarts = n_restarts_adversarial,
                                   n_adversarial_start = n_adversarial_start,
#                                n_cat_discriminator_train = n_cat_discriminator_train, 
#                                n_pert_discriminator_train = n_pert_discriminator_train,
                                   role = 'generator')
vae_params = {**vae_params, **vae_lr_params}
del vae_params['max_epochs']


# Discriminator:

# In[27]:


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
    'smooth_labels': smooth_labels, 
    'epsilon_smooth': np.nan
}


# In[28]:


# architecture -- pert >> cat bc harder classification problem
cat_n_layers_disc = 3
cat_disc_n_hidden_nodes = list(np.round(np.linspace(n_nodes, 
                                                    train_df[cat_col].nunique(),
                                                    cat_n_layers_disc + 2)).astype(int)[1:-1])
cat_discriminator_params = discriminator_params.copy()
cat_discriminator_params['n_hidden_nodes'] = cat_disc_n_hidden_nodes
cat_discriminator_params['dropout_rate'] = cat_dropout

cat_discriminator_params['spectral_norm'] = True
cat_discriminator_params['discriminator_lambda_L2'] = 0


pert_n_layers_disc = pert_n_layers
pert_disc_n_hidden_nodes = list(np.round(np.linspace(n_nodes, 
                                                     train_df[pert_col].nunique(), 
                                                     pert_n_layers_disc + 2)).astype(int)[1:-1])

# add 3 additional "starting" layers since classifying perturbation is difficult
pert_disc_n_hidden_nodes = [pert_disc_n_hidden_nodes[0]]*3 + pert_disc_n_hidden_nodes

pert_discriminator_params = discriminator_params.copy()
pert_discriminator_params['n_hidden_nodes'] = pert_disc_n_hidden_nodes
pert_discriminator_params['dropout_rate'] = pert_dropout

pert_discriminator_params['spectral_norm'] = pert_spectral_norm
if pert_spectral_norm:
    pert_discriminator_params['discriminator_lambda_L2'] = 0

cat_discriminator_params['epsilon_smooth'] = 1/tf_adata.obs.cell_line.nunique()
pert_discriminator_params['epsilon_smooth'] = pert_epsilon_smooth #1/tf_adata.obs.drug.nunique()


# In[29]:


# adverserial penalty curve
# cat_max_penalty_weight = 11
cat_b_adv = 2.5
pert_max_penalty_weight = 10 #15
pert_b_adv = 2 #10

if n_adversarial_start < max_epochs:
    cat_discriminator_params['discriminator_penalty_weight'] = pp.discriminator_weight_curve(
        n_epochs = max_epochs - n_adversarial_start,
        min_penalty_weight = 0.01,
        max_penalty_weight = cat_max_penalty_weight,
        a = 1,
        b = cat_b_adv, 
        curve_type = 'power')

    n_pert_train = int(max_epochs/2)
    pert_discriminator_penalty_weight = [0] * n_pert_train
    pert_discriminator_penalty_weight += pp.discriminator_weight_curve(
        n_epochs = max_epochs - n_pert_train - n_adversarial_start,
        min_penalty_weight = 1e-5,
        max_penalty_weight = pert_max_penalty_weight,
        a = 1,
        b = pert_b_adv, 
        curve_type = 'power')
    pert_discriminator_params['discriminator_penalty_weight'] = pert_discriminator_penalty_weight
else:
    cat_discriminator_params['discriminator_penalty_weight'] = [0]*max_epochs
    pert_discriminator_params['discriminator_penalty_weight'] = [0]*max_epochs




# # DEPRECATED: when n_discriminator_train trained generator less frequently instead of disc more frequently
# def expand_elements(lst, n_discriminator_train):
#     return [x for x in lst for _ in range(n_discriminator_train)]
# n_adj = 1 if n_cat_discriminator_train > 1 else 0
# cat_discriminator_penalty_weight = pp.discriminator_weight_curve(
#     n_epochs = int((max_epochs - n_adversarial_start)/n_cat_discriminator_train) - n_adj,
#     min_penalty_weight = 0.1,
#     max_penalty_weight = cat_max_penalty_weight,
#     a = 1,
#     b = cat_b_adv, 
#     curve_type = 'power')
# cat_discriminator_penalty_weight = expand_elements(cat_discriminator_penalty_weight, n_cat_discriminator_train)
# cat_discriminator_penalty_weight = [0]*n_adj*n_cat_discriminator_train + cat_discriminator_penalty_weight
# cat_discriminator_params['discriminator_penalty_weight'] = cat_discriminator_penalty_weight

# n_pert_train = 300
# pert_discriminator_penalty_weight = [0] * n_pert_train

# n_adj = n_pert_discriminator_train if n_pert_discriminator_train > 1 and n_pert_train == 0 else 0
# pdpw2 = pp.discriminator_weight_curve(
#     n_epochs = int((max_epochs - n_pert_train - n_adversarial_start)/n_pert_discriminator_train) - n_adj,
#     min_penalty_weight = 1e-5,
#     max_penalty_weight = pert_max_penalty_weight,
#     a = 1,
#     b = pert_b_adv, 
#     curve_type = 'power')
# pdpw2 = expand_elements(pdpw2, n_pert_discriminator_train)
# pert_discriminator_penalty_weight += [0]*n_adj*n_pert_discriminator_train + pdpw2


# pert_discriminator_params['discriminator_penalty_weight'] = pert_discriminator_penalty_weight


# In[30]:


# discriminator LRs

# categorical
discriminator_lr_params = generate_lr_params(
    n_epochs = max_epochs,
    max_lr = cat_max_lr,
    lr_scaling_factor = lr_scaling_factor, 
    lr_decay = lr_decay,
    n_restarts = n_restarts_adversarial,
    n_adversarial_start = n_adversarial_start, 
#     n_pert_discriminator_train = np.nan,
#     n_cat_discriminator_train = n_cat_discriminator_train,
    role = 'cat_discriminator')
del discriminator_lr_params['max_epochs']

cat_discriminator_params = {**cat_discriminator_params, **discriminator_lr_params}

# perturbation
discriminator_lr_params = generate_lr_params(
    n_epochs = max_epochs,
    max_lr = pert_max_lr,
    lr_scaling_factor = lr_scaling_factor, 
    lr_decay = lr_decay,
    n_restarts = n_restarts_adversarial,
    n_adversarial_start = 0, 
#     n_pert_discriminator_train = n_pert_discriminator_train,
#     n_cat_discriminator_train = np.nan,
    role = 'pert_discriminator')
del discriminator_lr_params['max_epochs']

pert_discriminator_params = {**pert_discriminator_params, **discriminator_lr_params}


# # Build model and trainer

# In[68]:


# input stimulation
X_in = pd.get_dummies(tf_adata.obs.drug).astype(int)
X_in.drop(columns = 'DMSO_TF', inplace = True) # all 0s


# In[69]:


# lr_mod = SignalingModel(
#     net = sn_ppis,
#     X_in = X_in,
#     y_out = tf_adata.to_df().copy(), 
#     expr = expr, 
#     covariates = tf_adata.obs.copy(),
#     categorical_covariate_keys = ['cell_line'],
#     projection_amplitude_in = projection_amplitude_in, 
#     projection_amplitude_out = projection_amplitude_out,
#     weight_label = weight_label, source_label = source_label, target_label = target_label,
#     bionet_params = bionet_params, 
#     dtype = torch.float32, device = device, seed = mod_seed)

# lr_mod.input_layer.weights.requires_grad = False # don't learn scaling factors for the ligand input concentrations
# lr_mod.signaling_network.prescale_weights(target_radius = target_spectral_radius) # spectral radius



# lr_trainer = TrainSC(
#     mod = lr_mod,
#     prediction_optimizer = torch.optim.Adam,
#     prediction_loss_fn = prediction_loss_fn, 
#     per_condition_loss = per_condition_loss,
#     n_adversarial_start = n_adversarial_start, 
#     n_cat_discriminator_train = n_cat_discriminator_train,
#     n_pert_discriminator_train = n_pert_discriminator_train,
#     gradient_ascent = True,
#     cat_discriminator_params = cat_discriminator_params,
#     pert_discriminator_params = pert_discriminator_params,
#     vae_params = vae_params,
#     hyper_params = training_params,
#     train_split = {'train': train_cells, 'test': test_cells, 'validation': None}, 
#     train_seed = mod_seed, 
#     track_test = True,
#     track_validation = False, 
#     n_eval_cells = np.nan, 
#     n_eval_bootstrap = np.nan
# )

# import collections
# from tqdm import trange
# lrs = collections.defaultdict(list)
# for e in trange(lr_trainer.hyper_params['max_epochs']):
#     lr_trainer._run_adv = (lr_trainer.n_adversarial_start <= e)
    
#     lrs['scl'].append(lr_trainer.prediction_optimizer.param_groups[0]['lr'])
#     lrs['vae'].append(lr_trainer.vae_learning['optimizer'].param_groups[0]['lr'])
#     lrs['cat_discriminator'].append(lr_trainer.cat_discriminator['optimizer'].param_groups[0]['lr'])
#     lrs['pert_discriminator'].append(lr_trainer.pert_discriminator['optimizer'].param_groups[0]['lr'])


#     lr_trainer.lr_scheduler.step()
#     if lr_trainer._run_adv:
#         lr_trainer.cat_discriminator['lr_scheduler'].step()
#         lr_trainer.pert_discriminator['lr_scheduler'].step()
#         lr_trainer.vae_learning['lr_scheduler'].step()
        
# lrs = pd.DataFrame(lrs)

# ncols = lrs.shape[1]
# fig, ax = plt.subplots(ncols = ncols, figsize = (5.1*ncols, 5))

# for (j, col) in enumerate(lrs.columns):
#     sns.lineplot(lrs[col], ax = ax[j])
#     ax[j].set_title(col)
    
# fig.tight_layout()
# ;
# del lr_trainer


# In[86]:


mod = SignalingModel(
    net = sn_ppis,
    X_in = X_in,
    y_out = tf_adata.to_df().copy(), 
    expr = expr, 
    covariates = tf_adata.obs.copy(),
    categorical_covariate_keys = ['cell_line'],
    projection_amplitude_in = projection_amplitude_in, 
    projection_amplitude_out = projection_amplitude_out,
    weight_label = weight_label, source_label = source_label, target_label = target_label,
    bionet_params = bionet_params, 
    dtype = torch.float32, device = device, seed = mod_seed)

mod.input_layer.weights.requires_grad = False # don't learn scaling factors for the ligand input concentrations
mod.signaling_network.prescale_weights(target_radius = target_spectral_radius) # spectral radius


trainer = TrainSC(
    mod = mod,
    prediction_optimizer = torch.optim.Adam,
    prediction_loss_fn = prediction_loss_fn, 
    per_condition_loss = False,
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
    train_split = {'train': train_cells, 'test': test_cells, 'validation': None}, 
    train_seed = mod_seed, 
    track_test = True,
    track_validation = False, 
    n_eval_cells = np.nan, 
    n_eval_bootstrap = np.nan
)
train_dataloader = trainer.train_dataloader
test_dataloader = trainer.test_dataloader


# In[87]:


from scLEMBAS.model.model_components import CatDiscriminator
from scLEMBAS.model.train import *
pert_discriminator_params_actual = pert_discriminator_params.copy()
pert_n_layers_disc = pert_n_layers
pert_disc_n_hidden_nodes = list(np.round(np.linspace(adata.n_vars, 
                                                     train_df[pert_col].nunique(), 
                                                     pert_n_layers_disc + 2)).astype(int)[1:-1])

# add 3 additional "starting" layers since classifying perturbation is difficult
pert_disc_n_hidden_nodes = [pert_disc_n_hidden_nodes[0]]*3 + pert_disc_n_hidden_nodes

pert_discriminator_params_actual['n_hidden_nodes'] = pert_disc_n_hidden_nodes

disc = CatDiscriminator(
    n_features_in = tf_adata.n_vars,
    n_labels = train_df[pert_col].nunique(), 
    dtype = trainer.mod.dtype, 
    device = device,
    batch_momentum = pert_discriminator_params_actual['batch_momentum'], 
    layer_norm = pert_discriminator_params_actual['layer_norm'],
    spectral_norm = pert_discriminator_params_actual['spectral_norm'],
    dropout_rate = pert_discriminator_params_actual['dropout_rate'],
    activation_fn = pert_discriminator_params_actual['activation_fn'], 
    bionet_activation = pert_discriminator_params_actual['bionet_activation'], 
    rnn_params = None,
    smooth_labels = pert_discriminator_params_actual['smooth_labels'],
    epsilon_smooth = pert_discriminator_params_actual['epsilon_smooth'],                                
    n_hidden_nodes = pert_discriminator_params_actual['n_hidden_nodes'], 
    seed = 888
)

optimizer = torch.optim.Adam(disc.parameters(), 
                            lr = pert_discriminator_params_actual['maximum_learning_rate'], 
                            weight_decay = 0)
lr_scheduler = WarmupCosineAnnealingWarmRestarts(
    optimizer = optimizer,
    T_0 = pert_discriminator_params_actual['lr_restart_epoch'],
    T_mul = pert_discriminator_params_actual['lr_restart_factor'], 
    gamma = pert_discriminator_params_actual['lr_decay'],
    eta_min = pert_discriminator_params_actual['minimum_learning_rate'],
    max_lr=pert_discriminator_params_actual['maximum_learning_rate'],
    warmup_steps = pert_discriminator_params_actual['warmup_epochs'],
    last_epoch = -1)

def to_target(X_in_, disc):
    target = X_in_.argmax(dim=1)
    no_pert = X_in_.sum(dim=1) == 0  
    target[no_pert] = disc.n_labels - 1 # -1 for indexing
    
    return target
    


# In[88]:


start_time = time.time()
# torch.autograd.set_detect_anomaly(True)


train_stats_cols = ['epoch', 'batch_index', 'pert_discriminator_learning_rate', #'iter_time', 
             'pert_discriminator_loss_prediction', 'pert_discriminator_param_reg_loss', 
             'pert_discriminator_loss_total']
# stats_train = np.empty((0, len(train_stats_cols)))

eval_stats_cols = ['epoch', 'batch_index', 'test_loss']
# stats_eval = np.empty((0, len(eval_stats_cols)))

stats_train = []
stats_eval = []


for e in trange(max_epochs):
    disc.train()
    cur_lr = optimizer.param_groups[0]['lr']
    utils.set_seeds(seed + e)
    for batch, (X_in_, y_out_, covariates_idx_, expr_) in enumerate(train_dataloader):
        X_in_, y_out_ = X_in_.to(device), y_out_.to(device) #covariates_idx_.to(self.mod.device), expr_.to(self.mod.device)
        
        optimizer.zero_grad()
        
        pred = disc(y_out_)
        target = to_target(X_in_, disc)
        pert_discriminator_loss_accuracy = disc.loss_fn(pred, target)    

        # discriminator regularization
        pert_discriminator_reg = disc.L2_reg(pert_discriminator_params['discriminator_lambda_L2'])
        pert_discriminator_loss = pert_discriminator_loss_accuracy + pert_discriminator_reg

        # discriminator optimization
        pert_discriminator_loss.backward()
        nn.utils.clip_grad_norm_(disc.parameters(), max_norm=100.0)

        optimizer.step()
        
        stats_train.append([
            e , batch, cur_lr, #time.time() - start_time,
            pert_discriminator_loss_accuracy.detach().item(),pert_discriminator_reg.detach().item(),  pert_discriminator_loss.detach().item()
        ])
        
        
        del target, pred, pert_discriminator_loss_accuracy, pert_discriminator_reg, pert_discriminator_loss
        del X_in_, y_out_, covariates_idx_, expr_
        utils.clear_memory()
        
        
    lr_scheduler.step()
    
    # eval
    if e % 25 == 0:
        disc.eval()
        for batch, (X_in_, y_out_, _, _) in enumerate(test_dataloader):
            X_in_, y_out_ = X_in_.to(device), y_out_.to(device)
            with torch.inference_mode():
                pred = disc(y_out_)
                target = to_target(X_in_, disc)
                eval_loss = disc.eval_loss_fn(pred, target)

            stats_eval.append([
                e+1, batch, 
                eval_loss.item()
            ])

            del X_in_, y_out_, pred, target, eval_loss
            utils.clear_memory()


# In[92]:


mins, secs = divmod(time.time() - start_time, 60)
print("Training ran in: {:.0f} min {:.2f} sec".format(mins, secs))


# In[93]:


stats_train = pd.DataFrame(stats_train, columns=train_stats_cols)
stats_eval = pd.DataFrame(stats_eval, columns=eval_stats_cols)

stats_train = stats_train.groupby('epoch').mean().reset_index()
stats_eval = stats_eval.groupby('epoch').mean().reset_index()


# In[101]:


import matplotlib.ticker as mticker

fig, ax = plt.subplots(ncols = 3, figsize = (12, 4))

i = 0
sns.lineplot(data = stats_train, x = 'epoch', y = 'pert_discriminator_learning_rate', ax = ax[i])
ax[i].set_ylabel('LR')

i = 1
viz_df = stats_train.drop(columns = ['batch_index', 'pert_discriminator_learning_rate']).melt(
    id_vars = 'epoch', var_name = 'loss_type', value_name='loss')

viz_df.loss_type = viz_df.loss_type.map({'pert_discriminator_loss_prediction': 'Prediction', 
                     'pert_discriminator_param_reg_loss': 'L2_reg', 
                     'pert_discriminator_loss_total': 'Total'})
viz_df.loss_type = pd.Categorical(viz_df.loss_type, 
               categories = ['L2_reg', 'Prediction', 'Total'], 
               ordered = True)

zeros = (
    viz_df.groupby("loss_type", observed=False)["loss"]
          .agg(lambda s: (s.size > 0) and np.all(s.values == 0))
)
zeros = zeros[zeros].index.values.tolist()  
viz_df = viz_df[~viz_df.loss_type.isin(zeros)].copy()
viz_df.loss_type = viz_df.loss_type.cat.remove_unused_categories().copy()
sns.lineplot(data = viz_df, x = 'epoch', y = 'loss', hue = 'loss_type', ax = ax[i])
ax[i].set_title('Train Loss Only')
# ax[i].yaxis.set_major_formatter(mticker.ScalarFormatter())
# ax[i].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax[i].ticklabel_format(style='plain', axis='y', useOffset=False, useMathText=False)
ax[i].yaxis.get_offset_text().set_visible(False)

cp = tf_adata[train_cells,:].obs[pert_col].value_counts(normalize = True).values
rand_loss = disc.random_loss(class_probs = cp, train_mode = True)
ax[i].axhline(y=rand_loss, color='dimgray', linestyle='--', label = 'random_loss')


i = 2
viz_df = stats_train[['epoch', 'pert_discriminator_loss_prediction']].copy()
viz_df.rename(columns = {'pert_discriminator_loss_prediction': 'loss'}, inplace = True)
viz_df['loss_type'] = 'train'

viz_df_2 = stats_eval[['epoch', 'test_loss']].rename(columns = {'test_loss': 'loss'})
viz_df_2['loss_type'] = 'test'
viz_df = pd.concat([viz_df, viz_df_2])
viz_df.loss_type = pd.Categorical(viz_df.loss_type, 
                                 categories = ['train', 'test'], 
                                 ordered = True)

sns.lineplot(data = viz_df, x = 'epoch', y = 'loss', 
             hue = 'loss_type', 
             palette = ["#ff7f0e", "#8c564b"], #,
             ax = ax[i])
cp = tf_adata[train_cells,:].obs[pert_col].value_counts(normalize = True).values
rand_loss = disc.random_loss(class_probs = cp, train_mode = False)
ax[i].axhline(y=rand_loss, color="#ffb877", linestyle='--', label = 'Random Train')

cp = tf_adata[test_cells,:].obs[pert_col].value_counts(normalize = True).values
rand_loss = disc.random_loss(class_probs = cp, train_mode = False)
ax[i].axhline(y=rand_loss, color="#b98a81", linestyle='--', label = 'Random Test')
ax[i].legend()

fig.suptitle("Training ran in: {:.0f} min {:.2f} sec".format(mins, secs))
fig.tight_layout()

fig.savefig(os.path.join(data_path, 'trash', '_'.join([fn, author, 'pert_disc.png'])), 
            dpi=300, bbox_inches='tight')


# In[ ]:


# import papermill as pm
# from nbconvert import HTMLExporter
# import nbformat
# import os

# input_notebook = 'test_visualize.ipynb' # in the current directory
# output_notebook = os.path.join(data_path, 'trash', fn +  '_' + author + '.ipynb')
# output_html = os.path.join(data_path, 'trash', fn + '_' + author + '.html')

# pm.execute_notebook(
#     input_path=input_notebook,
#     output_path=output_notebook,
#     parameters={"fn": fn}, 
#     kernel_name='python3'
# )

# nb = nbformat.read(output_notebook, as_version=4)
# html_exporter = HTMLExporter()
# html_exporter.exclude_input = True  # <-- hides code cells
# (body, _) = html_exporter.from_notebook_node(nb)

# with open(output_html, "w", encoding="utf-8") as f:
#     f.write(body)
    
# os.remove(output_notebook)


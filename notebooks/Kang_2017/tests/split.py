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
parser.add_argument("--run_type", type=str, required=True, help="Filename index")
# parser.add_argument("--seed", type=int, default=1, help="Filename index")
# parser.add_argument("--loo", type=str_to_bool, required=True, help="Leave-one-out flag (true/false or 1/0)")

parser.add_argument("--bn_weights_lambda_L2", type=float, default=1e-7, help="Adj matrix weights")
parser.add_argument("--uniform_lambda_L2", type=float, default=1e-7, help="Y_full regularization")
parser.add_argument("--cat_max_norm", type=float, default=100, help="Cat bias max norm")
parser.add_argument("--global_bias_lambda_L2", type=float, default=0, help="Tot bias regularization")
parser.add_argument("--cat_bias_lambda_L2", type=float, default=0, help="Tot bias regularization")
parser.add_argument("--vae_scaling_KL", type=float, default=1e-2, help="Tot bias regularization")
parser.add_argument("--global_bias_lambda_L1", type=float, default=0, help="Tot bias regularization")
parser.add_argument("--cat_bias_lambda_L1", type=float, default=0, help="Tot bias regularization")
parser.add_argument("--vae_prior_mu", type=float, default=0, help="Tot bias regularization")
parser.add_argument("--vae_prior_sigma", type=float, default=1, help="Tot bias regularization")
parser.add_argument("--adj_scaling_KL", type=float, default=0, help="Tot bias regularization")
parser.add_argument("--adj_prior_mu", type=float, default=0, help="Tot bias regularization")
parser.add_argument("--adj_prior_sigma", type=float, default=0.2, help="Tot bias regularization")

# loss
parser.add_argument("--loss_type", type=str, default='EMD', help="loss function")
parser.add_argument("--per_condition_loss", type=str_to_bool, default='true', help="loss function")

parser.add_argument("--cat_bias_orthogonality_scaler", type=float, default=0, help="Tot bias regularization")
parser.add_argument("--cp_method", type=str)
parser.add_argument("--cp_include_adjacency", type=str_to_bool)
parser.add_argument("--cp_per_label", type=str_to_bool)

# adversarial
parser.add_argument("--cat_max_penalty_weight", type=float, default=1.5, help="cat disc penalty weight b param")
parser.add_argument("--cat_b_adv", type=float, default=1.5, help="cat disc penalty weight b param")
parser.add_argument("--pert_max_penalty_weight", type=float, default=1.5, help="cat disc penalty weight b param")
parser.add_argument("--pert_b_adv", type=float, default=1.5, help="cat disc penalty weight b param")

# noise
parser.add_argument("--network_noise_scale", type=float, default=1.5, help="cat disc penalty weight b param")
parser.add_argument("--min_network_noise", type=float, default=1.5, help="cat disc penalty weight b param")
parser.add_argument("--include_gradient_noise_vae", type=str_to_bool, default=1.5, help="cat disc penalty weight b param")
parser.add_argument("--include_gradient_noise_embedding", type=str_to_bool, default=1.5, help="cat disc penalty weight b param")
parser.add_argument("--constant_gradient_noise", type=str_to_bool, default=1.5, help="cat disc penalty weight b param")
parser.add_argument("--gradient_noise_scale", type=float, default=1.5, help="cat disc penalty weight b param")
parser.add_argument("--lr_period", type=float, default=4, help="cat disc penalty weight b param")

parser.add_argument("--reset_state", type=str_to_bool, default='true', help="cat disc penalty weight b param")

# other adversarial
parser.add_argument("--train_batch", type=int)
parser.add_argument("--initialize_fc", type=str_to_bool)
parser.add_argument("--generator_dropout_rate", type=float)
parser.add_argument("--cat_discriminator_dropout_rate", type=float)
parser.add_argument("--pert_discriminator_dropout_rate", type=float)



parser.add_argument("--discriminator_batch_momentum", type=float)
parser.add_argument("--spectral_norm", type=str_to_bool)
parser.add_argument("--discriminator_lambda_L2", type=float)
parser.add_argument("--discriminator_bionet_activation", type=str_to_bool)
parser.add_argument("--smooth_labels", type=str_to_bool)
parser.add_argument("--gradient_ascent", type=str_to_bool)

parser.add_argument("--n_adversarial_start", type=int)
parser.add_argument("--n_discriminator_train", type=int)

parser.add_argument("--vae_lambda_l2", type=float)

parser.add_argument("--min_cat_adv_penalty", type=float)
parser.add_argument("--min_pert_adv_penalty", type=float)

parser.add_argument("--main_max_lr", type=float)
parser.add_argument("--generator_max_lr", type=float)
parser.add_argument("--cat_max_lr", type=float)
parser.add_argument("--pert_max_lr", type=float)
parser.add_argument("--lr_decay", type=float)

########################################################################
args = parser.parse_args()
fn = str(args.index)
run_type = str(args.run_type)
fn += run_type
# seed = args.seed
# loo = args.loo
bn_weights_lambda_L2 = args.bn_weights_lambda_L2
uniform_lambda_L2 = args.uniform_lambda_L2
cat_max_norm = args.cat_max_norm
global_bias_lambda_L2 = args.global_bias_lambda_L2   
cat_bias_lambda_L2 = args.cat_bias_lambda_L2
vae_scaling_KL = args.vae_scaling_KL
global_bias_lambda_L1 = args.global_bias_lambda_L1   
cat_bias_lambda_L1 = args.cat_bias_lambda_L1
vae_prior_mu = args.vae_prior_mu
vae_prior_sigma = args.vae_prior_sigma
adj_scaling_KL = args.adj_scaling_KL
adj_prior_mu = args.adj_prior_mu
adj_prior_sigma = args.adj_prior_sigma
loss_type = args.loss_type
per_condition_loss = args.per_condition_loss

cat_b_adv = args.cat_b_adv
cat_max_penalty_weight = args.cat_max_penalty_weight
pert_b_adv = args.pert_b_adv
pert_max_penalty_weight = args.pert_max_penalty_weight

network_noise_scale = args.network_noise_scale
min_network_noise = args.min_network_noise
include_gradient_noise_vae = args.include_gradient_noise_vae
include_gradient_noise_embedding = args.include_gradient_noise_embedding
constant_gradient_noise = args.constant_gradient_noise
gradient_noise_scale = args.gradient_noise_scale
lr_period = args.lr_period

reset_state = args.reset_state

train_batch = args.train_batch
initialize_fc = args.initialize_fc
generator_dropout_rate = args.generator_dropout_rate
cat_discriminator_dropout_rate = args.cat_discriminator_dropout_rate
pert_discriminator_dropout_rate = args.pert_discriminator_dropout_rate
discriminator_batch_momentum = None if args.discriminator_batch_momentum == 0 else args.discriminator_batch_momentum
spectral_norm = args.spectral_norm
discriminator_lambda_L2 = 0 if spectral_norm else args.discriminator_lambda_L2
discriminator_bionet_activation = args.discriminator_bionet_activation
smooth_labels = args.smooth_labels
gradient_ascent = args.gradient_ascent
n_adversarial_start = args.n_adversarial_start
n_discriminator_train = args.n_discriminator_train

vae_lambda_l2 = args.vae_lambda_l2
min_cat_adv_penalty = args.min_cat_adv_penalty
min_pert_adv_penalty = args.min_pert_adv_penalty

main_max_lr = args.main_max_lr
generator_max_lr = args.generator_max_lr
cat_max_lr = args.cat_max_lr
pert_max_lr = args.pert_max_lr
lr_decay = args.lr_decay

cat_bias_orthogonality_scaler = args.cat_bias_orthogonality_scaler
cp_method = args.cp_method
cp_include_adjacency = args.cp_include_adjacency
cp_per_label = args.cp_per_label

#python test_run.py --index 52 --run_type E --bn_weights_lambda_L2 1e-7 --uniform_lambda_L2 1e-7 --cat_max_norm 100 --global_bias_lambda_L2 0 --cat_bias_lambda_L2 1e-4 --vae_scaling_KL 1e-3 --global_bias_lambda_L1 0 --cat_bias_lambda_L1 0 --vae_prior_mu 0 --vae_prior_sigma 1 --adj_scaling_KL 0 --adj_prior_mu 0 --adj_prior_sigma 0.2 --loss_type MSE --per_condition_loss true --cat_max_penalty_weight 12 --cat_b_adv 2 --pert_max_penalty_weight 8 --pert_b_adv 3.5 --network_noise_scale 0.01 --min_network_noise 0.0025 --include_gradient_noise_vae true --include_gradient_noise_embedding true --constant_gradient_noise true --gradient_noise_scale 1e-9 --lr_period 4 --reset_state false --train_batch 500 --initialize_fc true --generator_dropout_rate 0.7 --cat_discriminator_dropout_rate 0.1 --pert_discriminator_dropout_rate 0.1 --discriminator_batch_momentum 0 --spectral_norm false --discriminator_lambda_L2 1e-3 --discriminator_bionet_activation false --smooth_labels true --gradient_ascent true --n_adversarial_start 200 --n_discriminator_train 5 --vae_lambda_l2 1e-5 --min_cat_adv_penalty 0.1 --min_pert_adv_penalty 0.1 --main_max_lr 2e-3 --generator_max_lr 5e-4 --cat_max_lr 1e-3 --pert_max_lr 1e-3 --lr_decay 0.9 --cat_bias_orthogonality_scaler 100 --cp_method orthogonality --cp_include_adjacency false --cp_per_label false


# 

# In[15]:


# index = "20v3"
# run_type = "E"
# bn_weights_lambda_L2 = 1e-7
# uniform_lambda_L2 = 1e-7
# cat_max_norm = 100
# global_bias_lambda_L2 = 0
# cat_bias_lambda_L2 = 1e-4
# vae_scaling_KL = 1e-3
# global_bias_lambda_L1 = 0
# cat_bias_lambda_L1 = 0
# vae_prior_mu = 0
# vae_prior_sigma = 1
# adj_scaling_KL = 0
# adj_prior_mu = 0
# adj_prior_sigma = 0.2
# loss_type = "MSE"
# per_condition_loss = True
# cat_max_penalty_weight = 12
# cat_b_adv = 2
# pert_max_penalty_weight = 8
# pert_b_adv = 3.5
# network_noise_scale = 0.01
# min_network_noise = 0.0025
# include_gradient_noise_vae = True
# include_gradient_noise_embedding = True
# constant_gradient_noise = True
# gradient_noise_scale = 1e-9
# lr_period = 4
# reset_state = False
# train_batch = 500
# initialize_fc = True
# generator_dropout_rate = 0.7
# cat_discriminator_dropout_rate = 0.1
# pert_discriminator_dropout_rate = 0.1
# discriminator_batch_momentum = 0
# spectral_norm = False
# discriminator_lambda_L2 = 1e-3
# discriminator_bionet_activation = False
# smooth_labels = True
# gradient_ascent = True
# n_adversarial_start = 200
# n_discriminator_train = 5
# vae_lambda_l2 = 1e-5
# min_cat_adv_penalty = 0.1
# min_pert_adv_penalty = 0.1
# main_max_lr = 2e-3
# generator_max_lr = 5e-4
# cat_max_lr = 1e-3
# pert_max_lr = 1e-3
# lr_decay = 0.9
# cat_bias_orthogonality_scaler = 100
# cp_method = "orthogonality"
# cp_include_adjacency = False
# cp_per_label = False


# In[16]:


run_types = {'A': (1, True),
            'B': (3, True), 
            'C': (7, True), 
            'D': (10, True), 
            'E': (1, False)
            }
seed, loo = run_types[run_type]


# In[7]:


visualize = False
mod_type = 'default'
subset = False
short_run = False
drop_low_counts = True
seed_split = 688 #888

n_eval_cells = 20 if per_condition_loss else 100

import torch
from geomloss import SamplesLoss
device = "cuda" if torch.cuda.is_available() else "cpu"

if loss_type == 'MSE':
    loss_scaler = 100
    prediction_loss_fn = torch.nn.MSELoss(reduction='mean')
else:
    loss_scaler = 1
    prediction_loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05).to(device)


# In[4]:


n_fraction = 0.2

train_frac, test_frac = 0.8, 0.2

# a_scale = 10
# b_scale = 2

# if mod_type not in ['default', 'tot_bias_scaler', 'global_bias_scaler', 'mu_bias_scaler', 
#                    'global_bias_regularizer', 'mu_bias_regularizer', 
#                    'standardized_weights']:
#     raise ValueError('Incorrect mod type specified')

# # assign file name accordingly

# fn = mod_type
# if mod_type == 'tot_bias_scaler':
#     fn += '_tot_a' + str(a_scale) + '_b' + str(b_scale)
# elif mod_type == 'global_bias_scaler':
#     fn += '_global_a' + str(a_scale) + '_b' + str(b_scale)
# elif mod_type == 'mu_bias_scaler':
#     fn += '_mu_a' + str(a_scale) + '_b' + str(b_scale)
# if subset:
#     fn += '_subset'
# if short_run:
#     fn += '_short_run'
# if loo:
#     fn += '_loo'


# In[2]:


import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="anndata.utils")
from scipy.sparse import SparseEfficiencyWarning
with warnings.catch_warnings():
    warnings.simplefilter("ignore", SparseEfficiencyWarning)

import os
from typing import Optional, Dict, List, Literal
import copy
import itertools
from tqdm import tqdm
from tqdm import trange
import gc
import math

import pandas as pd
import numpy as np
import scanpy as sc

from scipy import stats
from sklearn.metrics import normalized_mutual_info_score
import torch
import scanpy as sc
import pandas as pd
from typing import List, Literal
import torch
from geomloss import SamplesLoss
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from sklearn.model_selection import train_test_split


# In[3]:


import sys
sclembas_path = '/home/hmbaghda/Projects/scLEMBAS'
sys.path.insert(1, os.path.join(sclembas_path))
from scLEMBAS import io

# from scLEMBAS.model.train_dev_mu_regularizer import TrainSC as TrainSCDevMu
# from scLEMBAS.model.train_dev_weights_standard import TrainSC as TrainSCDevWstandard
from scLEMBAS.model.train import TrainSC
from scLEMBAS.preprocess import discriminator_weight_curve, get_alignment_score


# from scLEMBAS.model.scl_dev_tot_scaler import SignalingModel as SignalingModelDevTot
# from scLEMBAS.model.scl_dev_global_scaler import SignalingModel as SignalingModelDevGlobal
# from scLEMBAS.model.scl_dev_mu_scaler import SignalingModel as SignalingModelDevMu
# from scLEMBAS.model.scl_dev_weights_standard import SignalingModel as SignalingModelDevWstandard
from scLEMBAS.model.scl import SignalingModel

SM = {'default': SignalingModel}#, 
#      'tot_bias_scaler': SignalingModelDevTot, 
#      'global_bias_scaler': SignalingModelDevGlobal, 
#      'mu_bias_scaler': SignalingModelDevMu, 
#      'global_bias_regularizer': SignalingModel, 
#      'mu_bias_regularizer': SignalingModel,
#      'standardized_weights': SignalingModelDevWstandard # initializes weights much larger
#      }

TR = {'default': TrainSC}#, 
#      'tot_bias_scaler': TrainSC, 
#      'global_bias_scaler': TrainSC, 
#      'mu_bias_scaler': TrainSC, 
#      'global_bias_regularizer': TrainSC, # by default, regularizes global bias
#       'mu_bias_regularizer': TrainSCDevMu, 
#       'standardized_weights': TrainSCDevWstandard 
#      }

sys.path.insert(1, os.path.join(sclembas_path, 'notebooks/Kang_2017/'))
from Kang_utils import (rev_stim, stim_map, rev_stim_map, adata_dimviz_bias, clear_memory,
                        get_prediction, adata_dimviz_prediction, prepare_for_metrics, get_loss)


# In[4]:


n_cores = 30
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)

data_path = '/nobackup/users/hmbaghda/scLEMBAS/analysis'
device = "cuda" if torch.cuda.is_available() else "cpu"


# In[10]:


adata = sc.read_h5ad(os.path.join(data_path, 'processed', 'kang_expr_scored.h5ad'))
sn_ppis = pd.read_csv(os.path.join(data_path, 'processed', 'Kang_sn_ppis.csv'), index_col = 0)

tf_adata = io.read_tfad(os.path.join(data_path, 'processed', 'Kang_tf_activity.h5ad'))
tf_adata.obs['condition'] = tf_adata.obs['stim'].astype(str) + '^' + tf_adata.obs['seurat_annotations'].astype(str)

# ensures correct order of test data
# note, this already saved in order, matching the mod.y_out columns
tf_adata = tf_adata[:, sorted(tf_adata.var_names)] 

source_label = 'source_genesymbol'
target_label = 'target_genesymbol'
weight_label = 'mode_of_action'
stimulation_label = 'consensus_stimulation'
inhibition_label = 'consensus_inhibition'

pert_col = 'stim'
cat_col = 'seurat_annotations'


# In[8]:


if drop_low_counts:
    drop_ct = tf_adata.obs.seurat_annotations.value_counts().index.tolist()[-3:]
    tf_adata = tf_adata[~tf_adata.obs.seurat_annotations.isin(drop_ct)]
    adata = adata[tf_adata.obs_names,:]


# # 1. Create a novel train-test split:

# In[12]:


def ood_split(tf_adata, 
             train_frac: float, 
              stim_col: str = 'stim', 
              context_col: str = 'seurat_annotations', 
              context_bins: Optional[pd.DataFrame] = None, 
              context_bins_frac: float = 1, 
             max_iter: int = 1000, 
             seed: int = 888, 
             deviation_thresh: float = 0.025, 
             include_train_cond: Optional[Dict[str, List]] = None):
    """Generate an OOD train test split, where both the condition data and the single-cells have approximately
    the specified split.
    
    Rules: 
    1. cell split and condition split is similar (conditions is exactly the split, 
    cells is approximate by deviation_thresh)
    2. Each component (condition_cols) of the condition is seen atleast once in the training
    3. Each of the stimulation column is seen atleast once in the test
    4. The test context (cell type) needs to contain atleast one from context_frac of the total context_bins 

    Parameters
    ----------
    tf_adata : 
        AnnData object of TF activity
    train_frac : float
        the fraction of the data going to training
    stim_col : List[str]
        the metadata column defining the stimulation condition
    context_col: str
        the metadata column of the context, together with stim col defines the OOD split
    context_bins: Optional[pd.DataFrame]
        binning of the context categories, by default None
    context_bins_frac: float
        the number of bins that should be included in the test data from the context bins
    max_iter : int, optional
        total iterations to try to identify a suitable split, by default 1000
    seed : int, optional
        random state variable, by default 888
    deviation_thresh : float, optional
        extent to which to allow the single-cell split to deviate from the specified `train_frac split`, by default 0.025
    include_train_cond : Optional[Dict[str, List]]
        a dictionary with keys for each of stim_col and context_col and keys as a list that represents a subset
        of values in that column that must be included in the training set. 
    """
    test_frac = 1 - train_frac

    condition_cols = [stim_col, context_col]
    condition_combs = tf_adata.obs[condition_cols].apply(lambda row: '^'.join(row.astype(str)), axis=1)
    unique_conditions = condition_combs.drop_duplicates(keep = 'first', inplace = False).tolist()

    # define the conditions for stopping
    train_frac_deviation = np.inf
    zero_shot_bool = False

    test_stim = False
    n_stims = tf_adata.obs['stim'].nunique()

    if context_bins is not None:
        if context_bins_frac > 1 or context_bins_frac < 0:
            raise ValueError('The fraction must be between 0 and 1')
        n_contexts = np.floor(context_bins_frac*context_bins.nunique())
        context_present = False
    else:
        context_present = True

    if include_train_cond is not None:
        if sorted(set(include_train_cond).intersection(condition_cols)) != sorted(condition_cols):
            raise ValueError('Keys for `include_train_cond` must be the same as `condition_cols`')
        for cond, cond_vals in include_train_cond.items():
            if not set(cond_vals).issubset(tf_adata.obs[cond].tolist()):
                raise ValueError('The conditions to include in training for ' + cond + ' are not present in the metadata.')
        ict_bool = False
    else:
        ict_bool = True

    counter = 0
    while (train_frac_deviation > deviation_thresh or not zero_shot_bool or not ict_bool or not test_stim or not context_present) and (counter < max_iter):
        # ood at categorical level
        train_cond, test_cond = train_test_split(unique_conditions, test_size = test_frac, random_state = seed + counter, shuffle = True)
        train_cells = condition_combs[condition_combs.isin(train_cond)].index.tolist()
        test_cells = condition_combs[condition_combs.isin(test_cond)].index.tolist()

        train_frac_actual = len(train_cells)/tf_adata.shape[0]
        train_frac_deviation = abs(train_frac_actual - train_frac)

        # ensure that all individual conditions are seen atleast once (it's the combination of conditions that's unique)
        train_cond_map = {}
        for idx, cond in enumerate(condition_cols):
            train_cond_map[cond] = {tc.split('^')[idx] for tc in train_cond}

        test_cond_map = {}
        for idx, cond in enumerate(condition_cols):
            test_cond_map[cond] = {tc.split('^')[idx] for tc in test_cond}

        zero_shots = [len(test_cond_map[cond].difference(train_cond_map[cond])) for cond in condition_cols]    
        zero_shot_bool = all(value == 0 for value in zero_shots)

        test_stim = (len(test_cond_map['stim']) == n_stims)


        if context_bins is not None:
            context_present = context_bins.loc[list(test_cond_map[context_col])].nunique() >= n_contexts

        if include_train_cond is not None:
            ict_conds = [len(include_train_cond[cond].difference(cond_vals)) for cond, cond_vals in train_cond_map.items()]
            ict_bool = any(value == 0 for value in ict_conds)

        counter += 1

    if counter < max_iter:
        return train_cells, test_cells, train_cond, test_cond
    else:
        return None, None, None, None


# In[13]:


contingency_table = pd.crosstab(tf_adata.obs['stim'], tf_adata.obs['seurat_annotations'], 
                                margins=True, margins_name="Total")
contingency_table = contingency_table.T.sort_values(by = 'Total').T
bins = pd.qcut(contingency_table.T.Total, q = 4, labels = False)


# In[17]:


if not loo:
    condition_cols = ['stim', 'seurat_annotations']
    train_cells, test_cells, train_cond, test_cond = ood_split(tf_adata,
                                                                       train_frac = 0.8,
                                                                       stim_col = condition_cols[0], 
                                                                       context_col = condition_cols[1], 
                                                                       context_bins = bins, 
                                                                       context_bins_frac = 1, 
                                                                       max_iter = 1000, 
                                                                       seed = seed_split, 
                                                                       deviation_thresh = 0.025, 
                                                                       include_train_cond = None)

    test_conds = test_cond
    train_conds = train_cond
else:
    unique_conditions = sorted(set(tf_adata.obs.condition))

    if run_type in ['A', 'B']:
        unique_conditions.remove('CTRL^DC')
        unique_conditions.insert(0, 'CTRL^DC')
    elif run_type in ['C', 'D']:
        unique_conditions.remove('STIM^DC')
        unique_conditions.insert(0, 'STIM^DC')  
    
    test_cond = [unique_conditions[0]]
    train_cond = sorted(set(unique_conditions).difference(test_cond))
    
    train_conds = train_cond
    test_conds = test_cond
    
    test_cells = tf_adata.obs[tf_adata.obs.condition.isin(test_cond)].index.tolist()
    train_cells = tf_adata.obs[tf_adata.obs.condition.isin(train_cond)].index.tolist()


# In[18]:


condition_proportions = tf_adata.obs['condition'].value_counts()
condition_proportions.loc[train_cond].sort_values(ascending = True)


# In[19]:


condition_proportions.loc[test_cond].sort_values(ascending = True)


# # 2. Subset data

# In[20]:


if subset:
    adata_all = adata.copy()
    tf_adata_all = tf_adata.copy()
    train_cells_all = copy.deepcopy(train_cells)

    tf_adata_train = tf_adata_all[train_cells_all, :]
    condition_proportions = tf_adata_train.obs['condition'].value_counts(normalize=True)
    total_cells = tf_adata_train.n_obs
    n_subset = int(n_fraction * total_cells)

    # number of cells to sample per condition
    sample_sizes = (condition_proportions * n_subset).round().astype(int)

    # Subset the cells for each condition
    subset_indices = []
    for condition, n_cells in sample_sizes.items():
        condition_indices = tf_adata_train.obs[tf_adata_train.obs['condition'] == condition].index
        np.random.seed(seed_split)
        sampled_condition_indices = np.random.choice(condition_indices, size=n_cells, replace=False)
        subset_indices.extend(sampled_condition_indices)

    n_train = len(subset_indices)
    train_cells = copy.deepcopy(subset_indices)
    print('The number of train cells has been reduced from {} to {}'.format(tf_adata_train.shape[0], 
                                                                           n_train))

    # include the test data
    subset_indices.extend(test_cells)
    tf_adata = tf_adata_all[subset_indices,:]
    adata = adata_all[subset_indices,:]
    


# In[21]:


tf_adata


# In[22]:


condition_proportions = tf_adata.obs['condition'].value_counts()
condition_proportions.loc[train_cond].sort_values(ascending = True)


# In[23]:


condition_proportions.loc[test_cond].sort_values(ascending = True)


# In[26]:


tf_adata[train_cells, :].obs[cat_col].value_counts()


# In[27]:


tf_adata[test_cells, :].obs[cat_col].value_counts()


# In[28]:


tf_adata[train_cells, :].obs[pert_col].value_counts()


# In[29]:


tf_adata[test_cells, :].obs[pert_col].value_counts()


# # 3. Run training

# In[31]:


if not short_run:
    max_epochs = 600
else:
    max_epochs = 250 


# In[32]:


def generate_lr_params(n_epochs, max_lr, lr_scaling_factor=10, lr_decay=0.75, role='scl'):
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
            
#     n_restarts = 3 if total_active_epochs // n_discriminator_train_ > 500 else 2
    n_restarts = 4 if total_active_epochs // n_discriminator_train_ > 500 else 2


    T_0 = max(1, (total_active_epochs // n_discriminator_train_) // n_restarts)
    warmup_epochs = max(1, (total_active_epochs // n_discriminator_train_) // 10)

    if reset_state:
        if total_active_epochs // n_discriminator_train_ > 400:
            n_optimizer_resets = 2
        elif total_active_epochs // n_discriminator_train_ < 100:
            n_optimizer_resets = 0
        else:
            n_optimizer_resets = 1
    else:
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
                                  lr_scaling_factor = 10, lr_decay = lr_decay):
    general_params = generate_lr_params(n_epochs, #n_epochs - n_adversarial_start, 
                                        max_lr, 
                                        lr_scaling_factor = lr_scaling_factor, 
                                        lr_decay = lr_decay,
                                       role = 'discriminator')
    
    keys_to_keep = ['maximum_learning_rate', 'minimum_learning_rate', 'lr_restart_epoch', 
                   'warmup_epochs', 'lr_decay', 'n_optimizer_resets']
    discriminator_params = {'batch_momentum': discriminator_batch_momentum,
                            'layer_norm': False,
                            'spectral_norm': spectral_norm,
                            'dropout_rate': discriminator_dropout_rate,
                            'activation_fn': nn.LeakyReLU,
                            'n_hidden_nodes': [768, 512, 256],
                            'lr_restart_factor': 1,
                            'optimizer': torch.optim.Adam,
                            'discriminator_lambda_L2': discriminator_lambda_L2,
                            'discriminator_penalty_weight': discriminator_penalty_weight, 
                            'bionet_activation': discriminator_bionet_activation,
                           'initialize': initialize_fc, 
                           'smooth_labels': smooth_labels, 
                           'epsilon_smoothing': 0.1}
    discriminator_params = {**discriminator_params, 
                           **{k:v for k,v in general_params.items() if k in keys_to_keep}}
    
    return discriminator_params


# In[34]:


# hyperparameters
n_layers_vae = 2
n_nodes = len(set(sn_ppis[source_label].tolist() + sn_ppis[target_label].tolist()))
vae_n_hidden_nodes = list(np.round(np.linspace(adata.shape[1], n_nodes, n_layers_vae + 2)).astype(int)[1:-1])

# linear scaling of inputs/outputs
projection_amplitude_in = 10
projection_amplitude_out = 1

# other parameters
bionet_params = {'target_steps': 100, 
                 'max_steps': 120, 
                 'exp_factor':50, 
                 'tolerance': 1e-5, 
                 'leak':1e-2}

vae_mod_params = {'vae_batch_momentum': 0.01, 
              'vae_layer_norm': False, 
              'vae_dropout_rate': generator_dropout_rate,
              'vae_activation_fn': nn.LeakyReLU,
              'vae_n_hidden_nodes': vae_n_hidden_nodes, 
              'vae_var_min': 1e-4, 
             'vae_initialize': initialize_fc}
bionet_params = {**bionet_params, **vae_mod_params}


bionet_params['cat_max_norm'] = cat_max_norm


# In[35]:


# training parameters
batch_params_default = {'test_batch_size':round(len(test_cells)/2)}
spectral_radius_params = {'n_probes_spectral': 5, 
                          'power_steps_spectral': 5, #50, 
                          'subset_n_spectral': 5} #10}

target_spectral_radius = 0.9

regularization_params_default = {'input_lambda_L2': 0, # doesn't matter if setting the requires grad to False
                         'bn_weights_lambda_L2': bn_weights_lambda_L2, #1e-7, 
                         'global_bias_lambda_L2': global_bias_lambda_L2, #0, # don't incorporate because of KL divergence
                                 'cat_bias_lambda_L2': cat_bias_lambda_L2,
                         'output_weights_lambda_L2': 1e-7,
                         'output_bias_lambda_L2': 1e-7,
                         'moa_lambda_L1': 1e2,  
                         'uniform_lambda_L2': uniform_lambda_L2,#, 1e-7,
                         'uniform_min': 0,
                         'uniform_max': 1, 
                         'spectral_loss_factor': 1e-6,
                               'global_bias_lambda_L1': global_bias_lambda_L1, 
                                'cat_bias_lambda_L1': cat_bias_lambda_L1,
                                'adj_scaling_KL': adj_scaling_KL, 
                                 'adj_prior_mu': adj_prior_mu, 
                                 'adj_prior_sigma': adj_prior_sigma, 
                                 'cat_bias_orthogonality_scaler': cat_bias_orthogonality_scaler
                                }




if cp_method == 'orthogonality':
    cps = cat_bias_orthogonality_scaler
elif cp_method == 'kl_divergence':
    if cp_include_adjacency:
        cps = cat_bias_orthogonality_scaler*2
    else:
        cps = cat_bias_orthogonality_scaler/10
elif cp_method == 'info_nce':
    cps = cat_bias_orthogonality_scaler/150
cat_pert_params = {'regularization_scaler': cps, 
                       'method': cp_method, 
                       'per_label': cp_per_label, 
                       'include_adjacency': cp_include_adjacency, 
                       'temperature': 0.1
                      }

# if mod_type.endswith('_regularizer'):
#     regularization_params_default['bn_weights_lambda_l2'] = 1e-12 # decrease adj matrix regularization
#     regularization_params_default['global_bias_lambda_L2'] = 1e-5 # increase global bias regularization


# In[37]:


# max_lr = 0.001

if max_epochs > n_adversarial_start:
    cat_discriminator_penalty_weight = discriminator_weight_curve(n_epochs = max_epochs - n_adversarial_start,
                                                              min_penalty_weight = min_cat_adv_penalty,
                                                              max_penalty_weight = cat_max_penalty_weight,
                                                              a = 1,
                                                              b = cat_b_adv, 
                                                              curve_type = 'power')

    pert_discriminator_penalty_weight = discriminator_weight_curve(n_epochs = max_epochs - n_adversarial_start,
                                                                   min_penalty_weight = min_pert_adv_penalty,
                                                                   max_penalty_weight = pert_max_penalty_weight,
                                                                   a = 1,
                                                                   b = pert_b_adv, 
                                                                   curve_type = 'power')
else:
    cat_discriminator_penalty_weight, pert_discriminator_penalty_weight = 0,0



lr_params = generate_lr_params(n_epochs = max_epochs, 
                               max_lr = main_max_lr, lr_scaling_factor = 10, lr_decay = lr_decay, role = 'scl')

cat_discriminator_params = generate_discriminator_params(n_epochs = max_epochs, 
                                                         max_lr = cat_max_lr, 
                                                         discriminator_dropout_rate = cat_discriminator_dropout_rate,
                                                         discriminator_penalty_weight = cat_discriminator_penalty_weight, 
                                                         lr_scaling_factor = 10, lr_decay = lr_decay)
pert_discriminator_params = generate_discriminator_params(n_epochs = max_epochs, 
                                                         max_lr = pert_max_lr, 
                                                          discriminator_dropout_rate = pert_discriminator_dropout_rate,
                                                         discriminator_penalty_weight = pert_discriminator_penalty_weight, 
                                                         lr_scaling_factor = 10, lr_decay = lr_decay)


vae_params = {**{'prior_mu': vae_prior_mu,
              'prior_sigma': vae_prior_sigma,
              'lambda_l2': vae_lambda_l2,
              'scaling_KL': vae_scaling_KL, #1e-2
              'optimizer': torch.optim.Adam}, 
                 **generate_lr_params(n_epochs = max_epochs, 
                                     max_lr = generator_max_lr, 
                                     lr_scaling_factor = 10, lr_decay = lr_decay, 
                                     role = 'discriminator') # generator
             }
del vae_params['max_epochs']





regularization_params = regularization_params_default.copy()
batch_params = {**batch_params_default,
                **{'train_batch_size': train_batch,
                   'validation_batch_size': np.nan}}
training_params = {**lr_params, **batch_params, **regularization_params, **spectral_radius_params}
training_params['prediction_loss_fn_scaler'] = loss_scaler

# noise params
training_params['network_noise_scale'] = network_noise_scale
training_params['min_network_noise'] = min_network_noise

# gradient noise
training_params['include_gradient_noise_vae'] = include_gradient_noise_vae
training_params['include_gradient_noise_embedding'] = include_gradient_noise_embedding
training_params['constant_gradient_noise'] = constant_gradient_noise
training_params['gradient_noise_scale'] = gradient_noise_scale


# training_params['network_noise_scale'] = projection_amplitude_in/1000 # current default, Avlant's was 3*projection_amplitude_in/1000
# training_params['min_network_noise'] = training_params['network_noise_scale'] /4 # to try - 0 is default

# # gradient noise
# training_params['include_gradient_noise_vae'] = True # original default
# training_params['include_gradient_noise_embedding'] = True # original default
# training_params['constant_gradient_noise'] = True # original default
# training_params['gradient_noise_scale'] = 1e-9 # original default 


# In[4]:


# fm = SM[mod_type](net = sn_ppis,
#                  X_in = pd.DataFrame(tf_adata.obs.stim.cat.codes, columns = ['IFNB1']),
#                  y_out = tf_adata.to_df().copy(), 
#                  expr = adata.to_df().copy(), 
#                  covariates = tf_adata.obs.copy(),
#                  categorical_covariate_keys = ['seurat_annotations'],
#                  projection_amplitude_in = projection_amplitude_in, 
#                  projection_amplitude_out = projection_amplitude_out,
#                  weight_label = weight_label, source_label = source_label, target_label = target_label,
#                  bionet_params = bionet_params, 
#                  dtype = torch.float32, device = device, seed = seed)

# fm.input_layer.weights.requires_grad = False # don't learn scaling factors for the ligand input concentrations
# fm.signaling_network.prescale_weights(target_radius = target_spectral_radius) # spectral radius




# tr = TR[mod_type](mod = fm,
#                    prediction_optimizer = torch.optim.Adam,
#                    prediction_loss_fn = prediction_loss_fn, 
#                        per_condition_loss = per_condition_loss,
#                        n_adversarial_start = n_adversarial_start, 
#                        n_discriminator_train = n_discriminator_train,
#                        gradient_ascent = gradient_ascent,
#                   cat_discriminator_params = cat_discriminator_params,
#                        pert_discriminator_params = pert_discriminator_params,
#                        vae_params = vae_params,
#                    hyper_params = training_params,
#                    train_split = {'train': train_cells, 'test': test_cells, 'validation': None}, 
#                    train_seed = seed, 
#                    track_test = True,
#                    track_validation = False, 
#                       n_eval_cells = n_eval_cells, 
#                       n_eval_bootstrap = 3)

# lrs = {'mod': [], 
#       'cat_disc': [], 
#       'pert_disc': [], 
#       'vae': []}
# for e in range(tr.hyper_params['max_epochs']):
#     no_vae = (tr.n_adversarial_start > e) or (e % tr.n_discriminator_train != 0)
    
#     lrs['mod'].append(tr.prediction_optimizer.param_groups[0]['lr'])
#     lrs['vae'].append(tr.vae_learning['optimizer'].param_groups[0]['lr'])
#     lrs['cat_disc'].append(tr.cat_discriminator['optimizer'].param_groups[0]['lr'])
#     lrs['pert_disc'].append(tr.cat_discriminator['optimizer'].param_groups[0]['lr'])
#     if tr.n_adversarial_start <= e:
#         tr.cat_discriminator['lr_scheduler'].step()
#         tr.pert_discriminator['lr_scheduler'].step()
#     if not no_vae:
#         tr.vae_learning['lr_scheduler'].step()
#     tr.lr_scheduler.step()
    
# lrs = pd.DataFrame(lrs)
# lrs['epoch'] = range(lrs.shape[0])

# cols = ['mod', 'cat_disc', 'pert_disc', 'vae']
# ncols = len(cols)
# fig, ax = plt.subplots(ncols = ncols, figsize = (5.1*ncols, 5))

# for (i, col) in enumerate(cols):
#     sns.lineplot(data = lrs, x = 'epoch', y = col, ax = ax[i])
# fig.tight_layout()
# ;


# In[ ]:


# if 'cat_discriminator' not in tr.__dict__.keys():
#     tr.cat_discriminator = tr.discriminator

# fig, ax = plt.subplots(ncols = 2, figsize = (13,5))
# sns.lineplot(tr.cat_discriminator['params']['discriminator_penalty_weight'], ax = ax[0])
# ax[0].set_title('Categorical Discriminator')

# if 'pert_discriminator' in tr.__dict__.keys():
#     sns.lineplot(tr.pert_discriminator['params']['discriminator_penalty_weight'], ax = ax[1])
# ax[1].set_title('Perturbation Discriminator')

# for i in range(2):
#     ax[i].set_xlabel('Epochs')
#     ax[i].set_ylabel('Discriminator Penalty Weight')
    
# fig.tight_layout();


# In[57]:


mod = SM[mod_type](net = sn_ppis,
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

mod.input_layer.weights.requires_grad = False # don't learn scaling factors for the ligand input concentrations
mod.signaling_network.prescale_weights(target_radius = target_spectral_radius) # spectral radius


# In[60]:


trainer = TR[mod_type](mod = mod,
                   prediction_optimizer = torch.optim.Adam,
                   prediction_loss_fn = prediction_loss_fn, 
                       per_condition_loss = per_condition_loss,
                       n_adversarial_start = n_adversarial_start, 
                       n_cat_discriminator_train = n_discriminator_train,
                    n_pert_discriminator_train = n_discriminator_train,
                       gradient_ascent = gradient_ascent,
                  cat_discriminator_params = cat_discriminator_params,
                       pert_discriminator_params = pert_discriminator_params,
                       vae_params = vae_params,
                   hyper_params = training_params,
                       cat_pert_params = cat_pert_params, 
                   train_split = {'train': train_cells, 'test': test_cells, 'validation': None}, 
                   train_seed = seed, 
                   track_test = True,
                   track_validation = False, 
                      n_eval_cells = n_eval_cells, 
                      n_eval_bootstrap = 3)


# In[61]:


import cProfile
import pstats
from io import StringIO
profiler = cProfile.Profile()
profiler.enable()

mod = trainer.train_model(verbose = False)

profiler.disable()

s = StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
ps.print_stats()

lines = s.getvalue().split('\n')

# Convert to DataFrame
parsed = []
for line in lines[5:]:  
    parts = line.split(None, 5)  
    if len(parts) == 6:
        ncalls, tottime, percall1, cumtime, percall2, func = parts
        parsed.append({
            'ncalls': ncalls,
            'tottime': float(tottime),
            'percall_tottime': float(percall1),
            'cumtime': float(cumtime),
            'percall_cumtime': float(percall2),
            'func': func
        })

timer_df = pd.DataFrame(parsed)
timer_df.to_csv(os.path.join(data_path, 'trash', fn + '_timing.csv'))

io.write_pickled_object(trainer,  os.path.join(data_path, 'trash', fn + '_feb_trainer.pickle'))


# In[30]:


clear_memory()


# In[19]:


import papermill as pm
from nbconvert import HTMLExporter
import nbformat
import os

input_notebook = 'test_visualize.ipynb' # in the current directory
output_notebook = os.path.join(data_path, 'trash', fn + '.ipynb')
output_html = os.path.join(data_path, 'trash', fn + '.html')

pm.execute_notebook(
    input_path=input_notebook,
    output_path=output_notebook,
    parameters={"fn": fn}, 
    kernel_name='python3'
)

nb = nbformat.read(output_notebook, as_version=4)
html_exporter = HTMLExporter()
html_exporter.exclude_input = True  # <-- hides code cells
(body, _) = html_exporter.from_notebook_node(nb)

with open(output_html, "w", encoding="utf-8") as f:
    f.write(body)
    
os.remove(output_notebook)


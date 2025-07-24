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
parser.add_argument("--discriminator_dropout_rate", type=float)


parser.add_argument("--discriminator_batch_momentum", type=float)
parser.add_argument("--spectral_norm", type=str_to_bool)
parser.add_argument("--discriminator_lambda_L2", type=float)
parser.add_argument("--discriminator_bionet_activation", type=str_to_bool)
parser.add_argument("--smooth_labels", type=str_to_bool)
parser.add_argument("--gradient_ascent", type=str_to_bool)

parser.add_argument("--n_adversarial_start", type=int)
parser.add_argument("--n_discriminator_train", type=int)

parser.add_argument("--lr_decay", type=float)
parser.add_argument("--vae_lambda_l2", type=float)


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

cat_bias_orthogonality_scaler = args.cat_bias_orthogonality_scaler

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
discriminator_dropout_rate = args.discriminator_dropout_rate
discriminator_batch_momentum = None if args.discriminator_batch_momentum == 0 else args.discriminator_batch_momentum
spectral_norm = args.spectral_norm
discriminator_lambda_L2 = 0 if spectral_norm else args.discriminator_lambda_L2
discriminator_bionet_activation = args.discriminator_bionet_activation
smooth_labels = args.smooth_labels
gradient_ascent = args.gradient_ascent
n_adversarial_start = args.n_adversarial_start
n_discriminator_train = args.n_discriminator_train

lr_decay = args.lr_decay
vae_lambda_l2 = args.vae_lambda_l2


#python test_MAIN_scheduler.py --index dev --run_type E --bn_weights_lambda_L2 1e-7 --uniform_lambda_L2 1e-7 --cat_max_norm 100 --global_bias_lambda_L2 0 --cat_bias_lambda_L2 0 --vae_scaling_KL 1e-2 --global_bias_lambda_L1 0 --cat_bias_lambda_L1 0 --vae_prior_mu 0 --vae_prior_sigma 1 --adj_scaling_KL 0 --adj_prior_mu 0 --adj_prior_sigma 0.2 --loss_type MSE --per_condition_loss true --cat_bias_orthogonality_scaler 0 --cat_max_penalty_weight 12 --cat_b_adv 2 --pert_max_penalty_weight 8 --pert_b_adv 3.5 --network_noise_scale 0.01 --min_network_noise 0.0025 --include_gradient_noise_vae true --include_gradient_noise_embedding true --constant_gradient_noise true --gradient_noise_scale 1e-9 --lr_period 4 --reset_state false --train_batch 500 --initialize_fc true --generator_dropout_rate 0.7 --discriminator_dropout_rate 0.3 --discriminator_batch_momentum 0 --spectral_norm false --discriminator_lambda_L2 1e-3 --discriminator_bionet_activation false --smooth_labels true --gradient_ascent true --n_adversarial_start 200 --n_discriminator_train 5 --lr_decay 0.9 --vae_lambda_l2 1e-5


# 

# In[1]:


# fn = 'dev'#'trash'
# run_type = 'E'
# fn += run_type
# # seed = 3

# # loo = True

# # defaults
# bn_weights_lambda_L2 = 1e-7
# uniform_lambda_L2 = 1e-7
# cat_max_norm = 100
# global_bias_lambda_L2 = 0
# cat_bias_lambda_L2 = 0
# vae_scaling_KL = 1e-2
# global_bias_lambda_L1 = 0
# cat_bias_lambda_L1 = 0
# vae_prior_mu = 0
# vae_prior_sigma = 1
# adj_scaling_KL = 0
# adj_prior_mu = 0
# adj_prior_sigma = 0.2

# loss_type = 'MSE'
# per_condition_loss = True

# cat_bias_orthogonality_scaler = 0
# cat_b_adv = 2
# cat_max_penalty_weight = 12
# pert_b_adv = 3.5
# pert_max_penalty_weight = 8

# network_noise_scale=0.01 
# min_network_noise=0.0025
# include_gradient_noise_vae=True 
# include_gradient_noise_embedding=True
# constant_gradient_noise=True 
# gradient_noise_scale=1e-9
# lr_period = 4

# reset_state = True

# train_batch = 500
# initialize_fc = True
# generator_dropout_rate = 0.7
# discriminator_dropout_rate = 0.3
# spectral_norm = False
# discriminator_batch_momentum = None #if spectral_norm else 0.01
# discriminator_lambda_L2 = 0 if spectral_norm else 1e-3
# discriminator_bionet_activation = False
# smooth_labels = True
# gradient_ascent = True

# n_adversarial_start = 200
# n_discriminator_train = 5

# lr_decay = 0.9
# vae_lambda_l2 = 1e-5


# In[2]:


run_types = {'A': (1, True),
            'B': (3, True), 
            'C': (7, True), 
            'D': (10, True), 
            'E': (1, False)
            }
seed, loo = run_types[run_type]


# In[3]:


visualize = False
mod_type = 'default'
subset = False
short_run = False
drop_low_counts = True
seed_split = 888

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

a_scale = 10
b_scale = 2

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


# In[5]:


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

import warnings
from scipy.sparse import SparseEfficiencyWarning
with warnings.catch_warnings():
    warnings.simplefilter("ignore", SparseEfficiencyWarning)


# In[6]:


import sys
sclembas_path = '/home/hmbaghda/Projects/scLEMBAS'
sys.path.insert(1, os.path.join(sclembas_path))
from scLEMBAS import io

# from scLEMBAS.model.train_dev_mu_regularizer import TrainSC as TrainSCDevMu
# from scLEMBAS.model.train_dev_weights_standard import TrainSC as TrainSCDevWstandard
from scLEMBAS.model.train import TrainSC
from scLEMBAS.preprocess import discriminator_weight_curve, embed_tf_activity, get_alignment_score


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


# In[7]:


n_cores = 12
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)

data_path = '/nobackup/users/hmbaghda/scLEMBAS/analysis'
device = "cuda" if torch.cuda.is_available() else "cpu"


# In[8]:


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


# In[11]:


tf_adata.obs.seurat_annotations.nunique()


# In[ ]:





# In[10]:


if drop_low_counts:
    drop_ct = tf_adata.obs.seurat_annotations.value_counts().index.tolist()[-3:]
    tf_adata = tf_adata[~tf_adata.obs.seurat_annotations.isin(drop_ct)]
    adata = adata[tf_adata.obs_names,:]


# # 1. Create a novel train-test split:

# In[11]:


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


# In[12]:


contingency_table = pd.crosstab(tf_adata.obs['stim'], tf_adata.obs['seurat_annotations'], 
                                margins=True, margins_name="Total")
contingency_table = contingency_table.T.sort_values(by = 'Total').T
bins = pd.qcut(contingency_table.T.Total, q = 4, labels = False)


# In[13]:


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


# In[46]:


condition_proportions = tf_adata.obs['condition'].value_counts()
condition_proportions.loc[train_cond].sort_values(ascending = True)


# In[47]:


condition_proportions.loc[test_cond].sort_values(ascending = True)


# # 2. Subset data

# In[48]:


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
    


# In[49]:


tf_adata


# In[50]:


condition_proportions = tf_adata.obs['condition'].value_counts()
condition_proportions.loc[train_cond].sort_values(ascending = True)


# In[51]:


condition_proportions.loc[test_cond].sort_values(ascending = True)


# # 3. Run training

# In[ ]:


if not short_run:
#     cat_max_penalty_weight = 7.75
#     cat_b_adv = 0.6
    max_epochs = 600
else:
#     cat_max_penalty_weight = 8
#     cat_b_adv = 1.5
    max_epochs = 250 
    
# max_epochs = 10   

# if subset:
# #     batch_factor = 1 if n_fraction <= 0.2 else 3
# #     train_batch = int(np.round(n_train/batch_factor))
#     batch_factor = 2 if n_fraction <= 0.1 else 3
#     train_batch = int(np.round(n_train/batch_factor))
# else:
#     train_batch = 1024


# In[2]:


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


# In[53]:


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
# if mod_type in ['default', 'total_bias_scaler', 'mu_bias_regularizer', 'mu_bias_scaler']: 
#     bionet_params['cat_max_norm'] = cat_max_norm
# elif mod_type in ['global_bias_scaler', 'global_bias_regularizer']:
#     bionet_params['cat_max_norm'] = 50 # since scaling global bias, need to separately scale categorical with the regularization

    
if mod_type in ['tot_bias_scaler', 'global_bias_scaler', 'mu_bias_scaler']:
    bionet_params['signaling_weights_scaler'] = a_scale
    if mod_type == 'tot_bias_scaler':
        bionet_params['bias_tot_scaler'] = b_scale
    elif mod_type == 'global_bias_scaler':
        bionet_params['bias_global_scaler'] = b_scale
    elif mod_type == 'mu_bias_scaler':
        bionet_params['bias_mu_scaler'] = b_scale
        


# In[54]:


# training parameters
batch_params_default = {'test_batch_size':round(len(test_cells)/2)}
spectral_radius_params = {'n_probes_spectral': 5, 
                          'power_steps_spectral': 5, #50, 
                          'subset_n_spectral': 5} #10}

if mod_type != 'standardized_weights':
    target_spectral_radius = 0.9
else: 
    target_spectral_radius = 30

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

# if mod_type.endswith('_regularizer'):
#     regularization_params_default['bn_weights_lambda_l2'] = 1e-12 # decrease adj matrix regularization
#     regularization_params_default['global_bias_lambda_L2'] = 1e-5 # increase global bias regularization


# In[55]:


max_lr = 0.001

if max_epochs > n_adversarial_start:
    cat_discriminator_penalty_weight = discriminator_weight_curve(n_epochs = max_epochs - n_adversarial_start,
                                                              min_penalty_weight = 0.1,
                                                              max_penalty_weight = cat_max_penalty_weight,
                                                              a = 1,
                                                              b = cat_b_adv, 
                                                              curve_type = 'power')

    pert_discriminator_penalty_weight = discriminator_weight_curve(n_epochs = max_epochs - n_adversarial_start,
                                                                   min_penalty_weight = 0.1,
                                                                   max_penalty_weight = pert_max_penalty_weight,
                                                                   a = 1,
                                                                   b = pert_b_adv, 
                                                                   curve_type = 'power')
else:
    cat_discriminator_penalty_weight, pert_discriminator_penalty_weight = 0,0



lr_params = generate_lr_params(n_epochs = max_epochs, 
                               max_lr = max_lr, lr_scaling_factor = 10, lr_decay = lr_decay, role = 'scl')
cat_discriminator_params = generate_discriminator_params(n_epochs = max_epochs, 
                                                         max_lr = max_lr, 
                                                         discriminator_penalty_weight = cat_discriminator_penalty_weight, 
                                                         lr_scaling_factor = 10, lr_decay = lr_decay)

vae_params = {**{'prior_mu': vae_prior_mu,
              'prior_sigma': vae_prior_sigma,
              'lambda_l2': vae_lambda_l2,
              'scaling_KL': vae_scaling_KL, #1e-2
              'optimizer': torch.optim.Adam}, 
                 **generate_lr_params(n_epochs = max_epochs, 
                                     max_lr = max_lr, 
                                     lr_scaling_factor = 10, lr_decay = lr_decay, 
                                     role = 'discriminator') # generator
             }
del vae_params['max_epochs']



pert_discriminator_params = cat_discriminator_params.copy()
pert_discriminator_params['discriminator_penalty_weight'] = pert_discriminator_penalty_weight

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


gc.collect()
torch.cuda.empty_cache()  # Clears the cached memory
torch.cuda.ipc_collect()  # Collects unused memory blocks


# # Start

# In[62]:


# trainerA = io.read_pickled_object(os.path.join(data_path, 'trash', 'mini4_fcE' + '_feb_trainer.pickle'))
# modA = trainerA.mod


# In[66]:


# import torch

# def models_allclose(model1, model2, rtol=1e-5, atol=1e-8):
#     for p1, p2 in zip(model1.parameters(), model2.parameters()):
#         if not torch.allclose(p1, p2, rtol=rtol, atol=atol):
#             return False
#     return True


# are_equal = models_allclose(mod, modA)
# print("Models are equal:", are_equal)


# # End

# # Checkpoint: load the object

# In[29]:


trainer = io.read_pickled_object(os.path.join(data_path, 'trash', fn + '_feb_trainer.pickle'))
mod = trainer.mod

test_cells = trainer.X_test.index.tolist()
train_cells = trainer.X_train.index.tolist()

train_cond = sorted(tf_adata[train_cells, :].obs.condition.unique())
train_conds = train_cond
test_cond = sorted(tf_adata[test_cells, :].obs.condition.unique())
test_conds = test_cond
seed = mod.seed

test_cell_types = sorted(pd.Series([ct.split('^')[1] for ct in test_conds]).drop_duplicates(keep = 'first').tolist())
train_cell_types = pd.Series([ct.split('^')[1] for ct in train_conds]).drop_duplicates(keep = 'first').tolist()
train_cell_types = sorted(set(train_cell_types).difference(test_cell_types))


# In[30]:


if torch.isinf(mod.signaling_network.weights).any():
    raise ValueError('Exploding gradients')


# In[31]:


print('Index: {}'.format(fn))
# if not any(letter in fn for letter in ['A', 'B', 'C', 'D', 'E']):
loo_ = True if trainer.X_test.shape[0] < 1000 else False
if not loo_:
    run_type_ = 'E'
else:
    seed_ = mod.seed
    stim_ = test_conds[0].split('^')[0]

    if stim_ == 'CTRL':
        if seed_ == 3:
            run_type_ = 'B'
        elif seed_ == 1:
            run_type_ = 'A'
    elif stim_ == 'STIM':
        if seed_ == 7:
            run_type_ = 'C'
        elif seed == 10:
            run_type_ = 'D'
if run_type_ != run_type:
    raise ValueError('Problem in run type logic')
print('Run type: ' + run_type)

print()
print('-----Adj REGULARIZATIONS-----------')
print('adj reg: {:.3E}'.format(trainer.hyper_params['bn_weights_lambda_L2']))
print('uniform reg: {:.3E}'.format(trainer.hyper_params['uniform_lambda_L2']))

if not ('adj_scaling_KL' in trainer.hyper_params):
    for hp in ['adj_scaling_KL', 'adj_prior_mu', 'adj_prior_sigma']:
        trainer.hyper_params[hp] = None
askl = trainer.hyper_params['adj_scaling_KL']
print('Adj scaling KL: {:.3E}'.format(trainer.hyper_params['adj_scaling_KL']) if askl is 
      not None else 'Adj scaling KL: None')
print('Adj KL Dist: N({:.2f}, {:.2f})'.format(trainer.hyper_params['adj_prior_mu'], 
                                         trainer.hyper_params['adj_prior_sigma']) if askl is not None else 'Adj KL Dist: None')

print()
print('-----Global Bias REGULARIZATIONS-----------')
if not ('vae_prior_mu' in trainer.hyper_params):
    trainer.vae_learning['params']['prior_mu'] = 0
    trainer.vae_learning['params']['prior_sigma'] = 1
print('Global Bias scaling KL: {:.3E}'.format(trainer.vae_learning['params']['scaling_KL']))
print('Global Bias KL Dist: N({:.2f}, {:.2f})'.format(trainer.vae_learning['params']['prior_mu'], 
                                         trainer.vae_learning['params']['prior_sigma']))


for hp in ['global_bias_lambda_L2', 'global_bias_lambda_L1', 'cat_bias_lambda_L2', 
          'cat_bias_lambda_L1', 'cat_bias_orthogonality_scaler']:
    if not (hp in trainer.hyper_params):
        trainer.hyper_params[hp] = 0
        
print('global bias L2 reg: {:.3E}'.format(trainer.hyper_params['global_bias_lambda_L2']))
print('global bias L1 reg: {:.3E}'.format(trainer.hyper_params['global_bias_lambda_L1']))
# print('VAE reg: {:.3E}'.format(trainer.vae_learning['params']['lambda_l2']))

print()
print('-----Categorical Bias REGULARIZATIONS-----------')
print('cat max norm: {}'.format(mod.signaling_network.bionet_params['cat_max_norm']))
print('cat bias L2 reg: {:.3E}'.format(trainer.hyper_params['cat_bias_lambda_L2']))
print('cat bias L1 reg: {:.3E}'.format(trainer.hyper_params['cat_bias_lambda_L1']))
print('cat bias orthogonality scaler: {:.3f}'.format(trainer.hyper_params['cat_bias_orthogonality_scaler']))

print()
print('-----LOSS FUNCTION-----------')
if 'per_condition_loss' in trainer.__dict__.keys():
    new_loss = True
    loss_type = trainer._prediction_loss_name
else:
    new_loss = False
    loss_type = 'EMD'
    
if new_loss and trainer.per_condition_loss:
    print('Loss was calculated per-condition')
else:
    print('Loss was calculated on all conditions simultaneously')
print('The loss function used is: ' + loss_type)

print()
print('-----NOISE-----------')
if 'include_gradient_noise_vae' not in trainer.hyper_params:
    _noise_params = [0.01, 0, True, True, True, 1e-9]
else: 
    _noise_params = [trainer.hyper_params['network_noise_scale'], 
                    trainer.hyper_params['min_network_noise'], 
                    trainer.hyper_params['include_gradient_noise_vae'], 
                    trainer.hyper_params['include_gradient_noise_embedding'], 
                    trainer.hyper_params['constant_gradient_noise'], 
                    trainer.hyper_params['gradient_noise_scale']]
_noise_params = [nl.item() if type(nl) == torch.Tensor else nl for nl in _noise_params]
print('Network Noise: scaler: {:.3e} | min: {:.3e}'.format(*_noise_params[:2]))
print('Gradient Noise: vae included: {} | cat_embedding_included: {}'.format(*_noise_params[2:4]))
print('Gradient Noise: scaler included: {:.3e} | constant: {}'.format(*_noise_params[4:][::-1]))

print()
print('-----Adversarial Tuning-----------')
print('Train batch size: {}'.format(trainer.hyper_params['train_batch_size']))
print()

sl = False
if 'smooth_labels' in trainer.pert_discriminator['discriminator'].__dict__:
    sl = trainer.pert_discriminator['discriminator'].smooth_labels
ga = False
if 'gradient_ascent' in trainer.__dict__ and trainer.gradient_ascent:
    ga = True

print('Label smoothing: {}'.format(sl))
print('Gradient ascent: {}'.format(ga))

print()
print('Min/max Discriminator LR: ({:.2E}, {:.2E})'.format(trainer.pert_discriminator['params']['minimum_learning_rate'],
                                                        trainer.pert_discriminator['params']['maximum_learning_rate']))
print('Discriminator dropout rate: {}'.format(trainer.pert_discriminator['params']['dropout_rate']))
bm = False if trainer.pert_discriminator['params']['batch_momentum'] is None else True
print('Discriminator batch normalization: {}'.format(bm))
sn = False
if 'spectral_norm' in trainer.pert_discriminator['params']:
    sn = trainer.pert_discriminator['params']['spectral_norm']
print('Discriminator spectral normalization: {}'.format(sn))
print('Discriminator L2 regularization: {:.2E}'.format(trainer.pert_discriminator['params']['discriminator_lambda_L2']))
print()
print('Generator dropout rate: {}'.format(trainer.mod.signaling_network.bionet_params['vae_dropout_rate']))

print(())
n_adversarial_start, n_discriminator_train = 0, 1
if hasattr(trainer, 'n_adversarial_start'):
    n_adversarial_start = trainer.n_adversarial_start
if hasattr(trainer, 'n_discriminator_train'):
    n_discriminator_train = trainer.n_discriminator_train
print('Adversarial start epoch: {}'.format(n_adversarial_start))
print('Generator train frequency: {}'.format(n_discriminator_train))   



print()
print('-----OTHER-----------')
print('epochs: {}'.format(trainer.hyper_params['max_epochs']))
print('seed: {}'.format(mod.seed))
print('test cells: {}'.format(trainer.X_test.shape))
print('Reset optimizer epoch: {}'.format(trainer.hyper_params['reset_optimizer_epoch']))


# In[32]:


if 'cat_discriminator' not in trainer.__dict__.keys():
    trainer.cat_discriminator = trainer.discriminator

fig, ax = plt.subplots(ncols = 2, figsize = (13,5))
sns.lineplot(trainer.cat_discriminator['params']['discriminator_penalty_weight'], ax = ax[0])
ax[0].vlines(n_adversarial_start, ymin = 0, ymax = ax[0].get_ylim()[1], color = 'red', linestyle = '--')
ax[0].set_title('Categorical Discriminator')

if 'pert_discriminator' in trainer.__dict__.keys():
    sns.lineplot(trainer.pert_discriminator['params']['discriminator_penalty_weight'], ax = ax[1])
ax[1].vlines(n_adversarial_start, ymin = 0, ymax = ax[1].get_ylim()[1], color = 'red', linestyle = '--')
ax[1].set_title('Perturbation Discriminator')

for i in range(2):
    ax[i].set_xlabel('Epochs')
    ax[i].set_ylabel('Discriminator Penalty Weight')
    
fig.tight_layout();


# In[33]:


train_stats_df = trainer.stats['train'].copy()
train_stats_df = train_stats_df.groupby('epoch').mean().reset_index() # delete this

# formatting new/old versions
if 'sn_param_reg_loss' in train_stats_df.columns:
    train_stats_df.rename(columns = {'sn_param_reg_loss': 'sn_param_reg_tot_loss'}, 
                      inplace = True)
if 'output_param_reg_loss' in train_stats_df.columns:
    train_stats_df.rename(columns = {'output_param_reg_loss': 'output_param_reg_tot_loss'}, 
                      inplace = True)
    
if 'kl_divergence' in train_stats_df.columns:
    train_stats_df.rename(columns = {'kl_divergence': 'global_bias_kl_divergence'}, 
                      inplace = True)
    
if 'sn_param_reg_weights_kl_divergence' not in train_stats_df.columns:
    train_stats_df['sn_param_reg_weights_kl_divergence'] = 0
    
    
sn_param_loss_cols = ['sn_param_reg_weights_L2_loss',
             'sn_param_reg_weights_kl_divergence',
             'sn_param_reg_global_bias_L2_loss', 
             'sn_param_reg_global_bias_L1_loss',
             'sn_param_reg_cat_bias_L2_loss', 
             'sn_param_reg_cat_bias_L1_loss',
             'sn_param_reg_cat_bias_orthogonality']
for col in sn_param_loss_cols + ['output_param_reg_weights_loss', 'output_param_reg_bias_loss']:
    if col not in train_stats_df.columns:
        train_stats_df[col] = 0 # not necessarily accurate but helps print things out
        
        
for col in ['discriminator_learning_rate', 'discriminator_loss_total',
           'discriminator_loss_prediction', 'discriminator_param_reg_loss']:
    if col in train_stats_df.columns:
        train_stats_df.rename(columns = {col: 'cat_'+ col}, 
                  inplace = True)
        
for col in ['vae_grad_l2_norm', 'pert_discriminator_learning_rate', 'pert_discriminator_loss_total',
           'pert_discriminator_loss_prediction', 'pert_discriminator_param_reg_loss', 
           'cat_seurat_annotations_discriminator_grad_l2_norm', 
           'pert_discriminator_grad_l2_norm']:
    if col not in train_stats_df.columns:
        train_stats_df[col] = 0

# if 'test' in trainer.stats:
#     test_stats_df = trainer.stats['test'].copy()
#     test_stats_df = test_stats_df.groupby('epoch').mean().reset_index() # DELETE THIS


# # 4. Look at the loss curves:

# In[36]:


viz_df = train_stats_df.copy()
viz_df.epoch -= 1
fig, ax = plt.subplots(ncols = 5, figsize = (15, 4))

sns.lineplot(data = viz_df, x = 'epoch', y = 'learning_rate', ax = ax[0])
ax[0].set_ylabel('scLEMBAS Learning Rate')

sns.lineplot(data = viz_df, x = 'epoch', y = 'cat_discriminator_learning_rate', ax = ax[1])
ax[1].set_ylabel('Categorical Discriminator Learning Rate')

sns.lineplot(data = viz_df, x = 'epoch', y = 'pert_discriminator_learning_rate', ax = ax[2])
ax[2].set_ylabel('Perturbation Discriminator Learning Rate')

sns.lineplot(data = viz_df, x = 'epoch', y = 'vae_learning_rate', ax = ax[3])
ax[3].set_ylabel('Generator (VAE) Learning Rate')

sns.lineplot(data = train_stats_df, x = 'epoch', y = 'n_moa_violations', ax = ax[4])
ax[4].set_ylabel('MOA Violations')

for i in [1, 2, 3]:
    ax[i].vlines(n_adversarial_start, 
                 ymin = 0, 
                 ymax = ax[i].get_ylim()[1], color = 'red', linestyle = '--')


fig.tight_layout()
# plt.savefig(os.path.join(data_path, 'trash', fn + '_loss_epochs' + '.png'), dpi=300, bbox_inches='tight')


# ## Discriminator and Generator Gradient
# 
# - Discriminator: Bad if L2 norm is > 100 (see number [10](https://github.com/soumith/ganhacks))
# - Generator: if gradient disappears quick (standard GANs), not able to learn anything

# In[37]:


if visualize:
    fig, ax = plt.subplots(ncols = 3, figsize = (12, 4))

    sns.lineplot(data = train_stats_df, x = 'epoch', 
                 y = 'cat_seurat_annotations_discriminator_grad_l2_norm', ax = ax[0])
    ax[0].set_ylabel('Cat Discriminator Gradient L2 Norm')

    sns.lineplot(data = train_stats_df, x = 'epoch', 
                 y = 'pert_discriminator_grad_l2_norm', ax = ax[1])
    ax[1].set_ylabel('Pert Discriminator Gradient L2 Norm')
    
    sns.lineplot(data = train_stats_df, x = 'epoch', 
                 y = 'vae_grad_l2_norm', ax = ax[2])
    ax[2].set_ylabel('Generator Gradient L2 Norm')

    fig.tight_layout()


# ## Input network noise:

# In[ ]:


if visualize:
    import copy
    trainer_2 = copy.deepcopy(trainer)

    noise_tracker = []
    noise_tracker_unclipped = []
    for e in trange(trainer_2.hyper_params['max_epochs']):
        cur_lr = trainer_2.prediction_optimizer.param_groups[0]['lr']
        noise_scale = trainer_2.hyper_params['network_noise_scale']*cur_lr/trainer_2.lr_scheduler.max_lr
        noise_tracker_unclipped.append(noise_scale)
        noise_tracker.append(max(noise_scale, trainer_2.hyper_params['min_network_noise']))
        trainer_2.lr_scheduler.step()

    fig, ax = plt.subplots(ncols = 2, figsize = (10, 5))

    sns.lineplot(noise_tracker_unclipped, ax = ax[0])
    ax[0].set_title('Network Noise Scale without Clipping')

    sns.lineplot(noise_tracker, ax = ax[1])
    ax[1].set_title('Network Noise Scale with Clipping')

    for i in range(2):
        ax[i].set_ylabel('Noise scaler')
        ax[i].set_xlabel('Epoch')
    fig.tight_layout()
    ("")


# ## Gradient noise

# In[ ]:


if 'gradient_noise' in trainer.stats and visualize:
    gradient_noise = trainer.stats['gradient_noise'].copy()
    gradient_noise = gradient_noise.groupby('epoch').mean().reset_index() # DELETE THIS


    noise_layers = [col for col in gradient_noise.columns if col not in ['epoch', 'batch_index']]
    noise_layers = sorted(set([nl.split('_norm')[0].split('_noise_scale')[0] for nl in noise_layers]))

    ncols = 3
    nrows = math.ceil(len(noise_layers)/3)

    fig, axes = plt.subplots(ncols = ncols, nrows = nrows, squeeze=False,
                             figsize = (5.1*ncols, 5.1*nrows))
    ax = axes.flatten()

    for i, noise_layer in enumerate(noise_layers):

        vcols=[col for col in gradient_noise.columns if col == 'epoch' or col.startswith(noise_layer)]
        viz_df = gradient_noise[vcols].copy()

        viz_df = pd.melt(viz_df, id_vars = 'epoch', var_name='value_type', value_name='value_amount')
        viz_df.value_type = pd.Categorical(viz_df.value_type.apply(lambda x: x.split(noise_layer + '_')[1]), 
                                           categories = ['norm', 'noise_scale'], 
                                           ordered = True)
        sns.lineplot(data = viz_df, x = 'epoch', y = 'value_amount', hue = 'value_type', 
                       ax = ax[i])
        ax[i].set_yscale('log')
        if i != 0:
            ax[i].legend().remove()

        title_length = 28
        if len(noise_layer) > title_length:
            title = '\n'.join([noise_layer[i_:i_+28] for i_ in range(0, len(noise_layer), 28)])
        else:
            title = noise_layer
        ax[i].set_title(title)
    fig.tight_layout()
    ("")


# ## 4.1 Loss During Training

# In[25]:


if visualize: 
    fig, axes = plt.subplots(ncols = 2, nrows = 2, figsize = (10, 10), 
                            constrained_layout=True)
    ax = axes.flatten()

    colors = [
        "#d62728",  # Red
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Olive
        "#17a589",  #  Teal 
        "#6baed6",  # light blue
        "#ff9896"   # Light red
    ]
    palette = sns.color_palette(colors)

    # prediction loss
    i = 0

    loss_cols_main = [
           'train_loss_prediction', 'sign_reg_loss',
           'stability_reg_loss', 'uniform_reg_loss', 'input_param_reg_loss',
           'sn_param_reg_tot_loss', 'output_param_reg_tot_loss', 'vae_param_reg_loss', 
        'global_bias_kl_divergence']


    viz_df = train_stats_df[['epoch'] + loss_cols_main].copy()
    viz_df['total_train_loss_no_adverserial'] = viz_df[loss_cols_main].sum(axis = 1)
    viz_df = viz_df.melt(id_vars = ['epoch'], var_name = 'loss_type', value_name = 'loss')
    viz_df.loss_type = pd.Categorical(viz_df.loss_type, ordered = True, 
                                      categories=['total_train_loss_no_adverserial'] + loss_cols_main)
    viz_df_all = viz_df.copy()
    viz_df = viz_df[viz_df.loss_type.isin(['train_loss_prediction', 'total_train_loss_no_adverserial'])]
    viz_df.loss_type = viz_df.loss_type.cat.remove_unused_categories()

#     if 'test' in trainer.stats:
#         sns.lineplot(data = test_stats_df, x = 'epoch', y = 'test_loss_prediction',
#                      color = '#6baed6', ax = ax[i], linestyle = '--')
    sns.lineplot(data = viz_df, x = 'epoch', y = 'loss', hue='loss_type', palette = palette, ax = ax[i])
    ax[i].legend(loc='lower center', bbox_to_anchor=(0.5, 1.1), ncol = 3, fontsize = 'small')
    ax[i].set_title('Train Loss')
    ax[i].legend().remove()

    # all losses
    i = 1
    viz_df = viz_df_all.copy()
    del viz_df_all

    sns.lineplot(data = viz_df, x = 'epoch', y = 'loss', hue='loss_type', palette = palette, ax = ax[i])
    # ax[i].legend(loc='lower center', bbox_to_anchor=(-0.5, 1.1), ncol = 3, fontsize = 'small')
    handles_1, labels_1 = ax[i].get_legend_handles_labels()
    ax[i].legend().set_visible(False)
    ax[i].set_title('Train Loss - Individual Components')
    ax[i].set_yscale('log')


    # component losses
    i = 2


    loss_cols = sn_param_loss_cols + ['sn_param_reg_tot_loss']
    palette = ['tab:blue', 'tab:red', 'tab:purple', 'tab:orange', 'tab:cyan', 'tab:brown', 'tab:olive',
               (0.8901960784313725, 0.4666666666666667, 0.7607843137254902)]


    # linestyles = ['--', '-.', 'dotted', 'solid']
    # linestyles = dict(zip(loss_cols, linestyles))

    viz_df = train_stats_df[['epoch'] + loss_cols].copy()
    viz_df = viz_df.melt(id_vars=['epoch'], var_name='loss_type', value_name='loss')
    viz_df.loss_type = pd.Categorical(viz_df.loss_type, ordered=True, categories=loss_cols)

    zeros = viz_df.groupby('loss_type').loss.apply(lambda x: (x == 0).all())
    zeros = np.array(zeros.index)[np.where(zeros)]
    viz_df = viz_df[~viz_df.loss_type.isin(zeros)]
    viz_df.loss_type = viz_df.loss_type.cat.remove_unused_categories()
    palette = [col for i,col in enumerate(palette) if loss_cols[i] in viz_df.loss_type.cat.categories]
    palette = dict(zip(viz_df.loss_type.cat.categories, palette))

    for lt in viz_df.loss_type.cat.categories:
        viz_df_ = viz_df[viz_df.loss_type == lt]
        sns.lineplot(data = viz_df_, x = 'epoch', y = 'loss', 
                     color = palette[lt], # linestyle = linestyles[lt],
                     ax = ax[i])
    ax[i].set_title('Signaling Network Parameter Regularizations')
    ax[i].set_yscale('log')
    # legend_handles = [Line2D([0], [0], color=palette[lt],linestyle=linestyles[lt], label=lt) for lt in loss_cols]
    legend_handles = [Line2D([0], [0], color=palette[lt], label=lt) for lt in viz_df.loss_type.cat.categories]

    ax[i].legend(handles=legend_handles, loc = 'best')

    i = 3
    loss_cols = ['output_param_reg_weights_loss', 'output_param_reg_bias_loss', 'output_param_reg_tot_loss']
    palette = ['tab:purple', 'tab:brown', 
               (0.4980392156862745, 0.4980392156862745, 0.4980392156862745)]
    linestyles = ['--', 'dotted', 'solid']
    palette = dict(zip(loss_cols, palette))
    linestyles = dict(zip(loss_cols, linestyles))

    viz_df = train_stats_df[['epoch'] + loss_cols].copy()
    viz_df = viz_df.melt(id_vars=['epoch'], var_name='loss_type', value_name='loss')
    viz_df.loss_type = pd.Categorical(viz_df.loss_type, ordered=True, categories=loss_cols)

    for lt in loss_cols:
        viz_df_ = viz_df[viz_df.loss_type == lt]
        sns.lineplot(data = viz_df_, x = 'epoch', y = 'loss', 
                     color = palette[lt], linestyle = linestyles[lt],
                     ax = ax[i])
    ax[i].set_title('Output Layer Parameter Regularizations')
    ax[i].set_yscale('log')
    legend_handles = [Line2D([0], [0], color=palette[lt], linestyle=linestyles[lt], label=lt) for lt in loss_cols]
    ax[i].legend(handles=legend_handles, loc = 'best')

    fig.legend(handles_1, labels_1,
               loc='lower center', bbox_to_anchor=(0.5, 1.01), 
               ncol=3, fontsize='small', title='Loss Type')

    fig.tight_layout()  # reserve space at bottom
    # plt.savefig(os.path.join(data_path, 'trash', fn + '_loss_all' + '.png'), dpi=300, bbox_inches='tight')


# ## 4.2 Train - Test Comparison
# 
# To directly compare train with test with EMD Loss, we need to downsample to the same # of cells. This is because EMD Loss increases with decreasing number of samples (less information) given two datasets that are otherwise drawn from the same distribution. 
# 
# The first two panels below compare the loss on the downsampled data to the full batch size for each of train and test. We expect that the downsampled curve should follow the same trend as the full batch, but shifted up in the y-axis to account for the fact that there are fewer cells. The exception here is that if a loss calculation had fewer cells than the minimum to downsample to, in which case the losses should be the same. These cases are less directly comparable between train and test, though the discrepancy due to sample size should be very small since we set n_eval to a small number. The right panel allows for the comparison of the train and test on the downsampled data. 
# 
# Note that in the first panel, the full batch size loss calculation on the train data will be slightly different than that of the train loss curve in section 4.1 because it is just a simple forward pass without adding noise to the weights or gradients, as is done during training. 

# In[ ]:


if visualize and 'test' in trainer.stats:

    train_eval_df = trainer.stats['train_eval'].copy()
    test_df = trainer.stats['test'].copy()

    # TODO: remove this prior to full visualization
    train_eval_df = train_eval_df.groupby(['epoch']).mean().reset_index()
    test_df = test_df.groupby(['epoch']).mean().reset_index()

    train_eval_df['loss_type'] = 'train_eval'
    test_df['loss_type'] = 'test'
    
    palette = sns.color_palette([
        "#d62728",  # Red
        "#e377c2",  # Pink
        "#1f77b4",  # Blue
        "#6baed6",  # light blue
    ])

    if new_loss and loss_type == 'EMD': # only for EMD loss
        print('Train batches were between {} and {} cells'.format(train_eval_df.size_full.min(), train_eval_df.size_full.max()))
        print('Test batches were between {} and {} cells'.format(test_df.size_full.min(), test_df.size_full.max()))

        print('Downsampled train estimates were between {} and {} cells'.format(train_eval_df.size_downsample.min(), train_eval_df.size_downsample.max()))
        print('Downsampled test estimates were between {} and {} cells'.format(test_df.size_downsample.min(), test_df.size_downsample.max()))

        
        fig, ax = plt.subplots(ncols=3, figsize = (15, 5))

        viz_df = train_eval_df[['epoch', 'loss_full', 'loss_downsample']].melt(id_vars='epoch', 
                                                                              var_name='loss_type', 
                                                                              value_name='EMD Loss')
        viz_df.loss_type = pd.Categorical(viz_df.loss_type.apply(lambda x: x.split('loss_')[1]), 
                                          ordered = True, categories = ['full', 'downsample'])
        sns.lineplot(data = viz_df, x = 'epoch', y = 'EMD Loss', hue = 'loss_type', palette = palette, ax = ax[0])
        ax[0].set_title('Train Evaluation Loss')

        viz_df = test_df[['epoch', 'loss_full', 'loss_downsample']].melt(id_vars='epoch', 
                                                                              var_name='loss_type', 
                                                                              value_name='EMD Loss')
        viz_df.loss_type = pd.Categorical(viz_df.loss_type.apply(lambda x: x.split('loss_')[1]), 
                                          ordered = True, categories = ['full', 'downsample'])
        sns.lineplot(data = viz_df, x = 'epoch', y = 'EMD Loss', hue = 'loss_type', palette = palette[2:], ax = ax[1])
        ax[1].set_title('Test Evaluation Loss')

        viz_df = pd.concat([train_eval_df, test_df], axis = 0)
        viz_df.loss_type = pd.Categorical(viz_df.loss_type, ordered = True, categories = ['train_eval', 'test'])
        viz_df.rename(columns = {'loss_downsample': 'Downsampled EMD Loss'}, inplace = True)
        sns.lineplot(data = viz_df, x = 'epoch', y = 'Downsampled EMD Loss', hue = 'loss_type', palette = [palette[1], palette[3]],
                     ax = ax[2])
        ax[2].set_title('Train-Test Comparison')

        fig.tight_layout()
        ("")
    else:
        fig, ax = plt.subplots(figsize = (5, 5))
        
        viz_df = pd.concat([train_eval_df, test_df], axis = 0)
        viz_df.loss_type = pd.Categorical(viz_df.loss_type, ordered = True, categories = ['train_eval', 'test'])
        sns.lineplot(data = viz_df, x = 'epoch', y = 'loss_full', hue = 'loss_type', palette = [palette[1], palette[3]],
                     ax = ax)
        ax.set_title('Train-Test Comparison')
        ax.set_ylabel('MSE Loss')

        fig.tight_layout()


# # 5. Bias Adverserial Assessment

# In[26]:


if visualize:
    fig, ax = plt.subplots(ncols = 3, figsize = (16.5,5))
    ax = ax.flatten()

    colors = [
        "#d62728",  # Red
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#ff9896",   # Light red
        "#1f77b4",  # Blue
        "#8c564b",  # Brown
        "#17a589",  #  Teal 
        "#bcbd22",  # Olive
        "#9467bd",  # Purple
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#6baed6",  # light blue

    ]
    palette = sns.color_palette(colors)


    # Plot 1: full model, adverserial loss
    loss_cols_main = ['sign_reg_loss',
           'stability_reg_loss', 'uniform_reg_loss', 'input_param_reg_loss',
           'sn_param_reg_tot_loss', 'output_param_reg_tot_loss', 'vae_param_reg_loss', 
        'global_bias_kl_divergence']
    loss_cols = ['train_loss_total'] + loss_cols_main + ['cat_adverserial_loss', 'pert_adverserial_loss', 'train_loss_prediction']

    viz_df = train_stats_df[['epoch'] + loss_cols].copy()
    viz_df['total_train_loss_no_adverserial'] = viz_df[loss_cols_main + ['train_loss_prediction']].sum(axis = 1)

    viz_df.drop(columns = loss_cols_main, inplace = True)
    viz_df['train_loss: total - cat_adverserial'] = viz_df.total_train_loss_no_adverserial - viz_df.cat_adverserial_loss
    viz_df['train_loss: total - pert_adverserial'] = viz_df.total_train_loss_no_adverserial - viz_df.pert_adverserial_loss
    viz_df = viz_df.melt(id_vars = ['epoch'], var_name = 'loss_type', value_name = 'loss')
    viz_df.loss_type = pd.Categorical(viz_df.loss_type, ordered = True, categories=['total_train_loss_no_adverserial',
                                                                                    'train_loss: total - cat_adverserial', 
                                                                                    'train_loss: total - pert_adverserial',
                                                                                    'train_loss_total', 
                                                                                    'train_loss_prediction',
                                                                                    'cat_adverserial_loss',
                                                                                   'pert_adverserial_loss'])



    sns.lineplot(data = viz_df, x = 'epoch', y = 'loss', hue='loss_type', palette=palette, ax = ax[0])
    ax[0].legend(loc='best')#, bbox_to_anchor=(1.05, 0.5))
    ax[0].set_ylabel('Train Loss')
    ax[0].set_title('Full Model')

    # Plot 2: full model, categorical discriminator loss
    loss_cols_disc = ['cat_discriminator_loss_total',
           'cat_discriminator_loss_prediction', 'cat_discriminator_param_reg_loss']

    viz_df = train_stats_df[['epoch'] + loss_cols_disc].copy()
    viz_df = viz_df.melt(id_vars = ['epoch'], var_name = 'loss_type', value_name = 'loss')
    viz_df.loss_type = pd.Categorical(viz_df.loss_type, ordered = True, categories=loss_cols_disc)

    sns.lineplot(data = viz_df, x = 'epoch', y = 'loss', hue='loss_type', 
                 palette = [palette[0], palette[1], palette[-2]], ax = ax[1])
    n_cat = trainer.cat_discriminator['discriminators']['seurat_annotations'].n_labels
    ax[1].axhline(y=np.log(n_cat), color='gray', linestyle='--')

    p_i = tf_adata[train_cells,:].obs.seurat_annotations.value_counts(normalize = True).values
    entropy = -np.sum(p_i * np.log(p_i))
    ax[1].axhline(y=entropy, color='dimgray', linestyle='--')

    ax[1].legend(loc='best')
    ax[1].set_ylabel('Categorical Discriminator Loss')
    ax[1].set_title('Full Model')

    # Plot 3: full model, perturbation discriminator loss
    loss_cols_disc = ['pert_discriminator_loss_total',
           'pert_discriminator_loss_prediction', 'pert_discriminator_param_reg_loss']

    viz_df = train_stats_df[['epoch'] + loss_cols_disc].copy()
    viz_df = viz_df.melt(id_vars = ['epoch'], var_name = 'loss_type', value_name = 'loss')
    viz_df.loss_type = pd.Categorical(viz_df.loss_type, ordered = True, categories=loss_cols_disc)

    sns.lineplot(data = viz_df, x = 'epoch', y = 'loss', hue='loss_type', 
                 palette = [palette[0], palette[2], palette[-2]], ax = ax[2])
    n_cat = trainer.pert_discriminator['discriminator'].n_labels
    ax[2].axhline(y=np.log(n_cat), color='gray', linestyle='--')


    p_i = tf_adata[train_cells,:].obs.stim.value_counts(normalize = True).values
    entropy = -np.sum(p_i * np.log(p_i))
    ax[2].axhline(y=entropy, color='dimgray', linestyle='--')

    ax[2].legend(loc='best')
    ax[2].set_ylabel('Perturbation Discriminator Loss')
    ax[2].set_title('Full Model')


    fig.tight_layout()
    # plt.savefig(os.path.join(data_path, 'trash', fn + '_loss_adverserial' + '.png'), dpi=300, bbox_inches='tight')
    ("")


# In[155]:


cf_map = {'in_distribution': train_cells}
counterfactual_types = list(cf_map.keys()) + ['opposite']


# ## 5.1 Test Biases

# In[156]:


biases_res = {}
for counterfactual_type in ['opposite']:#, 'in_distribution']:
    biases_res[counterfactual_type] = {}
    biases = get_prediction(mod = mod,
                                      train_cells = train_cells,
                                      tf_adata = tf_adata,
                                      train_mode = False,
                                      counterfactual = True,
                                      remove_type = 'none',
                                      return_bias = True, 
                           max_cells = 5000)
    bias_global, bias_mu, bias_sigma, bias_cats, bias_tot, obs = biases
    
    biases_res[counterfactual_type]['adverserial'] = (bias_global, bias_mu, bias_sigma, bias_cats, bias_tot)
    biases_res[counterfactual_type]['obs'] = obs
    del biases
    torch.cuda.empty_cache()


# The categorical embeddings has a few different visualizations: We show all the learned embeddings (Everything - Unweighted), those cell types that fell into the test conditions (Test - Unweighted), and those that did not fall into the test conditions (Train - Unweighted). It is important to keep in mind since this isn't a condition (stimulation x cell type combination), the embeddings categorized as test were seen and learned during training. Finally, we get the embedding outputs from running the forward prediction. This is simply taking each learned embedding, and repeating it as the same number of times as there are test cells for a given condition. It is essentially the same as Test - Unweighted, but now it is weighted by the number of cells predicted in each test condition cell type.

# In[157]:


if visualize:
    fig, axes = plt.subplots(ncols = 5, nrows = 1, figsize = (25, 5))
    ax = axes.flatten()

    np.random.seed(seed_split)

    i = 0
    # weights = mod.signaling_network.weights.detach().cpu().numpy().flatten()
    # weights = weights[weights != 0]
    weights = mod.signaling_network.weights[~mod.signaling_network.mask].detach().cpu().numpy()
    mean, std = np.mean(weights), np.std(weights)
    print('Weights: N({:.4f}, {:.4f})'.format(mean, std))
    if 'signaling_weights_scaler' in mod.signaling_network.bionet_params:
        weights *= mod.signaling_network.bionet_params['signaling_weights_scaler']
    weights = np.random.choice(weights, 5000, replace = False)

    counts, bins = np.histogram(weights, bins=30, density=True)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    # hist = ax[i].bar(bin_centers, counts, width=(bins[1] - bins[0]), 
    #           color=sns.color_palette()[0], edgecolor = 'black', label="Sample Histogram")
    sns.kdeplot(weights, ax = ax[i], 
                color = 'black', linestyle = '-', 
                label = 'Sample KDE')
    ax[i].set_xlabel('Trained Bionet Weights')

    x = np.linspace(-4, 4, 100)
    _, = ax[i].plot(x, stats.norm.pdf(x), color = 'green', linestyle = '--', label="Standard Normal")

    x = np.linspace(mean - 4*std, mean + 4*std, 100)
    _, = ax[i].plot(x, stats.norm.pdf(x, loc=mean, scale=std), color = 'red', 
                      linestyle = '--', label='Sample Normal Distribution')

    # ax[i].legend()
    # ax[i].legend(handles=[hist, sn], labels=["Sample Distribution", "Standard Normal"], 
    #             loc='upper left', bbox_to_anchor=(-0.7, 1))

    i = 1
    bias = bias_tot.detach().cpu().numpy().flatten()
    mean, std = np.mean(bias), np.std(bias)
    print('Total Bias: N({:.4f}, {:.4f})'.format(mean, std))
    bias = np.random.choice(bias, 5000, replace = False)

    # counts, bins = np.histogram(bias, bins=30, density=True)
    # bin_centers = 0.5 * (bins[1:] + bins[:-1])
    # hist = ax[i].bar(bin_centers, counts, width=(bins[1] - bins[0]), 
    #           color=sns.color_palette()[0], edgecolor = 'black', label="Sample Histogram")
    # sns.kdeplot(bias, ax = ax[i], color = 'gray', linestyle = '--', label = 'Sample KDE', zorder = 0)
    sns.kdeplot(bias, ax = ax[i], 
                color = 'black', linestyle = '-', 
                label = 'Sample KDE')
    ax[i].set_xlabel('Trained Total Bias')

    x = np.linspace(-4, 4, 100)
    _, = ax[i].plot(x, stats.norm.pdf(x), color = 'green', linestyle = '--', label="Standard Normal")

    x = np.linspace(mean - 4*std, mean + 4*std, 100)
    _, = ax[i].plot(x, stats.norm.pdf(x, loc=mean, scale=std), color = 'red', 
                      linestyle = '--', label='Sample Normal Distribution')

    ax[i].set_xlabel('Trained Total Bias')
    ax[i].legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2)


    i = 2
    bias = bias_global.detach().cpu().numpy().flatten()
    mean, std = np.mean(bias), np.std(bias)
    print('Global Bias: N({:.4f}, {:.4f})'.format(mean, std))
    bias = np.random.choice(bias, 5000, replace = False)

    # counts, bins = np.histogram(bias, bins=30, density=True)
    # bin_centers = 0.5 * (bins[1:] + bins[:-1])
    # hist = ax[i].bar(bin_centers, counts, width=(bins[1] - bins[0]), 
    #           color=sns.color_palette()[0], edgecolor = 'black', label="Sample Histogram")
    # sns.kdeplot(bias, ax = ax[i], color = 'gray', linestyle = '--', label = 'Sample KDE', zorder = 0)
    sns.kdeplot(bias, ax = ax[i], 
                color = 'black', linestyle = '-', 
                label = 'Sample KDE')
    ax[i].set_xlabel('Trained Bionet bias')

    x = np.linspace(-4, 4, 100)
    _, = ax[i].plot(x, stats.norm.pdf(x), color = 'green', linestyle = '--', label="Standard Normal")

    x = np.linspace(mean - 4*std, mean + 4*std, 100)
    _, = ax[i].plot(x, stats.norm.pdf(x, loc=mean, scale=std), color = 'red', 
                      linestyle = '--', label='Sample Normal Distribution')

    ax[i].set_xlabel('Trained Global Bias')
    # ax[i].legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    i = 3
    bias = bias_cats.detach().cpu().numpy().flatten()
    mean, std = np.mean(bias), np.std(bias)
    bias = np.random.choice(bias, 5000, replace = False)

    # counts, bins = np.histogram(bias, bins=30, density=True)
    # bin_centers = 0.5 * (bins[1:] + bins[:-1])
    # hist = ax[i].bar(bin_centers, counts, width=(bins[1] - bins[0]), 
    #           color=sns.color_palette()[0], edgecolor = 'black', label="Sample Histogram")
    # sns.kdeplot(bias, ax = ax[i], color = 'gray', linestyle = '--', label = 'Sample KDE', zorder = 0)
    sns.kdeplot(bias, ax = ax[i], 
                color = 'black', linestyle = '-', 
                label = 'Sample KDE')
    ax[i].set_xlabel('Trained Categorical Bias (Predicted - Weighted Test)')

    x = np.linspace(-4, 4, 100)
    _, = ax[i].plot(x, stats.norm.pdf(x), color = 'green', linestyle = '--', label="Standard Normal")

    x = np.linspace(mean - 4*std, mean + 4*std, 100)
    _, = ax[i].plot(x, stats.norm.pdf(x, loc=mean, scale=std), color = 'red', 
                      linestyle = '--', label='Sample Normal Distribution')

    ax[i].set_xlabel('Trained Categorical Bias')

    i = 4
    bias = bias_cats.detach().cpu().numpy().flatten()
    bias_cat_embeddings = mod.signaling_network.cat_embeddings.seurat_annotations.weight.detach().cpu().numpy()

    test_ct_idx = [mod.signaling_network.cat_mapper['seurat_annotations'][ct.split('^')[1]] for ct in test_conds]
    bias_cat_test = bias_cat_embeddings[test_ct_idx,:].flatten()

    train_ct_idx = sorted(set(mod.signaling_network.cat_mapper['seurat_annotations'].values()).difference(test_ct_idx))
    bias_cat_train = bias_cat_embeddings[train_ct_idx,:].flatten()

    bias_cat_embeddings = bias_cat_embeddings.flatten()

    cmd = 'Categorical Bias: Predicted (Weighted Test) | Test (Unweighted) | Train (Unweighted) | Everything (Unweighted): '
    cmd += 'N({:.4f}, {:.4f}) | '.format(np.mean(bias), np.std(bias))
    cmd += 'N({:.4f}, {:.4f}) | '.format(np.mean(bias_cat_test), np.std(bias_cat_test))
    cmd += 'N({:.4f}, {:.4f}) | '.format(np.mean(bias_cat_train), np.std(bias_cat_train))
    cmd += 'N({:.4f}, {:.4f})'.format(np.mean(bias_cat_embeddings), np.std(bias_cat_embeddings))
    print(cmd)

    sns.kdeplot(np.random.choice(bias, 5000, replace = False), 
                ax = ax[i], color = 'blue', label = 'Predicted (Weighted Test)')
    sns.kdeplot(bias_cat_test, 
                ax = ax[i], color = 'red', label = 'Test (Unweighted)')
    sns.kdeplot(bias_cat_train, 
                ax = ax[i], color = 'green', label = 'Train (Unweighted)')
    sns.kdeplot(bias_cat_embeddings, 
                ax = ax[i], color = 'black', label = 'Everything (Unweighted)')
    ax[i].legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2)

    ax[i].set_xlabel('Trained Categorical Biases')

    fig.tight_layout()
    # plt.savefig(os.path.join(data_path, 'trash', fn + '_bias_dist' + '.png'), dpi=300, bbox_inches='tight')
    ("")


# Run the embedding:
# 
# We will visualize both the UMAP and PCA space to see whether linear and non-linear mixing is being captured.

# In[173]:


# if run_type == 'E': # nothing to do with just one test condition - but will use in train + test
if bias_global.shape[0] < 50:
    n_components = 10
else:
    n_components = 50

biases_clustered = {}
for counterfactual_type, br in biases_res.items():
    print(counterfactual_type)
    bias_global, _, _, bias_cats, bias_tot = br['adverserial']
    obs = br['obs']

    # full model
    bias_global = sc.AnnData(X = bias_global.detach().cpu().numpy(), obs = obs)
    embed_tf_activity(bias_global, scanpy_pca = False, cluster_col_name = 'leiden', resolution = 1,
                     n_components = n_components)

    # full model -- categorical information added
    bias_tot = sc.AnnData(X = bias_tot.detach().cpu().numpy().T, obs = obs)
    embed_tf_activity(bias_tot, scanpy_pca = False, cluster_col_name = 'leiden', resolution = 1,
                     n_components = n_components)

    # full model -- categorical information only
    bias_cats = sc.AnnData(X = bias_cats.detach().cpu().numpy().T, obs = obs)
    embed_tf_activity(bias_cats, scanpy_pca = False, cluster_col_name = 'leiden', resolution = 1, 
                     n_components = n_components)

    biases_clustered[counterfactual_type] = (bias_global, bias_tot, bias_cats)

io.write_pickled_object(biases_clustered,  
                        os.path.join(data_path, 'trash', fn + '_feb_clustered_biases.pickle'))


# In[2]:


if visualize and run_type == 'E':
    biases_clustered = io.read_pickled_object(os.path.join(data_path, 'trash', fn + '_feb_clustered_biases.pickle'))

    bias_global, bias_tot, bias_cats = biases_clustered[counterfactual_type]
    bias_types = {'Global': bias_global, 
                 'Categorical': bias_cats, 
                 'Total': bias_tot}

    cat_map = {'seurat_annotations': 'Cell Type', 
              'stim': 'Stimulation'}
    
    ncols = len(bias_types)
    nrows = len(cat_map)
    fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize = (ncols *5.1, nrows*5.1))

    subset_size = None
    for j, (bias_type, bias) in enumerate(bias_types.items()):
        for i, cat in enumerate(['seurat_annotations', 'stim']):
            viz_df, nmi = adata_dimviz_bias(adata = bias, 
                                            reduction_type = 'umap', 
                                            cat = cat,
                                            subset_size = subset_size)


            sns.scatterplot(data = viz_df, x = 'UMAP1', y = 'UMAP2', hue = cat, 
                            s=10,
                            ax = ax[i,j])
            ax[i,j].annotate('NMI (Leiden Clusters, ' + cat_map[cat] + '): {:.4f}'.format(nmi),
                            xy = (0.325, 0.95), xycoords='axes fraction', fontsize = 9)
            ax[i,j].set_title(bias_type + ' Bias')

        


# In[ ]:


if visualize and run_type == 'E':
    
    biases_clustered = io.read_pickled_object(os.path.join(data_path, 
                                                           'trash', fn + '_feb_clustered_biases.pickle'))
    bias_global, bias_tot, bias_cats = biases_clustered[counterfactual_type]
    bias_types = {'Global': bias_global, 
                 'Categorical': bias_cats, 
                 'Total': bias_tot}

    cat_map = {'seurat_annotations': 'Cell Type', 
              'stim': 'Stimulation'}
    

    ncols = len(bias_types)
    nrows = len(cat_map)
    fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize = (ncols *5.1, nrows*5.1))

    subset_size = None
    for j, (bias_type, bias) in enumerate(bias_types.items()):
        for i, cat in enumerate(['seurat_annotations', 'stim']):
            viz_df, nmi = adata_dimviz_bias(adata = bias, 
                                            reduction_type = 'pca', 
                                            cat = cat,
                                            subset_size = subset_size)


            sns.scatterplot(data = viz_df, x = 'PC1', y = 'PC2', hue = cat, 
                            s=10,
                            ax = ax[i,j])
            ax[i,j].annotate('NMI (Leiden Clusters, ' + cat_map[cat] + '): {:.4f}'.format(nmi),
                            xy = (0.325, 0.95), xycoords='axes fraction', fontsize = 9)
            ax[i,j].set_title(bias_type + ' Bias')

        


# ## 5.2: Train Biases
# 
# This is visualized to see that train matchest test. Furthermore, in LOO test conditions, the visualizations aren't informative because there are no conditions to check for mixing. 
# 
# <span style="color:red">Note, since the test is using a counterfactual and the train is not, the test output should simply be a subset of the train for the generator input.</span>

# In[ ]:


for counterfactual in [False]:#True, False]:
    fn_mode = '' if not counterfactual else '_counterfactual'
    
    # get biases
    res = get_prediction(mod = mod,
                                      train_cells = train_cells,
                                      tf_adata = tf_adata,
                                      train_mode = True,
                                      counterfactual = counterfactual,
                                      remove_type = 'none',
                                      return_bias = True, 
                           max_cells = 5000)
    bias_global_train, _, _, bias_cats_train, bias_tot_train, obs_train = res

    # format
    counterfactual_type = 'opposite'
    biases_res_train = {}
    biases_res_train[counterfactual_type] = {}
    biases_res_train[counterfactual_type]['adverserial'] = (bias_global_train, bias_cats_train, bias_tot_train)
    biases_res_train[counterfactual_type]['obs'] = obs_train


    # run embedding
    if bias_global_train.shape[0] < 50:
        n_components = 10
    else:
        n_components = 50

    biases_clustered = {}

    for counterfactual_type, br in biases_res_train.items():
        bias_global, bias_cats, bias_tot = br['adverserial']
        obs = br['obs']

        # full model
        bias_global = sc.AnnData(X = bias_global.detach().cpu().numpy(), obs = obs)
        embed_tf_activity(bias_global, scanpy_pca = False, cluster_col_name = 'leiden', resolution = 1,
                         n_components = n_components)

        # full model -- categorical information added
        bias_tot = sc.AnnData(X = bias_tot.detach().cpu().numpy().T, obs = obs)
        embed_tf_activity(bias_tot, scanpy_pca = False, cluster_col_name = 'leiden', resolution = 1,
                         n_components = n_components)

        # full model -- categorical information only
        bias_cats = sc.AnnData(X = bias_cats.detach().cpu().numpy().T, obs = obs)
        embed_tf_activity(bias_cats, scanpy_pca = False, cluster_col_name = 'leiden', resolution = 1, 
                         n_components = n_components)

        biases_clustered[counterfactual_type] = (bias_global, bias_tot, bias_cats)

    io.write_pickled_object(biases_clustered,  
                            os.path.join(data_path, 'trash', fn + '_feb_clustered_biases_train' + fn_mode + '.pickle'))


# ### Biases Train - No Counterfactual

# In[ ]:


bias_train_pred = os.path.isfile(os.path.join(data_path, 'trash', fn + '_feb_clustered_biases_train.pickle'))
if visualize and bias_train_pred:

    if not bias_train_pred:
        print('We do not have training predictions for bias')
    else: 
        biases_clustered_train = io.read_pickled_object(os.path.join(data_path, 'trash', fn + '_feb_clustered_biases_train.pickle'))


    
    bias_global, bias_tot, bias_cats = biases_clustered_train[counterfactual_type]
    bias_types = {'Global': bias_global, 
                 'Categorical': bias_cats, 
                 'Total': bias_tot}

    cat_map = {'seurat_annotations': 'Cell Type', 
              'stim': 'Stimulation'}

    ncols = len(bias_types)
    nrows = len(cat_map)
    fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize = (ncols *5.1, nrows*5.1))

    subset_size = 5000 if bias_global.shape[0] > 5000 else None
    for j, (bias_type, bias) in enumerate(bias_types.items()):
        for i, cat in enumerate(['seurat_annotations', 'stim']):
            viz_df, nmi = adata_dimviz_bias(adata = bias, 
                                            reduction_type = 'umap', 
                                            cat = cat,
                                            subset_size = subset_size)


            sns.scatterplot(data = viz_df, x = 'UMAP1', y = 'UMAP2', hue = cat, 
                            s=10,
                            ax = ax[i,j])
            ax[i,j].annotate('NMI (Leiden Clusters, ' + cat_map[cat] + '): {:.4f}'.format(nmi),
                            xy = (0.325, 0.95), xycoords='axes fraction', fontsize = 9)
            ax[i,j].set_title(bias_type + ' Bias')

            if cat == 'seurat_annotations':
                ax[i,j].legend().set_visible(False)

    fig.suptitle('Train Biases - No Counterfactual')
    fig.tight_layout();


    bias_global, bias_tot, bias_cats = biases_clustered_train[counterfactual_type]
    bias_types = {'Global': bias_global, 
                 'Categorical': bias_cats, 
                 'Total': bias_tot}

    cat_map = {'seurat_annotations': 'Cell Type', 
              'stim': 'Stimulation'}

    ncols = len(bias_types)
    nrows = len(cat_map)
    fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize = (ncols *5.1, nrows*5.1))

    subset_size = 5000 if bias_global.shape[0] > 5000 else None
    for j, (bias_type, bias) in enumerate(bias_types.items()):
        for i, cat in enumerate(['seurat_annotations', 'stim']):
            viz_df, nmi = adata_dimviz_bias(adata = bias, 
                                            reduction_type = 'pca', 
                                            cat = cat,
                                            subset_size = subset_size)


            sns.scatterplot(data = viz_df, x = 'PC1', y = 'PC2', hue = cat, 
                            s=10,
                            ax = ax[i,j])
            ax[i,j].annotate('NMI (Leiden Clusters, ' + cat_map[cat] + '): {:.4f}'.format(nmi),
                            xy = (0.325, 0.95), xycoords='axes fraction', fontsize = 9)
            ax[i,j].set_title(bias_type + ' Bias')

            if cat == 'seurat_annotations':
                ax[i,j].legend().set_visible(False)
    fig.suptitle('Train Biases - No Counterfactual')
    fig.tight_layout();


# ### Train Biases with Counterfactual

# In[ ]:


# bias_train_pred = os.path.isfile(os.path.join(data_path, 'trash', fn + '_feb_clustered_biases_train_counterfactual.pickle'))
# if visualize and bias_train_pred:

#     if not bias_train_pred:
#         print('We do not have training predictions for bias')
#     else: 
#         biases_clustered_train = io.read_pickled_object(os.path.join(data_path, 'trash', fn + '_feb_clustered_biases_train_counterfactual.pickle'))


    
#     bias_global, bias_tot, bias_cats = biases_clustered_train[counterfactual_type]
#     bias_types = {'Global': bias_global, 
#                  'Categorical': bias_cats, 
#                  'Total': bias_tot}

#     cat_map = {'seurat_annotations': 'Cell Type', 
#               'stim': 'Stimulation'}

#     ncols = len(bias_types)
#     nrows = len(cat_map)
#     fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize = (ncols *5.1, nrows*5.1))

#     subset_size = 5000 if bias_global.shape[0] > 5000 else None
#     for j, (bias_type, bias) in enumerate(bias_types.items()):
#         for i, cat in enumerate(['seurat_annotations', 'stim']):
#             viz_df, nmi = adata_dimviz_bias(adata = bias, 
#                                             reduction_type = 'umap', 
#                                             cat = cat,
#                                             subset_size = subset_size)


#             sns.scatterplot(data = viz_df, x = 'UMAP1', y = 'UMAP2', hue = cat, 
#                             s=10,
#                             ax = ax[i,j])
#             ax[i,j].annotate('NMI (Leiden Clusters, ' + cat_map[cat] + '): {:.4f}'.format(nmi),
#                             xy = (0.325, 0.95), xycoords='axes fraction', fontsize = 9)
#             ax[i,j].set_title(bias_type + ' Bias')

#             if cat == 'seurat_annotations':
#                 ax[i,j].legend().set_visible(False)

#     fig.suptitle('Train Biases - With Counterfactual')
#     fig.tight_layout();


#     bias_global, bias_tot, bias_cats = biases_clustered_train[counterfactual_type]
#     bias_types = {'Global': bias_global, 
#                  'Categorical': bias_cats, 
#                  'Total': bias_tot}

#     cat_map = {'seurat_annotations': 'Cell Type', 
#               'stim': 'Stimulation'}

#     ncols = len(bias_types)
#     nrows = len(cat_map)
#     fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize = (ncols *5.1, nrows*5.1))

#     subset_size = 5000 if bias_global.shape[0] > 5000 else None
#     for j, (bias_type, bias) in enumerate(bias_types.items()):
#         for i, cat in enumerate(['seurat_annotations', 'stim']):
#             viz_df, nmi = adata_dimviz_bias(adata = bias, 
#                                             reduction_type = 'pca', 
#                                             cat = cat,
#                                             subset_size = subset_size)


#             sns.scatterplot(data = viz_df, x = 'PC1', y = 'PC2', hue = cat, 
#                             s=10,
#                             ax = ax[i,j])
#             ax[i,j].annotate('NMI (Leiden Clusters, ' + cat_map[cat] + '): {:.4f}'.format(nmi),
#                             xy = (0.325, 0.95), xycoords='axes fraction', fontsize = 9)
#             ax[i,j].set_title(bias_type + ' Bias')

#             if cat == 'seurat_annotations':
#                 ax[i,j].legend().set_visible(False)
#     fig.suptitle('Train Biases - With Counterfactual')
#     fig.tight_layout();


# ## 5.3 Train+ Test Biases
# 
# 
# This simply enables visualization of all stimulation conditions together. 
# 
# ### Train (no counterfactual)

# In[ ]:


bias_train_pred = os.path.isfile(os.path.join(data_path, 'trash', fn + '_feb_clustered_biases_train.pickle'))
biases_clustered_test = io.read_pickled_object(os.path.join(data_path, 'trash', fn + '_feb_clustered_biases.pickle'))
if bias_train_pred:
    biases_clustered_train = io.read_pickled_object(os.path.join(data_path, 'trash', fn + '_feb_clustered_biases_train.pickle'))

    
    combined_biases = []
    for i in trange(3):

        adata1 = biases_clustered_test['opposite'][i].copy()
        adata2 = biases_clustered_train['opposite'][i].copy()

        adata1.obs['split'] = 'test'
        adata2.obs['split'] = 'train'
        adata1.obs.drop(columns=['leiden'], inplace = True)
        adata2.obs.drop(columns=['leiden'], inplace = True)
        
        X_combined = np.vstack([adata1.X, adata2.X])
        obs_combined = pd.concat([adata1.obs, adata2.obs], axis=0)

        adata_combined = sc.AnnData(
            X = np.vstack([adata1.X, adata2.X]),
            obs = pd.concat([adata1.obs, adata2.obs], axis=0),
            var = adata1.var.copy()
        )
        adata_combined.obs_names_make_unique()
        
        if min(adata_combined.n_obs, adata_combined.n_vars) < 50:
            n_components = 10
        else:
            n_components = 50
        
        embed_tf_activity(adata_combined, 
                          scanpy_pca = False, 
                          cluster_col_name = 'leiden', 
                          resolution = 1,
                          n_components = n_components)
        
        combined_biases.append(adata_combined)

        
    biases_clustered_both = {'opposite': tuple(combined_biases)}
    
    io.write_pickled_object(biases_clustered_both,  
                            os.path.join(data_path, 'trash', fn + '_feb_clustered_biases_both.pickle'))
    


# In[ ]:


if visualize and bias_both_pred:
    bias_both_pred = os.path.isfile(os.path.join(data_path, 'trash', fn + '_feb_clustered_biases_both.pickle'))

    biases_clustered_both = io.read_pickled_object(os.path.join(data_path, 'trash', fn + '_feb_clustered_biases_both.pickle'))
    
    
    bias_global, bias_tot, bias_cats = biases_clustered_both[counterfactual_type]
    bias_types = {'Global': bias_global, 
                 'Categorical': bias_cats, 
                 'Total': bias_tot}

    cat_map = {'seurat_annotations': 'Cell Type', 
              'stim': 'Stimulation', 
              'split': 'Data Split'}

    ncols = len(bias_types)
    nrows = len(cat_map)
    fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize = (ncols *5.1, nrows*5.1))

    subset_size = 5000 if bias_global.shape[0] > 5000 else None
    for j, (bias_type, bias) in enumerate(bias_types.items()):
        for i, cat in enumerate(['seurat_annotations', 'stim', 'split']):
            viz_df, nmi = adata_dimviz_bias(adata = bias, 
                                            reduction_type = 'umap', 
                                            cat = cat,
                                            subset_size = subset_size)


            sns.scatterplot(data = viz_df, x = 'UMAP1', y = 'UMAP2', hue = cat, 
                            s=10,
                            ax = ax[i,j])
            ax[i,j].annotate('NMI (Leiden Clusters, ' + cat_map[cat] + '): {:.4f}'.format(nmi),
                            xy = (0.325, 0.95), xycoords='axes fraction', fontsize = 9)
            ax[i,j].set_title(bias_type + ' Bias')
            
            if j != 0:
                ax[i,j].legend_.remove()

            if cat == 'seurat_annotations':
                ax[i,j].legend().set_visible(False)
    fig.suptitle('Test + Train Biases - No Counterfactual')
    fig.tight_layout()
    ("")
    


# In[ ]:


if visualize and bias_both_pred:
    bias_both_pred = os.path.isfile(os.path.join(data_path, 'trash', fn + '_feb_clustered_biases_both.pickle'))

    biases_clustered_both = io.read_pickled_object(os.path.join(data_path, 'trash', fn + '_feb_clustered_biases_both.pickle'))
    
    
    bias_global, bias_tot, bias_cats = biases_clustered_both[counterfactual_type]
    bias_types = {'Global': bias_global, 
                 'Categorical': bias_cats, 
                 'Total': bias_tot}

    cat_map = {'seurat_annotations': 'Cell Type', 
              'stim': 'Stimulation', 
              'split': 'Data Split'}

    ncols = len(bias_types)
    nrows = len(cat_map)
    fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize = (ncols *5.1, nrows*5.1))

    subset_size = 5000 if bias_global.shape[0] > 5000 else None
    for j, (bias_type, bias) in enumerate(bias_types.items()):
        for i, cat in enumerate(['seurat_annotations', 'stim', 'split']):
            viz_df, nmi = adata_dimviz_bias(adata = bias, 
                                            reduction_type = 'pca', 
                                            cat = cat,
                                            subset_size = subset_size)


            sns.scatterplot(data = viz_df, x = 'PC1', y = 'PC2', hue = cat, 
                            s=10,
                            ax = ax[i,j])
            ax[i,j].annotate('NMI (Leiden Clusters, ' + cat_map[cat] + '): {:.4f}'.format(nmi),
                            xy = (0.325, 0.95), xycoords='axes fraction', fontsize = 9)
            ax[i,j].set_title(bias_type + ' Bias')
            
            if j != 0:
                ax[i,j].legend_.remove()

            if cat == 'seurat_annotations':
                ax[i,j].legend().set_visible(False)
    fig.suptitle('Test + Train Biases - No Counterfactual')
    fig.tight_layout()
    ("")


# # 6. Get the predictions
# 
# We get both the test and train predictions. Test allows to assess for generalization. Train allows to assess for any structural probelms in the model. 
# 
# We assess train both as in the training -- not on the counterfactual, and as would be done for test -- on the counterfactual

# In[175]:


best_resolution = tf_adata.uns['leiden']['params']['resolution']
calculation_type = 'project' # project data rather than embed
n_neighbors = 15
run_umap = True


# In[176]:


cf_map = {'in_distribution': train_cells}
counterfactual_types = list(cf_map.keys()) + ['opposite']

counterfactual_type = 'opposite'
remove_components = ['none', 
                         ['adj', 'categorical_bias'], 
                         ['adj', 'global_bias'],
                         'total_bias', 'adj',
                         'categorical_bias',
                         'global_bias']


# In[ ]:


# gets test predictions with counterfactual
# gets train predictions with and without counterfactual

for train_mode, counterfactual in tqdm(list(itertools.product([True, False], repeat=2))):
    if not train_mode and not counterfactual:
        continue
    fn_mode = ''
    if train_mode:
        fn_mode += '_train'
        if counterfactual:
            fn_mode += '_counterfactual'    
    
    tf_res = {}
    loss_res = []
    for remove_type in remove_components:
        # get prediction
        tf_adata_predicted = get_prediction(mod = mod,
                                      train_cells = train_cells,
                                      tf_adata = tf_adata,
                                      train_mode = train_mode,
                                      counterfactual = counterfactual,
                                      remove_type = remove_type,
                                      return_bias = False, 
                                           max_cells = 2000)
        if type(remove_type) == list:
            remove_type = '_'.join(remove_type)

        # project prediction
        run_umap = True if remove_type == 'none' else False # for visualization
        tf_adata_predicted = prepare_for_metrics(tf_adata, 
                                                 tf_adata_predicted, 
                                                 resolution = best_resolution,
                                                 calculation_type = calculation_type, 
                                                 n_neighbors = n_neighbors, 
                                                 run_umap = run_umap
                                                )
        tf_res[remove_type] = tf_adata_predicted
        
        # calculate prediction loss
        loss = get_loss(tf_adata, tf_adata_predicted[tf_adata_predicted.obs.batch == 'predicted', ])
        loss = pd.DataFrame(loss, index = [0])
        loss['Removed Model Component'] = remove_type
        loss_res.append(loss)

    loss_res = pd.concat(loss_res, axis = 0).reset_index(drop = True)
    loss_res.to_csv(os.path.join(data_path, 'trash', fn + '_predictions_loss' + fn_mode + '.csv'))

    io.write_pickled_object(tf_res, 
                           os.path.join(data_path, 'trash', fn + '_predictions' + fn_mode + '.pickle'))

    clear_memory()


# In[40]:


train_pred = os.path.isfile(os.path.join(data_path, 'trash', fn + '_predictions_train.pickle'))
    
if train_pred:
    tf_res_train = io.read_pickled_object(os.path.join(data_path, 'trash', fn + '_predictions_train.pickle'))
    loss_res_train = pd.read_csv(os.path.join(data_path, 'trash', fn + '_predictions_loss_train.csv'), 
                      index_col = 0)
else:
    print('We do not have train predictions')
    tf_res_train, loss_res_train = None, None

tf_res_train_ct = io.read_pickled_object(os.path.join(data_path, 'trash', fn + '_predictions_train_counterfactual.pickle'))    
loss_res_train_ct = pd.read_csv(os.path.join(data_path, 'trash', fn + '_predictions_loss_train_counterfactual.csv'), 
                      index_col = 0)     
    
tf_res = io.read_pickled_object(os.path.join(data_path, 'trash', fn + '_predictions.pickle'))
loss_res = pd.read_csv(os.path.join(data_path, 'trash', fn + '_predictions_loss.csv'), 
                      index_col = 0)

loss_res_map = {'train - no counterfactual': loss_res_train, 
                'train - counterfactual': loss_res_train_ct, 
                'test': loss_res}


# # 7. Visualize Predictions

# # 7.1 Visualize loss of each component:

# ## Total Loss

# In[178]:


if visualize:
    ncols = len(loss_res_map)
    fig, ax = plt.subplots(ncols = ncols, figsize = (ncols*5.1, 5))


    xtick_map = {'none': 'Full Model', 
     'adj': 'Adjacency Matrix', 
     'total_bias': 'Total Bias', 
    'categorical_bias': 'Categorical Bias', 
    'global_bias': 'Global Bias', 
                'adj_categorical_bias': 'Adjacency Matrix and \n Categorical Bias', 
                'adj_global_bias': 'Adjacency Matrix and \n Global Bias'}


    for i, (prediction_type, loss_res) in enumerate(loss_res_map.items()):
        if loss_res is not None:

            viz_df = loss_res.copy()
            viz_df['Removed Model Component'] = viz_df['Removed Model Component'].map(xtick_map)

            sns.scatterplot(data = viz_df, x = 'Removed Model Component', y = 'EMD Loss', ax = ax[i])

            for x, y in zip(viz_df['Removed Model Component'], viz_df['EMD Loss']):
                ax[i].vlines(x, ymin=0, ymax=y, linestyle='dashed', color='gray', zorder = 0)
            ax[i].axhline(y=0, color='black')
            ax[i].axhline(y=viz_df.set_index('Removed Model Component').loc['Full Model']['EMD Loss'], 
                       color='red', linestyle = '--')

            ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=30, ha='right')

        ax[i].set_title(prediction_type)

    fig.tight_layout()
    ("")


# ## Individual conditions

# In[ ]:


if visualize:
    prediction_type = 'test'
    loss_res = loss_res_map[prediction_type]
    
    conds = list(set(loss_res.columns).difference(['Removed Model Component', 'EMD Loss']))
    conds_ = pd.Series(conds).str.split('^', expand=True)
    conds_['combined'] = conds
    conds = conds_.sort_values(by=[1,0]).combined.tolist()
    
    ncols = min(len(conds), 7)
    nrows = math.ceil(len(conds)/ncols)

    if len(conds) == 1:
        print('Test individual condition is same as above full test for LOO')
    else:
        fig, axes = plt.subplots(ncols = ncols, nrows = nrows, figsize = (ncols*5.1, nrows*5))
        if isinstance(axes, np.ndarray):
            ax = axes.flatten()
        else:
            ax = [axes]

        xtick_map = {'none': 'Full Model', 
         'adj': 'Adjacency Matrix', 
         'total_bias': 'Total Bias', 
        'categorical_bias': 'Categorical Bias', 
        'global_bias': 'Global Bias', 
                    'adj_categorical_bias': 'Adjacency Matrix and \n Categorical Bias', 
                    'adj_global_bias': 'Adjacency Matrix and \n Global Bias'}

        for i, cond in enumerate(conds):
            viz_df = loss_res.copy()
            viz_df['Removed Model Component'] = viz_df['Removed Model Component'].map(xtick_map)

            sns.scatterplot(data = viz_df, x = 'Removed Model Component', y = cond, ax = ax[i])

            for x, y in zip(viz_df['Removed Model Component'], viz_df[cond]):
                ax[i].vlines(x, ymin=0, ymax=y, linestyle='dashed', color='gray', zorder = 0)
            ax[i].axhline(y=0, color='black')
            ax[i].axhline(y=viz_df.set_index('Removed Model Component').loc['Full Model'][cond], 
                       color='red', linestyle = '--')

            ax[i].set_xticks(ax[i].get_xticks())  
            ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=30, ha='right')
            ax[i].set_title(cond)

        fig.suptitle(prediction_type)
        fig.tight_layout()
        ("")


# In[ ]:


# if visualize and train_pred:
#     prediction_type = 'train'
#     loss_res = loss_res_map[prediction_type]

#     conds = list(set(loss_res.columns).difference(['Removed Model Component', 'EMD Loss']))
#     conds_ = pd.Series(conds).str.split('^', expand=True)
#     conds_['combined'] = conds
#     conds = conds_.sort_values(by=[1,0]).combined.tolist()
    
#     ncols = min(len(conds), 7)
#     nrows = math.ceil(len(conds)/ncols)

#     fig, axes = plt.subplots(ncols = ncols, nrows = nrows, figsize = (ncols*5.1, nrows*5))
#     if isinstance(axes, np.ndarray):
#         ax = axes.flatten()
#     else:
#         ax = [axes]

#     xtick_map = {'none': 'Full Model', 
#      'adj': 'Adjacency Matrix', 
#      'total_bias': 'Total Bias', 
#     'categorical_bias': 'Categorical Bias', 
#     'global_bias': 'Global Bias', 
#                 'adj_categorical_bias': 'Adjacency Matrix and \n Categorical Bias', 
#                 'adj_global_bias': 'Adjacency Matrix and \n Global Bias'}

#     for i, cond in enumerate(conds):
#         viz_df = loss_res.copy()
#         viz_df['Removed Model Component'] = viz_df['Removed Model Component'].map(xtick_map)

#         sns.scatterplot(data = viz_df, x = 'Removed Model Component', y = cond, ax = ax[i])

#         for x, y in zip(viz_df['Removed Model Component'], viz_df[cond]):
#             ax[i].vlines(x, ymin=0, ymax=y, linestyle='dashed', color='gray', zorder = 0)
#         ax[i].axhline(y=0, color='black')
#         ax[i].axhline(y=viz_df.set_index('Removed Model Component').loc['Full Model'][cond], 
#                    color='red', linestyle = '--')

#         ax[i].set_xticks(ax[i].get_xticks())  
#         ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=30, ha='right')
#         ax[i].set_title(cond)

#     fig.suptitle(prediction_type)
#     fig.tight_layout()
#     ;    


# ## 7.2 Visualize first 2 PCs of each model component
# 
# The title indicates which component was removed (e.g., if "total_bias" is shown, that was removed, and predictions were run only using the adjacency matrix).
# 
# ### Full TF PC space of actual data
# 
# This visualization provides a reference for the global distribution of cell types/stimulation when looking at teh specific ones below. We can see that stimulation provides strong separation. 

# In[ ]:


if visualize: 
    fig, ax = plt.subplots(ncols = 2, figsize = (10,5))

    tf_adata_viz = io.read_tfad(os.path.join(data_path, 'processed', 'Kang_tf_activity.h5ad'))

    viz_df = adata_dimviz_prediction(adata = tf_adata_viz, reduction_type = 'pca',
                                     cats = ['seurat_annotations',  'stim'],
                                     max_condition_size = None if counterfactual_type == 'opposite' else 3000,
                                     seed = seed_split)
    viz_df.stim = pd.Categorical(viz_df.stim, categories = ['CTRL', 'STIM'], ordered = True)

    marker_dict = {'CTRL': 'o', 'STIM': '^'}  

    i = 0
    sns.scatterplot(data = viz_df,
                    x = 'PC1', y = 'PC2', hue = 'seurat_annotations',
                    s = 10, 
    #                 style = 'stim', markers = marker_dict, s = 20,
                    ax = ax[i], legend = False)
    ax[i].set_title('Cell Type')

    i = 1
    sns.scatterplot(data = viz_df,
                    x = 'PC1', y = 'PC2', hue = 'stim',
                    s = 10, 
    #                 style = 'stim', markers = marker_dict, s = 20,
                    ax = ax[i])
    ax[i].set_title('Stimulation Condition')

    fig.tight_layout()
    ("")


# ### Test

# In[179]:


tf_res_ = io.read_pickled_object(os.path.join(data_path, 'trash', fn + '_predictions.pickle'))


# In[184]:


remove_components = ['none', 
                     ['adj', 'global_bias'],
                     'global_bias',
                     'total_bias',
                     ['adj', 'categorical_bias'],
                     'adj',
                     'categorical_bias']
# map to more inuitive title
remaining_components = ['Full Model', 'Categorical Bias Only', 
                        'Adj + Categorical Bias',
                       'Adjacency Matrix Only',  
                        'Global Bias Only', 'Total Bias', 'Adj + Global Bias', 
                       ]

cell_types = test_cell_types
nrows = len(test_conds)
ncols = len(remove_components)

counterfactual_type = 'opposite'


# In[ ]:


remove_component = 'total_bias'
tf_res = tf_res_.copy()
tf_adata_all = tf_res[remove_component]
md = tf_adata_all.obs

md = md[md.batch == 'predicted']

stim_idx = md[md.stim == 'CTRL'].index.tolist()
print('Unstimulated (PC1, PC2) coordinates of A only prediction:')
print(np.unique(tf_adata_all[stim_idx, :].obsm['X_pca'][:, :2]))

stim_idx = md[md.stim == 'STIM'].index.tolist()
print('Stimulation (PC1, PC2) coordinates of A only prediction:')
print(np.unique(tf_adata_all[stim_idx, :].obsm['X_pca'][:, :2]))

remaining_components = ['Full Model', 'Global Bias Only', 'Categorical Bias Only', 
                       'Adjacency Matrix Only', 'Total Bias', 'Adj + Global Bias', 
                       'Adj + Categorical Bias']


# In[186]:


if visualize: 
    fig, axes = plt.subplots(ncols = ncols, nrows = nrows, figsize = (5.1*ncols, 5.1*nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])
    
    
    ax_top = axes[0,0]
    marker_dict = {'CTRL': 'o', 'STIM': '^'}  

    for j, remove_component in enumerate(remove_components):
        if type(remove_component) == list:
            remove_component = '_'.join(remove_component)
        tf_res = tf_res_.copy()
        tf_adata_all = tf_res[remove_component]

        tf_adata_viz = tf_adata_all[tf_adata_all.obs[tf_adata_all.obs.seurat_annotations.isin(cell_types)].index.tolist(),:].copy()

        tf_adata_viz = tf_adata_viz.copy()  
        tf_adata_viz.obs['condition'] = (
            tf_adata_viz.obs['stim'] + '^' + 
            tf_adata_viz.obs['seurat_annotations'] + '^' + 
            tf_adata_viz.obs['batch']
        )

        np.random.seed(seed)
        tf_adata_viz = tf_adata_viz[np.random.permutation(tf_adata_viz.obs_names), :]

        viz_df = adata_dimviz_prediction(adata = tf_adata_viz, reduction_type = 'pca',
                                         cats = ['seurat_annotations', 'batch', 'stim', 'condition'],
                                         max_condition_size = None if counterfactual_type == 'opposite' else 3000,
                                         seed = seed_split)
        viz_df.seurat_annotations = pd.Categorical(viz_df.seurat_annotations, 
                                                  categories = cell_types, ordered=True)
        viz_df.stim = pd.Categorical(viz_df.stim, categories = ['CTRL', 'STIM'], ordered = True)


        for i, cell_type in enumerate(cell_types):
            viz_df_ = viz_df[viz_df.seurat_annotations == cell_type]

            stim_pred = viz_df_[viz_df_.batch == 'predicted'].stim.iloc[0]
            order = [stim_pred + '^' + cell_type + '^' + 'predicted', 
                    stim_pred + '^' + cell_type + '^' + 'actual', 
                    rev_stim[stim_pred] + '^' + cell_type + '^' + 'actual']
            viz_df_ = viz_df_.copy()
            order_values = ['predicted', 'test', 'train']
            order_map = dict(zip(order, order_values))
            viz_df_['condition'] = viz_df_.condition.map(order_map)

            viz_df_['condition'] = pd.Categorical(viz_df_.condition, 
                                                  categories = order_values, ordered=True)

            np.random.seed(seed_split)
            viz_df_ = viz_df_.loc[np.random.permutation(viz_df_.index),:]

            if i == 0 and j == 0:
                legend = True
            else: 
                legend = False

            if remove_component in ['global_bias', 'total_bias', 'adj_global_bias']:
                viz_df_1 = viz_df_[viz_df_.condition.isin(['test', 'train'])].copy()
                viz_df_1.condition = viz_df_1.condition.cat.remove_unused_categories()
                sns.scatterplot(data = viz_df_1, 
                                x = 'PC1', y = 'PC2', 
                                hue = 'condition', palette = sns.color_palette("deep")[1:3],
                                style = 'stim', markers = marker_dict, s = 20, 
                                ax = axes[i, j], legend = legend)

                viz_df_2 = viz_df_[viz_df_.condition == 'predicted'].copy()
                viz_df_2.condition = viz_df_2.condition.cat.remove_unused_categories()
                if not viz_df_2.PC1.nunique():
                    raise ValueError('Unexpected distribution in predicted values')
                else:
                    viz_df_2 = pd.DataFrame(viz_df_2.iloc[0, :]).T
                sns.scatterplot(data = viz_df_2, 
                                x = 'PC1', y = 'PC2', 
                                color = sns.color_palette("deep")[0],
                                marker = marker_dict[viz_df_2.stim.tolist()[0]], s = 100, 
                               ax = axes[i, j], legend = legend)

            else:
                sns.scatterplot(data = viz_df_, 
                                x = 'PC1', y = 'PC2', hue = 'condition', 
                                style = 'stim', markers = marker_dict, s = 20,
                                ax = axes[i, j], legend = legend)


            if legend:
                handles, labels = axes[i, j].get_legend_handles_labels()
                labels[1:4] = ['predicted', 'test', 'train']
                axes[i, j].legend(handles=handles, labels=labels)
            axes[i,j].set_title(cell_type + ' | ' + remaining_components[j])


            if i == 0 and j == 0:
                legend_actual = axes[i,j].legend_  # Store the legend
                axes[i,j].legend_.remove() 

    fig.legend(handles=legend_actual.legendHandles,
               ncols = 2,
               labels=[t.get_text() for t in legend_actual.get_texts()], 
               loc="upper center", 
               fontsize = 15, 
               markerscale=2, 
               bbox_to_anchor=(0.5, 1.5),
               bbox_transform=ax_top.transAxes)
    fig.tight_layout()
    # plt.savefig('example.png', dpi=300, bbox_inches='tight')
    ("")


# ### Train - No Counterfactual

# In[ ]:


if visualize and train_pred:
    tf_res_ = io.read_pickled_object(os.path.join(data_path, 'trash', fn + '_predictions_train.pickle'))


    remove_components = ['none', 
                         ['adj', 'global_bias'],
                         'global_bias',
                         'total_bias',
                         ['adj', 'categorical_bias'],
                         'adj',
                         'categorical_bias']
    # map to more inuitive title
    remaining_components = ['Full Model', 'Categorical Bias Only', 
                            'Adj + Categorical Bias',
                           'Adjacency Matrix Only',  
                            'Global Bias Only', 'Total Bias', 'Adj + Global Bias', 
                           ]
    train_conds = sorted(train_cond, key=lambda x: (x.split('^')[1], x.split('^')[0]))
    cell_types = set([tc.split('^')[1] for tc in train_conds])
    test_ct = set([tc.split('^')[1] for tc in test_conds])
    cell_types = test_cell_types + train_cell_types
    nrows = len(train_conds)
    ncols = len(remove_components)

    counterfactual_type = 'opposite'

    remove_component = 'total_bias'
    tf_res = tf_res_.copy()
    tf_adata_all = tf_res[remove_component]
    md = tf_adata_all.obs

    md = md[md.batch == 'predicted']

    ctrl_idx = md[md.stim == 'CTRL'].index.tolist()
    ctrl_coords = np.unique(tf_adata_all[ctrl_idx, :].obsm['X_pca'][:, :2])
    print('Unstimulated (PC1, PC2) coordinates of A only prediction:')
    print(ctrl_coords)

    stim_idx = md[md.stim == 'STIM'].index.tolist()
    stim_coords = np.unique(tf_adata_all[stim_idx, :].obsm['X_pca'][:, :2])
    print('Stimulation (PC1, PC2) coordinates of A only prediction:')
    print(stim_coords)

    print('The shift in predictions of A only upon the introduction of stimulation is (PC1, PC2):')
    #print(stim_coords - ctrl_coords)


# In[ ]:


if visualize and train_pred: 
    fig, axes = plt.subplots(ncols = ncols, nrows = nrows, figsize = (5.1*ncols, 5.1*nrows))
    ax_top = axes[0, 0]

    marker_dict = {'CTRL': 'o', 'STIM': '^'}  

    for j, remove_component in enumerate(remove_components):
        if type(remove_component) == list:
            remove_component = '_'.join(remove_component)
        tf_res = tf_res_.copy()
        tf_adata_all = tf_res[remove_component]

        tf_adata_viz = tf_adata_all[tf_adata_all.obs[tf_adata_all.obs.seurat_annotations.isin(cell_types)].index.tolist(),:].copy()

        tf_adata_viz = tf_adata_viz.copy()  
        tf_adata_viz.obs['condition'] = (
            tf_adata_viz.obs['stim'] + '^' + 
            tf_adata_viz.obs['seurat_annotations'] + '^' + 
            tf_adata_viz.obs['batch']
        )

        np.random.seed(seed)
        tf_adata_viz = tf_adata_viz[np.random.permutation(tf_adata_viz.obs_names), :]

        viz_df = adata_dimviz_prediction(adata = tf_adata_viz, reduction_type = 'pca',
                                         cats = ['seurat_annotations', 'batch', 'stim', 'condition'],
                                         max_condition_size = None if counterfactual_type == 'opposite' else 3000,
                                         seed = seed_split)
        viz_df.seurat_annotations = pd.Categorical(viz_df.seurat_annotations, 
                                                  categories = cell_types, ordered=True)
        viz_df.stim = pd.Categorical(viz_df.stim, categories = ['CTRL', 'STIM'], ordered = True)


    #     # alternative with fewer plots would be to still iterate
    #     # by cell type and color by batch and marker by stim (visualizing both predictions for a cell type in one plot)
    #     # this visualization gets tricky to see
    #     for i, cell_type in enumerate(cell_types):
    #         viz_df_ = viz_df[viz_df.seurat_annotations == cell_type]

    #         np.random.seed(seed_split)
    #         viz_df_ = viz_df_.loc[np.random.permutation(viz_df_.index),:]

    #         # no ordering by condition or remapping
    #         sns.scatterplot(data = viz_df_, 
    #                     x = 'PC1', y = 'PC2', 
    #                     hue = 'batch', 
    #                     style = 'stim', markers = marker_dict, s = 20,
    #                     ax = axes[i,j], legend = legend)


        for i, cond in enumerate(train_conds):
            stim_pred, cell_type = cond.split('^')

            viz_df_ = viz_df[viz_df.seurat_annotations == cell_type]
            viz_df_ = viz_df_[((viz_df_.batch == 'predicted') & (viz_df_.stim == stim_pred)) |
                              (viz_df_.batch == 'actual')]

            order = [stim_pred + '^' + cell_type + '^' + 'predicted', 
                    stim_pred + '^' + cell_type + '^' + 'actual', 
                    rev_stim[stim_pred] + '^' + cell_type + '^' + 'actual']
            viz_df_ = viz_df_.copy()
            order_values = ['predicted', 'test', 'train']
            order_map = dict(zip(order, order_values))
            viz_df_['condition'] = viz_df_.condition.map(order_map)

            viz_df_['condition'] = pd.Categorical(viz_df_.condition, 
                                                  categories = order_values, ordered=True)

            np.random.seed(seed_split)
            viz_df_ = viz_df_.loc[np.random.permutation(viz_df_.index),:]

            if i == 0 and j == 0:
                legend = True
            else: 
                legend = False

            if remove_component in ['global_bias', 'total_bias', 'adj_global_bias']:
                viz_df_1 = viz_df_[viz_df_.condition.isin(['test', 'train'])].copy()
                viz_df_1.condition = viz_df_1.condition.cat.remove_unused_categories()
                sns.scatterplot(data = viz_df_1, 
                                x = 'PC1', y = 'PC2', 
                                hue = 'condition', palette = sns.color_palette("deep")[1:3],
                                style = 'stim', markers = marker_dict, s = 20, 
                                ax = axes[i, j], legend = legend)

                viz_df_2 = viz_df_[viz_df_.condition == 'predicted'].copy()
                viz_df_2.condition = viz_df_2.condition.cat.remove_unused_categories()
                if not viz_df_2.PC1.nunique():
                    raise ValueError('Unexpected distribution in predicted values')
                else:
                    viz_df_2 = pd.DataFrame(viz_df_2.iloc[0, :]).T
                sns.scatterplot(data = viz_df_2, 
                                x = 'PC1', y = 'PC2', 
                                color = sns.color_palette("deep")[0],
                                marker = marker_dict[viz_df_2.stim.tolist()[0]], s = 100, 
                               ax = axes[i, j], legend = legend)

            else:
                sns.scatterplot(data = viz_df_, 
                                x = 'PC1', y = 'PC2', hue = 'condition', 
                                style = 'stim', markers = marker_dict, s = 20,
                                ax = axes[i, j], legend = legend)


            if legend:
                handles, labels = axes[i, j].get_legend_handles_labels()
                labels[1:4] = ['predicted', 'test', 'train']
                axes[i, j].legend(handles=handles, labels=labels)
            axes[i,j].set_title(cell_type + ' | ' + remaining_components[j])


            if i == 0 and j == 0:
                legend_actual = axes[i,j].legend_  # Store the legend
                axes[i,j].legend_.remove() 

    fig.legend(handles=legend_actual.legendHandles,
               ncols = 2,
               labels=[t.get_text() for t in legend_actual.get_texts()], 
               loc="upper center", 
               fontsize = 15, 
               markerscale=2, 
               bbox_to_anchor=(0.5, 1.5),
               bbox_transform=ax_top.transAxes)
    fig.tight_layout()

    ("")


# ### Train - Counterfactual
# 
# <span style="color:red">Note that for the cell types that are in the test conditions, this is an OOD in the sense that gene expression has not been seen before.</span>
# 

# In[ ]:


if visualize:
    tf_res_ = io.read_pickled_object(os.path.join(data_path, 'trash', fn + '_predictions_train_counterfactual.pickle'))


    remove_components = ['none', 
                         ['adj', 'global_bias'],
                         'global_bias',
                         'total_bias',
                         ['adj', 'categorical_bias'],
                         'adj',
                         'categorical_bias']
    # map to more inuitive title
    remaining_components = ['Full Model', 'Categorical Bias Only', 
                            'Adj + Categorical Bias',
                           'Adjacency Matrix Only',  
                            'Global Bias Only', 'Total Bias', 'Adj + Global Bias', 
                           ]
    train_conds = sorted(train_cond, key=lambda x: (x.split('^')[1], x.split('^')[0]))
    cell_types = set([tc.split('^')[1] for tc in train_conds])
    test_ct = set([tc.split('^')[1] for tc in test_conds])
    cell_types = test_cell_types + train_cell_types
    nrows = len(train_conds)
    ncols = len(remove_components)

    counterfactual_type = 'opposite'

    remove_component = 'total_bias'
    tf_res = tf_res_.copy()
    tf_adata_all = tf_res[remove_component]
    md = tf_adata_all.obs

    md = md[md.batch == 'predicted']

    ctrl_idx = md[md.stim == 'CTRL'].index.tolist()
    ctrl_coords = np.unique(tf_adata_all[ctrl_idx, :].obsm['X_pca'][:, :2])
    print('Unstimulated (PC1, PC2) coordinates of A only prediction:')
    print(ctrl_coords)

    stim_idx = md[md.stim == 'STIM'].index.tolist()
    stim_coords = np.unique(tf_adata_all[stim_idx, :].obsm['X_pca'][:, :2])
    print('Stimulation (PC1, PC2) coordinates of A only prediction:')
    print(stim_coords)

    print('The shift in predictions of A only upon the introduction of stimulation is (PC1, PC2):')
    #print(stim_coords - ctrl_coords)

    if train_pred:
        fig, axes = plt.subplots(ncols = ncols, nrows = nrows, figsize = (5.1*ncols, 5.1*nrows))
        ax_top = axes[0, 0]

        marker_dict = {'CTRL': 'o', 'STIM': '^'}  

        for j, remove_component in enumerate(remove_components):
            if type(remove_component) == list:
                remove_component = '_'.join(remove_component)
            tf_res = tf_res_.copy()
            tf_adata_all = tf_res[remove_component]

            tf_adata_viz = tf_adata_all[tf_adata_all.obs[tf_adata_all.obs.seurat_annotations.isin(cell_types)].index.tolist(),:].copy()

            tf_adata_viz = tf_adata_viz.copy()  
            tf_adata_viz.obs['condition'] = (
                tf_adata_viz.obs['stim'] + '^' + 
                tf_adata_viz.obs['seurat_annotations'] + '^' + 
                tf_adata_viz.obs['batch']
            )

            np.random.seed(seed_split)
            tf_adata_viz = tf_adata_viz[np.random.permutation(tf_adata_viz.obs_names), :]

            viz_df = adata_dimviz_prediction(adata = tf_adata_viz, reduction_type = 'pca',
                                             cats = ['seurat_annotations', 'batch', 'stim', 'condition'],
                                             max_condition_size = None if counterfactual_type == 'opposite' else 3000,
                                             seed = seed_split)
            viz_df.seurat_annotations = pd.Categorical(viz_df.seurat_annotations, 
                                                      categories = cell_types, ordered=True)
            viz_df.stim = pd.Categorical(viz_df.stim, categories = ['CTRL', 'STIM'], ordered = True)

            for i, cond in enumerate(train_conds):
                stim_pred, cell_type = cond.split('^')

                viz_df_ = viz_df[viz_df.seurat_annotations == cell_type]
                viz_df_ = viz_df_[((viz_df_.batch == 'predicted') & (viz_df_.stim == stim_pred)) |
                                  (viz_df_.batch == 'actual')]

                order = [stim_pred + '^' + cell_type + '^' + 'predicted', 
                        stim_pred + '^' + cell_type + '^' + 'actual', 
                        rev_stim[stim_pred] + '^' + cell_type + '^' + 'actual']
                viz_df_ = viz_df_.copy()
                order_values = ['predicted', 'test', 'train']
                order_map = dict(zip(order, order_values))
                viz_df_['condition'] = viz_df_.condition.map(order_map)

                viz_df_['condition'] = pd.Categorical(viz_df_.condition, 
                                                      categories = order_values, ordered=True)

                np.random.seed(seed_split)
                viz_df_ = viz_df_.loc[np.random.permutation(viz_df_.index),:]

                if i == 0 and j == 0:
                    legend = True
                else: 
                    legend = False

                if remove_component in ['global_bias', 'total_bias', 'adj_global_bias']:
                    viz_df_1 = viz_df_[viz_df_.condition.isin(['test', 'train'])].copy()
                    viz_df_1.condition = viz_df_1.condition.cat.remove_unused_categories()
                    sns.scatterplot(data = viz_df_1, 
                                    x = 'PC1', y = 'PC2', 
                                    hue = 'condition', palette = sns.color_palette("deep")[1:3],
                                    style = 'stim', markers = marker_dict, s = 20, 
                                    ax = axes[i, j], legend = legend)

                    viz_df_2 = viz_df_[viz_df_.condition == 'predicted'].copy()
                    viz_df_2.condition = viz_df_2.condition.cat.remove_unused_categories()
                    if not np.allclose(viz_df_2.PC1, viz_df_2.PC1.iloc[0], atol = 1e-6):
                        raise ValueError('Unexpected distribution in predicted values')
                    else:
                        viz_df_2 = pd.DataFrame(viz_df_2.iloc[0, :]).T
                        pc_coords = 'PC coords: ({:.3f}, {:.3f})'.format(viz_df_2.PC1.tolist()[0], 
                                                          viz_df_2.PC2.tolist()[0])
                    sns.scatterplot(data = viz_df_2, 
                                    x = 'PC1', y = 'PC2', 
                                    color = sns.color_palette("deep")[0],
                                    marker = marker_dict[viz_df_2.stim.tolist()[0]], s = 100, 
                                   ax = axes[i, j], legend = legend)
                    axes[i, j].annotate(pc_coords, xy=(0.9, 0.95), xycoords='axes fraction',
                                        ha='right', va='top')

                else:
                    sns.scatterplot(data = viz_df_, 
                                    x = 'PC1', y = 'PC2', hue = 'condition', 
                                    style = 'stim', markers = marker_dict, s = 20,
                                    ax = axes[i, j], legend = legend)


                if legend:
                    handles, labels = axes[i, j].get_legend_handles_labels()
                    labels[1:4] = ['predicted', 'test', 'train']
                    axes[i, j].legend(handles=handles, labels=labels)
                axes[i,j].set_title(cell_type + ' | ' + remaining_components[j])


                if i == 0 and j == 0:
                    legend_actual = axes[i,j].legend_  # Store the legend
                    axes[i,j].legend_.remove() 

        fig.legend(handles=legend_actual.legendHandles,
                   ncols = 2,
                   labels=[t.get_text() for t in legend_actual.get_texts()], 
                   loc="upper center", 
                   fontsize = 15, 
                   markerscale=2, 
                   bbox_to_anchor=(0.5, 1.5),
                   bbox_transform=ax_top.transAxes)
        fig.tight_layout()
        fig.suptitle('In-Distribution Predictions without Counterfactual')

        ("")


# ### Train (no Counterfactual) + Test 
# 
# Same cell type in the same PC plot, all stim conditions

# In[ ]:


if visualize:
    train_pred = os.path.isfile(os.path.join(data_path, 'trash', fn + '_predictions_train.pickle'))
    if train_pred:
        remove_components = ['none', 
                             ['adj', 'global_bias'],
                             'global_bias',
                             'total_bias',
                             ['adj', 'categorical_bias'],
                             'adj',
                             'categorical_bias']

        import warnings

        tf_res_test = io.read_pickled_object(os.path.join(data_path, 'trash', fn + '_predictions.pickle'))
        tf_res_train = io.read_pickled_object(os.path.join(data_path, 'trash', fn + '_predictions_train.pickle'))
        tf_res_ = {}

        for remove_type, tf_res in tf_res_train.items():
            tf_res.obs['prediction_type'] = 'train'
            tf_res_test[remove_type].obs['prediction_type'] = 'test'

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                tf_res = sc.concat([tf_res_test[remove_type], tf_res], axis = 0, join = 'outer')
                tf_res.obs_names_make_unique()

            tf_res_[remove_type] = tf_res


        # map to more inuitive title
        remaining_components = ['Full Model', 'Categorical Bias Only', 
                                'Adj + Categorical Bias',
                               'Adjacency Matrix Only',  
                                'Global Bias Only', 'Total Bias', 'Adj + Global Bias', 
                               ]

        tf_res_temp = tf_res.copy()
    #     test_conds = tf_res_temp.obs[(tf_res_temp.obs.batch == 'predicted') & (tf_res_temp.obs.prediction_type == 'test')].condition.unique().tolist()
    #     train_conds = set(tf_res_temp.obs.condition).difference(test_conds)
    #     # test_conds = sorted(test_conds,
    #     #                     key=lambda x: (x.split('^')[1], x.split('^')[0]))
    #     train_conds = sorted(train_conds, 
    #                              key=lambda x: (x.split('^')[1], x.split('^')[0]))

    #     train_conds = test_conds + train_conds


        cell_types = test_cell_types + train_cell_types
        nrows = len(cell_types)
        ncols = len(remove_components)


# In[ ]:


if train_pred and visualize:
    fig, axes = plt.subplots(ncols = ncols, nrows = nrows, figsize = (5.1*ncols, 5.1*nrows))
    ax_top = axes[0, 0]

    marker_dict = {'CTRL': 'o', 'STIM': '^'}  
    
    palette = sns.color_palette('deep')
    palette = palette[4:8]
    palette = [palette[-1], palette[1], palette[0], palette[2]]

    for j, remove_component in enumerate(remove_components):
        if type(remove_component) == list:
            remove_component = '_'.join(remove_component)
        tf_res = tf_res_.copy()
        tf_adata_all = tf_res[remove_component]

        tf_adata_viz = tf_adata_all.copy()#tf_adata_all[tf_adata_all.obs[tf_adata_all.obs.seurat_annotations.isin(cell_types)].index.tolist(),:].copy()

#         tf_adata_viz = tf_adata_viz.copy()  
        tf_adata_viz.obs['condition'] = (
            tf_adata_viz.obs['stim'] + '^' + 
            tf_adata_viz.obs['seurat_annotations'] + '^' + 
            tf_adata_viz.obs['batch']
        )

        np.random.seed(seed_split)
        tf_adata_viz = tf_adata_viz[np.random.permutation(tf_adata_viz.obs_names), :]

        viz_df = adata_dimviz_prediction(adata = tf_adata_viz, reduction_type = 'pca',
                                         cats = ['seurat_annotations', 'batch', 'stim', 'condition'],
                                         max_condition_size = None if counterfactual_type == 'opposite' else 3000,
                                         seed = seed_split)
        viz_df.seurat_annotations = pd.Categorical(viz_df.seurat_annotations, 
                                                  categories = cell_types, ordered=True)
        viz_df.stim = pd.Categorical(viz_df.stim, categories = ['CTRL', 'STIM'], ordered = True)


    #     # alternative with fewer plots would be to still iterate
    #     # by cell type and color by batch and marker by stim (visualizing both predictions for a cell type in one plot)
    #     # this visualization gets tricky to see
    #     for i, cell_type in enumerate(cell_types):
    #         viz_df_ = viz_df[viz_df.seurat_annotations == cell_type]

    #         np.random.seed(seed_split)
    #         viz_df_ = viz_df_.loc[np.random.permutation(viz_df_.index),:]

    #         # no ordering by condition or remapping
    #         sns.scatterplot(data = viz_df_, 
    #                     x = 'PC1', y = 'PC2', 
    #                     hue = 'batch', 
    #                     style = 'stim', markers = marker_dict, s = 20,
    #                     ax = axes[i,j], legend = legend)


        for i, cell_type in enumerate(cell_types):

            viz_df_ = viz_df[viz_df.seurat_annotations == cell_type].copy()
            viz_df_.condition = pd.Categorical(viz_df_.batch.astype('str') + '^' + viz_df_.stim.astype('str'), 
                                              categories = ['actual^CTRL', 'actual^STIM',  
                                                            'predicted^CTRL', 'predicted^STIM'],
                                               ordered = True)

            # max max_n points ( + permute)
            max_n = 2000
            np.random.seed(seed_split)
            permute_size = min(max_n, viz_df_.shape[0])
            rand_idx = np.random.choice(viz_df_.shape[0], permute_size)
            viz_df_ = viz_df_.iloc[rand_idx, :]

            # equal size
            min_size = viz_df_['condition'].value_counts().min()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                viz_df_ = viz_df_.groupby('condition', group_keys=False, observed=False).apply(lambda x: x.sample(min_size, random_state=seed_split)).copy()

            if i == 0 and j == 0:
                legend = True
            else: 
                legend = False

            if remove_component in ['global_bias', 'total_bias', 'adj_global_bias']:
                n_unique_conds = 1 if remove_component == 'adj_global_bias' else 2
                
                viz_df_1 = viz_df_[viz_df_.batch == 'actual'].copy()
                viz_df_1.condition = viz_df_1.condition.cat.remove_unused_categories()
                sns.scatterplot(data = viz_df_1, 
                                x = 'PC1', y = 'PC2', 
                                hue = 'condition', palette = palette[:2],
                                style = 'stim', markers = marker_dict, s = 20, 
                                ax = axes[i, j], legend = legend)
                viz_df_2 = viz_df_[viz_df_.batch == 'predicted'].copy()
                viz_df_2.condition = viz_df_2.condition.cat.remove_unused_categories()

                if not viz_df_2.PC1.nunique() == n_unique_conds:
                    raise ValueError('Unexpected distribution in predicted values')
                viz_df_2 = viz_df_2.drop_duplicates(subset='PC1', keep='first')
                viz_df_2.condition = viz_df_2.condition.cat.remove_unused_categories()


                sns.scatterplot(data = viz_df_2, 
                                x = 'PC1', y = 'PC2', 
                                hue = 'condition', palette = palette[2:],
                                style = 'stim', markers = marker_dict, s = 100,
                               ax = axes[i, j], legend = legend)
                
                if n_unique_conds == 2:
                    pc_coords_stim = viz_df_2[viz_df_2.stim == 'STIM'][['PC1', 'PC2']].values.tolist()[0]
                    pc_coords_ctrl = viz_df_2[viz_df_2.stim == 'CTRL'][['PC1', 'PC2']].values.tolist()[0]
                    diff=np.array(pc_coords_ctrl) - np.array(pc_coords_stim)
                    
                    pc_coords_stim = 'PC coords STIM: ({:.3f}, {:.3f})'.format(*pc_coords_stim)
                    pc_coords_ctrl = 'PC coords CTRL: ({:.3f}, {:.3f})'.format(*pc_coords_ctrl)
                    diff = 'Diff (CTRL - STIM): ({:.3f}, {:.3f})'.format(*diff)
                    
                    
                    axes[i, j].annotate(pc_coords_ctrl, xy=(0.9, 0.95), xycoords='axes fraction',
                                        ha='right', va='top')
                    axes[i, j].annotate(pc_coords_stim, xy=(0.9, 0.9), xycoords='axes fraction',
                                        ha='right', va='top')
                    axes[i, j].annotate(diff, xy=(0.9, 0.85), xycoords='axes fraction',
                                        ha='right', va='top')
                    
                    
                else:
                    pc_coords = viz_df_2.iloc[0, :2].tolist()
                    pc_coords = 'PC coords cat: ({:.3f}, {:.3f})'.format(*pc_coords)
                    
                    axes[i, j].annotate(pc_coords, xy=(0.9, 0.95), xycoords='axes fraction',
                                        ha='right', va='top')



            else:
                sns.scatterplot(data = viz_df_, 
                                x = 'PC1', y = 'PC2', 
                                hue = 'condition', palette = palette, 
                                style = 'stim', markers = marker_dict, s = 20,
                                ax = axes[i, j], legend = legend)


            if legend:
                handles, labels = axes[i, j].get_legend_handles_labels()
#                 labels[1:4] = ['predicted', 'test', 'train']
                axes[i, j].legend(handles=handles, labels=labels)
            axes[i,j].set_title(cell_type + ' | ' + remaining_components[j])


            if i == 0 and j == 0:
                legend_actual = axes[i,j].legend_  # Store the legend
                axes[i,j].legend_.remove() 

                
    split_idx = 5
    handles_ordered = handles[:split_idx] + handles[split_idx:]
    labels_ordered = labels[:split_idx] + labels[split_idx:]

    fig.legend(
        handles=handles_ordered,
        labels=labels_ordered,
        ncol=2,
        loc="upper center",
        fontsize=15,
        markerscale=2,
        bbox_to_anchor=(0.5, 1.5),
        bbox_transform=ax_top.transAxes
    )
    fig.tight_layout()

    ("")


# #### Probe Classifier - Train Biases (no counterfactual)
# 
# On the training data, we see how well a second classifier trained on the global bias can achieve separation using 5-fold CV. This is because despite seeing decent mixing in the UMAP/NMI, we still see that the full model forward pass of the global bias only separates by perturbation information.
# 
# Thus we test the following outputs associated with the global bias generated by the model on the train data without counterfactuals:
# - generator output (global bias): "train_generator"
# - forward pass using __only__ the global bias, excluding the ProjectOutput layer: "train_bionet"
# - a full forward pass using __only__ the global bias: "train_fullforward"
# - the generator output on the test data with counterfactuals (which really is just a subset of the train data): "test" <-- analogous to "train_generator"

# In[ ]:


# get the forward pass prior ProjectOutput and with only the global bias on the train data 
global_bias = get_prediction(mod = mod,
                              train_cells = train_cells,
                              tf_adata = tf_adata,
                              train_mode = True,
                              counterfactual = False,
                              remove_type = ['adj', 'categorical_bias'],
                              return_bias = False,
                                    max_cells = 2000, 
                     return_full = True)
embed_tf_activity(global_bias, 
                  scanpy_pca = False, 
                  cluster_col_name = 'leiden', 
                  resolution = 1,
                  n_components = 10 if global_bias.shape[0] < 50 else 10)
io.write_pickled_object(global_bias, 
                       os.path.join(data_path, 'trash', fn + 'Yfull_gb.pickle'))



# full forward pass on train data only (no counterfactual) using only the global bias
tf_res_train = io.read_pickled_object(os.path.join(data_path, 'trash', fn + '_predictions_train.pickle'))
global_bias = tf_res_train['adj_categorical_bias']
global_bias = global_bias[global_bias.obs.batch == 'predicted',:].copy()
# re-run embedding -- was projected during prediction
embed_tf_activity(global_bias, 
                  scanpy_pca = False, 
                  cluster_col_name = 'leiden', 
                  resolution = 1,
                  n_components = 10 if global_bias.shape[0] < 50 else 10)
io.write_pickled_object(global_bias, 
                       os.path.join(data_path, 'trash', fn + 'yhat_gb.pickle'))


def setup_predtypes():
    """Various stages of model output using only the global bias"""
    
    pred_types = {}
    # global bias output from train data only (no counterfactual)
    global_bias = io.read_pickled_object(os.path.join(data_path, 'trash', fn + '_feb_clustered_biases_train.pickle'))
    global_bias = global_bias['opposite'][0]
    pred_types['train_generator'] = global_bias

    # forward pass with no ProjectOutput
    pred_types['train_bionet'] = io.read_pickled_object(os.path.join(data_path, 'trash', fn + 'Yfull_gb.pickle'))

    # full forward pass on train data only (no counterfactual) using only the global bias
    pred_types['train_fullforward'] = io.read_pickled_object(os.path.join(data_path, 'trash', fn + 'yhat_gb.pickle'))

    if run_type == 'E':
        # global bias output from test  (counterfactuals, meaning input gene expression is still in train)
        global_bias = io.read_pickled_object(os.path.join(data_path, 'trash', fn + '_feb_clustered_biases.pickle'))
        global_bias = global_bias['opposite'][0]
        pred_types['test'] = global_bias
    
    return pred_types


# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# set up the outputs to classify
pred_types = setup_predtypes()


probe_res = pd.DataFrame(columns=['category_name', 'linear_accuracy', 'nonlinear_accuracy', 
                                  'prediction_type'])
n_folds = 5
counter = 0

for pred_type, global_bias in pred_types.items():
    bias_global = global_bias.to_df()
    obs = global_bias.obs

    categories = {
        'cell_type': obs.seurat_annotations,
        'perturbation': obs.stim
    }

    for category_name, category_labels in categories.items():
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed_split)
        X = bias_global.to_numpy()
        y = category_labels.to_numpy()

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            
            # linear probe
            probe_linear = LogisticRegression(max_iter=2000, solver='lbfgs', multi_class='auto', 
                                              random_state=seed_split)
            probe_linear.fit(X_train, y_train)

            y_pred = probe_linear.predict(X_test)
            linear_accuracy = accuracy_score(y_test, y_pred)
            
            # nonlinear probe
            probe_nonlinear = RandomForestClassifier(
                n_estimators=100,          
                max_depth=None,            
                random_state=seed_split,
                n_jobs=20,                 
                verbose=False                  
            )

            probe_nonlinear.fit(X_train, y_train)
            y_pred = probe_nonlinear.predict(X_test)
            nonlinear_accuracy = accuracy_score(y_test, y_pred)
            
        
#             mlp_probe = MLPClassifier(hidden_layer_sizes=(100,),
#                                       activation='relu', 
#                                       solver = 'adam', 
#                                       max_iter=500, 
#                                      shuffle = True, 
#                                      random_state = seed_split)
#             mlp_probe.fit(X_train, y_train)
#             y_pred = mlp_probe.predict(X_test)
#             nonlinear_accuracy = accuracy_score(y_test, y_pred)
            

            probe_res.loc[counter, 'category_name'] = category_name
            probe_res.loc[counter, 'linear_accuracy'] = linear_accuracy
            probe_res.loc[counter, 'nonlinear_accuracy'] = nonlinear_accuracy
            probe_res.loc[counter, 'prediction_type'] = pred_type
            
            
            counter += 1

probe_res = probe_res.melt(
    id_vars=['prediction_type', 'category_name'], 
    value_vars=['linear_accuracy', 'nonlinear_accuracy'],
    var_name='probe_type', 
    value_name='accuracy'
)

# Clean up probe_type values
probe_res['probe_type'] = probe_res['probe_type'].str.replace('_accuracy', '')

# Explode if accuracies are stored as lists (e.g., one per fold)
probe_res = probe_res.explode('accuracy').reset_index(drop=True)
probe_res.to_csv(os.path.join(data_path, 'trash', fn + '_probe.csv'))


# The following plot visualizes accuracy of a Logistic Regression (linear)  and RandomForest (nonlinear) classifier trained on the global bias output of the fully trained scLEMBAS model. 
# - "train": direct global bias output on the train conditions (no counterfactuals)
# - "train_fullforward": full forward pass using just the global bias output (run through the ProjectOutput layer) (no counterfactuals)
# - "test": direct global bias output on the test conditions (with counterfactual, so input g has been seen during training -- this is then just a subset of train)
# 
# For each output, the Logistic classifier was trained to predict either the perturbation or cell type information. Gray dashed lines represent prediction of the model if it was random chance (1/n_labels). Values above this mean a lack of removal of information. Distributions are from multiple train-test splits of the global bias when running the Logistic classifier.

# In[ ]:


if visualize:
    probe_res = pd.read_csv(os.path.join(data_path, 'trash', fn + '_probe.csv'), index_col = 0)
    train_types = ['train_generator', 'train_bionet', 'train_fullforward']
    probe_res.prediction_type = pd.Categorical(probe_res.prediction_type, 
                                              categories = train_types +  ['test'], 
                                              ordered = True)
    probe_res.prediction_type = probe_res.prediction_type.cat.remove_unused_categories().copy()
    probe_res.probe_type = pd.Categorical(probe_res.probe_type, 
                                         categories = ['linear', 'nonlinear'], 
                                         ordered = True)
    
    
    # random chance given class imbalances and that StratifiedKfold gives same split as real
    pert_rand_train = np.square(tf_adata[train_cells, :].obs.stim.value_counts(normalize = True)).sum()
    ct_rand_train = np.square(tf_adata[train_cells, :].obs.seurat_annotations.value_counts(normalize = True)).sum()
    
    pert_rand_test = np.square(tf_adata[test_cells, :].obs.stim.value_counts(normalize = True)).sum()
    ct_rand_test = np.square(tf_adata[test_cells, :].obs.seurat_annotations.value_counts(normalize = True)).sum()    

    random_chance_ = {'cell_type': {'train': ct_rand_train, 
                                  'test': ct_rand_test},
                     'perturbation': {'train': pert_rand_train, 
                                  'test': pert_rand_test}
                    }
    
    
    fig, ax = plt.subplots(figsize = (7,3), ncols = 2)

    for (i, c) in enumerate(['cell_type', 'perturbation']):
        viz_df = probe_res[probe_res.category_name == c]
        sns.boxplot(data = viz_df, x = 'prediction_type', y = 'accuracy', hue = 'probe_type', ax = ax[i])

        if c == 'cell_type':
            random_chance = {'train': 1/len(set([cond.split('^')[1] for cond in train_conds])), 
                             'test': 1/len(set([cond.split('^')[1] for cond in test_conds]))}
        else:
            random_chance = {'train': 0.5, 'test': 0.5}
            
        random_chance = {'test': random_chance_[c]['test']}
        for key in train_types:
            random_chance[key] = random_chance_[c]['train']
        
#         del random_chance['train']

        iter_ticks = train_types + ['test'] if run_type == 'E' else train_types
        for xtick, label in enumerate(iter_ticks):
            ax[i].hlines(
                y=random_chance[label], xmin=xtick - 0.3, xmax=xtick + 0.3, 
                colors='gray', linestyles='dashed', linewidth=1
            )
            
        ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=45)
        ax[i].set_title(c)
        ax[i].set_ylabel('Accuracy')
        ax[i].legend_.remove()

        
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title='Probe Type',
        loc='upper left',
        bbox_to_anchor=(-0.2, 0.9),
    )
        
    fig.suptitle('Probe Classifier: 5-fold CV')
    fig.tight_layout()
    ("")


# In[2]:


probe_stats = probe_res.groupby(['category_name', 'prediction_type', 'probe_type'], observed=True).mean()
print(probe_stats)


# In[ ]:


if visualize: 
    pred_types = setup_predtypes()
    ncols = len(pred_types)
    nrows = 3

    fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize = (5.1*ncols, 5.1*nrows))

    cat_map = {'seurat_annotations': 'Cell Type', 
              'stim': 'Stimulation'}
    reduction_map = {'umap': 'UMAP', 
                    'pca': 'PC'}

    cat = 'stim'
    for j, (pred_type, global_bias) in enumerate(pred_types.items()):
        for i, reduction_type in enumerate(['umap', 'pca']): 
            viz_df, nmi = adata_dimviz_bias(adata = global_bias, 
                                    reduction_type = reduction_type, 
                                    cat = cat,
                                    subset_size = 5000 if global_bias.shape[0] > 5000 else None)
            sns.scatterplot(data = viz_df, 
                            x = reduction_map[reduction_type] + '1', 
                            y = reduction_map[reduction_type] + '2', 
                            hue = cat, s=10, ax = ax[i,j])
            ax[i,j].annotate('NMI (Leiden Clusters, ' + cat_map[cat] + '): {:.3f}'.format(nmi),
                    xy = (0.25, 0.95), xycoords='axes fraction', fontsize = 9)
            ax[i,j].set_title(pred_type)
            ax[i,j].legend().set_visible(False)

            if reduction_type == 'pca':
                sns.scatterplot(data = viz_df, 
                            x = reduction_map[reduction_type] + '3', 
                            y = reduction_map[reduction_type] + '4', 
                            hue = cat, s=10, ax = ax[i+1,j])
                ax[i+1,j].set_title('')
                ax[i+1,j].legend().set_visible(False)




    fig.suptitle('Stimulation Embeddings - Global Bias Outputs Only')
    fig.tight_layout()
    ("")


# In[ ]:


pred_types = setup_predtypes()
ncols = len(pred_types)
nrows = 3

fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize = (5.1*ncols, 5.1*nrows))

cat_map = {'seurat_annotations': 'Cell Type', 
          'stim': 'Stimulation'}
reduction_map = {'umap': 'UMAP', 
                'pca': 'PC'}

cat = 'seurat_annotations'
for j, (pred_type, global_bias) in enumerate(pred_types.items()):
    for i, reduction_type in enumerate(['umap', 'pca']): 
        viz_df, nmi = adata_dimviz_bias(adata = global_bias, 
                                reduction_type = reduction_type, 
                                cat = cat,
                                subset_size = 5000 if global_bias.shape[0] > 5000 else None)
        sns.scatterplot(data = viz_df, 
                        x = reduction_map[reduction_type] + '1', 
                        y = reduction_map[reduction_type] + '2', 
                        hue = cat, s=10, ax = ax[i,j])
        ax[i,j].annotate('NMI (Leiden Clusters, ' + cat_map[cat] + '): {:.3f}'.format(nmi),
                xy = (0.25, 0.95), xycoords='axes fraction', fontsize = 9)
        ax[i,j].set_title(pred_type)
        ax[i,j].legend().set_visible(False)
        
        if reduction_type == 'pca':
            sns.scatterplot(data = viz_df, 
                        x = reduction_map[reduction_type] + '3', 
                        y = reduction_map[reduction_type] + '4', 
                        hue = cat, s=10, ax = ax[i+1,j])
            ax[i+1,j].set_title('')
            ax[i+1,j].legend().set_visible(False)
            
        
        
        
fig.suptitle('Cell Type Embeddings - Global Bias Outputs Only')
fig.tight_layout()
("")


# ### Train (with Counterfactual) + Test 

# In[ ]:


if visualize:
    train_pred = os.path.isfile(os.path.join(data_path, 'trash', fn + '_predictions_train_counterfactual.pickle'))
    if train_pred:
        remove_components = ['none', 
                             ['adj', 'global_bias'],
                             'global_bias',
                             'total_bias',
                             ['adj', 'categorical_bias'],
                             'adj',
                             'categorical_bias']

        import warnings

        tf_res_test = io.read_pickled_object(os.path.join(data_path, 'trash', fn + '_predictions.pickle'))
        tf_res_train = io.read_pickled_object(os.path.join(data_path, 'trash', fn + '_predictions_train_counterfactual.pickle'))
        tf_res_ = {}

        for remove_type, tf_res in tf_res_train.items():
            tf_res.obs['prediction_type'] = 'train'
            tf_res_test[remove_type].obs['prediction_type'] = 'test'

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                tf_res = sc.concat([tf_res_test[remove_type], tf_res], axis = 0, join = 'outer')
                tf_res.obs_names_make_unique()

            tf_res_[remove_type] = tf_res


        # map to more inuitive title
        remaining_components = ['Full Model', 'Categorical Bias Only', 
                                'Adj + Categorical Bias',
                               'Adjacency Matrix Only',  
                                'Global Bias Only', 'Total Bias', 'Adj + Global Bias', 
                               ]

        tf_res_temp = tf_res.copy()

        cell_types = test_cell_types + train_cell_types
        nrows = len(cell_types)
        ncols = len(remove_components)


    if train_pred:
        fig, axes = plt.subplots(ncols = ncols, nrows = nrows, figsize = (5.1*ncols, 5.1*nrows))
        ax_top = axes[0, 0]

        marker_dict = {'CTRL': 'o', 'STIM': '^'}  

        palette = sns.color_palette('deep')
        palette = palette[4:8]
        palette = [palette[-1], palette[1], palette[0], palette[2]]

        for j, remove_component in enumerate(remove_components):
            if type(remove_component) == list:
                remove_component = '_'.join(remove_component)
            tf_res = tf_res_.copy()
            tf_adata_all = tf_res[remove_component]

            tf_adata_viz = tf_adata_all.copy()#tf_adata_all[tf_adata_all.obs[tf_adata_all.obs.seurat_annotations.isin(cell_types)].index.tolist(),:].copy()

    #         tf_adata_viz = tf_adata_viz.copy()  
            tf_adata_viz.obs['condition'] = (
                tf_adata_viz.obs['stim'] + '^' + 
                tf_adata_viz.obs['seurat_annotations'] + '^' + 
                tf_adata_viz.obs['batch']
            )

            np.random.seed(seed_split)
            tf_adata_viz = tf_adata_viz[np.random.permutation(tf_adata_viz.obs_names), :]

            viz_df = adata_dimviz_prediction(adata = tf_adata_viz, reduction_type = 'pca',
                                             cats = ['seurat_annotations', 'batch', 'stim', 'condition'],
                                             max_condition_size = None if counterfactual_type == 'opposite' else 3000,
                                             seed = seed_split)
            viz_df.seurat_annotations = pd.Categorical(viz_df.seurat_annotations, 
                                                      categories = cell_types, ordered=True)
            viz_df.stim = pd.Categorical(viz_df.stim, categories = ['CTRL', 'STIM'], ordered = True)

            for i, cell_type in enumerate(cell_types):

                viz_df_ = viz_df[viz_df.seurat_annotations == cell_type].copy()
                viz_df_.condition = pd.Categorical(viz_df_.batch.astype('str') + '^' + viz_df_.stim.astype('str'), 
                                                  categories = ['actual^CTRL', 'actual^STIM',  
                                                                'predicted^CTRL', 'predicted^STIM'],
                                                   ordered = True)

                # max max_n points ( + permute)
                max_n = 2000
                np.random.seed(seed_split)
                permute_size = min(max_n, viz_df_.shape[0])
                rand_idx = np.random.choice(viz_df_.shape[0], permute_size)
                viz_df_ = viz_df_.iloc[rand_idx, :]

                # equal size
                min_size = viz_df_['condition'].value_counts().min()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=DeprecationWarning)
                    viz_df_ = viz_df_.groupby('condition', group_keys=False, observed=False).apply(lambda x: x.sample(min_size, random_state=seed_split)).copy()

                if i == 0 and j == 0:
                    legend = True
                else: 
                    legend = False

                if remove_component in ['global_bias', 'total_bias', 'adj_global_bias']:
                    n_unique_conds = 1 if remove_component == 'adj_global_bias' else 2

                    viz_df_1 = viz_df_[viz_df_.batch == 'actual'].copy()
                    viz_df_1.condition = viz_df_1.condition.cat.remove_unused_categories()
                    sns.scatterplot(data = viz_df_1, 
                                    x = 'PC1', y = 'PC2', 
                                    hue = 'condition', palette = palette[:2],
                                    style = 'stim', markers = marker_dict, s = 20, 
                                    ax = axes[i, j], legend = legend)
                    viz_df_2 = viz_df_[viz_df_.batch == 'predicted'].copy()
                    viz_df_2.condition = viz_df_2.condition.cat.remove_unused_categories()

    #                 if not viz_df_2.PC1.nunique() == n_unique_conds:
                    unique_vals = np.unique(viz_df_2.PC1.values)
                    grouped_vals = []

                    for val in unique_vals:
                        if not any(np.isclose(val, g, rtol=1e-5, atol=1e-6) for g in grouped_vals):
                            grouped_vals.append(val)

                    if len(grouped_vals) != n_unique_conds:
                        raise ValueError('Unexpected distribution in predicted values')
                    viz_df_2 = viz_df_2.drop_duplicates(subset='PC1', keep='first')
                    viz_df_2.condition = viz_df_2.condition.cat.remove_unused_categories()

                    if n_unique_conds == 2:
                        pc_coords_stim = viz_df_2[viz_df_2.stim == 'STIM'][['PC1', 'PC2']].values.tolist()[0]
                        pc_coords_ctrl = viz_df_2[viz_df_2.stim == 'CTRL'][['PC1', 'PC2']].values.tolist()[0]
                        diff=np.array(pc_coords_ctrl) - np.array(pc_coords_stim)

                        pc_coords_stim = 'PC coords STIM: ({:.3f}, {:.3f})'.format(*pc_coords_stim)
                        pc_coords_ctrl = 'PC coords CTRL: ({:.3f}, {:.3f})'.format(*pc_coords_ctrl)
                        diff = 'Diff (CTRL - STIM): ({:.3f}, {:.3f})'.format(*diff)


                        axes[i, j].annotate(pc_coords_ctrl, xy=(0.9, 0.95), xycoords='axes fraction',
                                            ha='right', va='top')
                        axes[i, j].annotate(pc_coords_stim, xy=(0.9, 0.9), xycoords='axes fraction',
                                            ha='right', va='top')
                        axes[i, j].annotate(diff, xy=(0.9, 0.85), xycoords='axes fraction',
                                            ha='right', va='top')


                    else:
                        pc_coords = viz_df_2.iloc[0, :2].tolist()
                        pc_coords = 'PC coords cat: ({:.3f}, {:.3f})'.format(*pc_coords)

                        axes[i, j].annotate(pc_coords, xy=(0.9, 0.95), xycoords='axes fraction',
                                            ha='right', va='top')



                else:
                    sns.scatterplot(data = viz_df_, 
                                    x = 'PC1', y = 'PC2', 
                                    hue = 'condition', palette = palette, 
                                    style = 'stim', markers = marker_dict, s = 20,
                                    ax = axes[i, j], legend = legend)


                if legend:
                    handles, labels = axes[i, j].get_legend_handles_labels()
    #                 labels[1:4] = ['predicted', 'test', 'train']
                    axes[i, j].legend(handles=handles, labels=labels)
                axes[i,j].set_title(cell_type + ' | ' + remaining_components[j])


                if i == 0 and j == 0:
                    legend_actual = axes[i,j].legend_  # Store the legend
                    axes[i,j].legend_.remove() 


        split_idx = 5
        handles_ordered = handles[:split_idx] + handles[split_idx:]
        labels_ordered = labels[:split_idx] + labels[split_idx:]

        fig.legend(
            handles=handles_ordered,
            labels=labels_ordered,
            ncol=2,
            loc="upper center",
            fontsize=15,
            markerscale=2,
            bbox_to_anchor=(0.5, 1.5),
            bbox_transform=ax_top.transAxes
        )
        fig.tight_layout()

        ("")


# ## 7.3 Visualize first 3 PCs of full model

# In[190]:


tf_res = io.read_pickled_object(os.path.join(data_path, 'trash', fn + '_predictions.pickle'))
tf_adata_all = tf_res['none']


# In[191]:


n_pcs = 3


# In[194]:


if visualize: 
    marker_dict = {'CTRL': 'o', 'STIM': '^'}  


    pc_combs = list(itertools.combinations(range(1, n_pcs + 1), 2))

    ncols = len(test_conds)
    nrows = len(pc_combs)
    fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize = (5.1*ncols, 5.1*nrows))
    if nrows == 1 and ncols == 1:
        ax = np.array([[ax]])
    elif nrows == 1:
        ax = np.array([ax])
    elif ncols == 1:
        ax = np.array([[a] for a in ax])


    cell_types = test_cell_types

    tf_adata_viz = tf_adata_all[tf_adata_all.obs[tf_adata_all.obs.seurat_annotations.isin(cell_types)].index.tolist(),:].copy()

    tf_adata_viz = tf_adata_viz.copy()  
    tf_adata_viz.obs['condition'] = (
        tf_adata_viz.obs['stim'] + '^' + 
        tf_adata_viz.obs['seurat_annotations'] + '^' + 
        tf_adata_viz.obs['batch']
    )

    np.random.seed(seed)
    tf_adata_viz = tf_adata_viz[np.random.permutation(tf_adata_viz.obs_names), :]

    viz_df = adata_dimviz_prediction(adata = tf_adata_viz, reduction_type = 'pca', 
                                     cats = ['seurat_annotations', 'batch', 'stim', 'condition'],
                                     max_condition_size = None if counterfactual_type == 'opposite' else 1000, 
                             seed = seed_split)
    viz_df.seurat_annotations = pd.Categorical(viz_df.seurat_annotations, 
                                              categories = cell_types, ordered=True)
    viz_df.stim = pd.Categorical(viz_df.stim, categories = ['CTRL', 'STIM'], ordered = True)



    for j, cell_type in enumerate(cell_types):
        for i, comb in enumerate(pc_combs):
    #         print(comb)
            viz_df_ = viz_df[viz_df.seurat_annotations == cell_type]

            stim_pred = viz_df_[viz_df_.batch == 'predicted'].stim.iloc[0]
            order = [stim_pred + '^' + cell_type + '^' + 'predicted', 
                    stim_pred + '^' + cell_type + '^' + 'actual', 
                    rev_stim[stim_pred] + '^' + cell_type + '^' + 'actual']
            viz_df_ = viz_df_.copy()
            order_values = ['predicted', 'test', 'train']
            order_map = dict(zip(order, order_values))
            viz_df_['condition'] = viz_df_.condition.map(order_map)

            viz_df_['condition'] = pd.Categorical(viz_df_.condition, 
                                                  categories = order_values, ordered=True)

            np.random.seed(seed)
            viz_df_ = viz_df_.loc[np.random.permutation(viz_df_.index),:]

            if not (i == 0 and j == 0):
                legend = False
            else:
                legend = True
            sns.scatterplot(data = viz_df_, 
                            x = 'PC{}'.format(comb[0]), 
                            y = 'PC{}'.format(comb[1]), 
                            hue = 'condition', 
                            style = 'stim', markers = marker_dict, s = 20,
                            ax = ax[i, j], legend = legend)


            if legend:
                handles, labels = ax[i, j].get_legend_handles_labels()
                labels[1:4] = ['predicted', 'test', 'train']
                ax[i, j].legend(handles=handles, labels=labels)

            ax[i,j].set_title(cell_type)

    # fig.text(0.5, title_coords[i], 
    #          counterfactual_type_title[counterfactual_type], 
    #          ha='center', va='center', fontsize=18, fontweight='bold')

    fig.tight_layout(rect=[0,0,1,0.9])


# # 8. CPA-like metrics
# 
# ## 8.1 Feature Correlations
# 
# Gene-wise mean/variance and compared to a random subset of the training data

# In[ ]:


if visualize:
    tf_res = io.read_pickled_object(os.path.join(data_path, 'trash', fn + '_predictions.pickle'))
    tf_adata_predicted = tf_res['none']

    res_all = list()
    for cond in tqdm(test_conds):
        res = pd.DataFrame(index = tf_adata.var_names, columns = ['actual_mean', 'predicted_mean', 'rand_mean',  
                                                           'actual_var', 'predicted_var', 'rand_var'])


        stim, ct = cond.split('^')
        tf_adata_ = tf_adata_predicted[(tf_adata_predicted.obs.condition == cond)].copy()

        # random baselin as in CPA 
        # random train subset is equal to maximum of the actual and predicted test condition
        tf_adata_rand = tf_adata_predicted[tf_adata_predicted.obs.condition.isin(train_conds) &
                          (tf_adata_predicted.obs.batch == 'actual')].copy()
        np.random.seed(seed_split)
        rand_obs = list(np.random.choice(tf_adata_rand.obs_names, size = max(tf_adata_.obs.batch.value_counts())))
        tf_adata_rand = tf_adata_rand[rand_obs, :].to_df()


        for batch_type in ['actual', 'predicted', 'rand']:
            if batch_type != 'rand':
                df = tf_adata_[tf_adata_.obs.batch == batch_type].to_df()
                res[batch_type + '_mean'] = df.mean().tolist()
                res[batch_type + '_var'] = df.var().tolist()
            else:
                res[batch_type + '_mean'] = tf_adata_rand.mean().tolist()
                res[batch_type + '_var'] = tf_adata_rand.var().tolist()
        res['test_condition'] = cond
        res.reset_index(names=['feature'], inplace = True)
        res_all.append(res)   
    res_all = pd.concat(res_all, axis=0, ignore_index=True)


    stats_df = pd.DataFrame(index = test_conds, columns = ['pearson_mean_actual', 'pearson_mean_rand', 
                                                          'pearson_var_actual', 'pearson_var_rand'])

    for cond in test_conds:
        cond_df = res_all[res_all.test_condition == cond]
        for metric_type in ['mean', 'var']:
            real_pearson = stats.pearsonr(cond_df['actual_' + metric_type],
                                          cond_df['predicted_' + metric_type]
                                         ).statistic
            rand_pearson = stats.pearsonr(cond_df['rand_' + metric_type],
                                          cond_df['predicted_' + metric_type]
                                         ).statistic

            stats_df.loc[cond, 'pearson_' + metric_type + '_actual'] = real_pearson
            stats_df.loc[cond, 'pearson_' + metric_type + '_rand'] = rand_pearson
    stats_df.reset_index(names = ['condition'], inplace = True)


# In[ ]:


if visualize:
    fig, ax = plt.subplots(ncols = 2, figsize = (7, 3))
    for i, metric_type in enumerate(['mean', 'var']):
        viz_df = stats_df[[col for col in stats_df.columns if metric_type in col]]
        viz_df = pd.melt(viz_df, var_name = 'comparison', value_name='pearson')
        viz_df.comparison = pd.Categorical(viz_df.comparison.apply(lambda x: x.split('_')[-1]), 
                                           ordered = True, 
                                           categories = ['actual', 'rand'])

        sns.boxplot(data = viz_df, x = 'comparison', y = 'pearson', ax = ax[i])
        ax[i].set_title(metric_type)

    fig.suptitle('Gene-Wise Comparison across OODs')
    fig.tight_layout()
    ("")


# In[ ]:


if visualize:
    n_total = len(test_conds)
    ncols = min(3, n_total)
    nrows = math.ceil(n_total / ncols)

    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5.1*ncols, 5.1*nrows))
    ax = np.array(ax).reshape(-1)

    for i, cond in enumerate(test_conds):
        viz_df = res_all[res_all.test_condition == cond]    
        sns.regplot(data=viz_df, x='predicted_mean', y='actual_mean', ax=ax[i])
        ax[i].set_title(cond)

        pearson_val = stats_df[stats_df.condition == cond]['pearson_mean_actual'].tolist()[0]
        ax[i].annotate('Pearson: {:.2f}'.format(pearson_val),
                       xy=(0.95, 0.05), xycoords='axes fraction',
                       ha='right', va='bottom', fontsize=10)

    fig.tight_layout()


# ## 8.2 Top Markers
# 
# CPA Fig. 3B and S3

# In[ ]:


from cliffs_delta import cliffs_delta
from statsmodels.stats.multitest import multipletests

from typing import Literal
def TF_de(df_A, 
          df_B, 
          fdr_thresh = 0.1, 
          effect_size_thresh: Literal['negligible', 'small', 'medium', 'large'] = 'small', 
         rank_by: Literal['fdr', 'effect_size'] = 'effect_size'):
    """Conduct the DE between two DFs. 

    Parameters
    ----------
    df_A : _type_
        _description_
    df_B : _type_
        _description_
    fdr_thresh : float, optional
        only retain DE TFs with FDRs less than or equal to this threshold, by default 0.1
        if None, will not threshold on FDR
    effect_size_thresh : Literal['negligible', 'small', 'medium', 'large'], optional
        only retain effect sizes greater than or equal to this category, by default 'small'
        effect size is measured with Cliff's Delta, which heuristically categorizes values in one of these four categories
    rank_by :  Literal['fdr', 'effect_size'], optional
        rank order TFs by false discovery rate (fdr) or effect size (effect_size), by default 'effect_size'
    """
    if fdr_thresh is None:
        fdr_thresh = 1
    pvals = stats.mannwhitneyu(df_A, df_B, alternative = 'two-sided', axis = 0).pvalue
#     _, pvals = stats.ttest_ind(df_A, df_B, alternative = 'two-sided', equal_var=False, axis = 0)
    _, fdrs, _, _ = multipletests(pvals, method='fdr_bh')

    cd = pd.DataFrame(columns = ['effect_size', 'effect'])
    for feature in df_A.columns:
        cd_val = cliffs_delta(df_A[feature], df_B[feature])
        cd.loc[feature, :] = cd_val
    cd['effect'] = pd.Categorical(cd.effect, ordered = True, 
                                        categories = ['negligible', 'small', 'medium', 'large'])
    cd['fdr'] = fdrs
    de = cd[cd['fdr'] <= fdr_thresh] 
    if effect_size_thresh is not None:
        de = de[de['effect'] >= effect_size_thresh]
    
    if rank_by == 'effect_size':
        de = de.sort_values(by='effect_size', key=abs, ascending = False)
    elif rank_by == 'fdr':
        de = de.sort_values(by = 'fdr', ascending = True)
    else:
        raise ValueError('Incorrect rank_by parameter specified')

    de_pos = de[de.effect_size >= 0]
    de_neg = de[de.effect_size < 0]
    
    return de_pos, de_neg, de


# Get the top 5 markers for each OOD:

# In[ ]:


if visualize: 
    tf_res = io.read_pickled_object(os.path.join(data_path, 'trash', fn + '_predictions.pickle'))
    tf_adata_predicted = tf_res['none']

    n_markers = 3
    de_res_pos = pd.DataFrame(columns = test_cell_types, index = range(n_markers))
    de_res_neg = de_res_pos.copy()
    for cond in test_conds:
        stim, ct = cond.split('^')

        # this is the same as using the actual tf_adata
        tf_adata_ = tf_adata_predicted[(tf_adata_predicted.obs.batch == 'actual') & 
                                      (tf_adata_predicted.obs.seurat_annotations == ct)].copy()
        stim_df = tf_adata_[tf_adata_.obs.stim == stim].to_df()
        ctrl_df = tf_adata_[tf_adata_.obs.stim == rev_stim[stim]].to_df()
        de_pos, de_neg, _ = TF_de(stim_df, ctrl_df, 
                             fdr_thresh = 0.1,
                             effect_size_thresh = 'small')

        de_res_pos[ct] = de_pos.index.tolist()[:n_markers]
        de_res_neg[ct] = de_neg.index.tolist()[:n_markers]


# In[ ]:


if visualize:
    ncols = n_markers
    nrows = len(test_conds)

    fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize = (5.1*ncols, 5.1*nrows))
    if nrows == 1:
        ax = np.expand_dims(ax, axis=0)

    de_res = de_res_pos.copy()
    for i, cond in enumerate(test_conds):
        stim, ct = cond.split('^')
        tf_adata_ = tf_adata_predicted[(tf_adata_predicted.obs.seurat_annotations == ct)].copy()

        top_markers = de_res[ct].tolist()

        tf_adata_ = tf_adata_[:, top_markers].copy()

        viz_df = tf_adata_.to_df()
        viz_df['condition'] = pd.Categorical(tf_adata_.obs.batch.astype(str) + '^' + tf_adata_.obs.stim.astype(str).tolist(), 
                                    ordered = True,
                                    categories = ['actual^' + rev_stim[stim], 
                                                  'actual^' + stim, 
                                                  'predicted^' + stim]
                                   )
        for j, marker in enumerate(top_markers):
            sns.violinplot(data = viz_df, y = marker, x = 'condition', ax = ax[i,j])
            ax[i,j].set_title(ct)

    fig.suptitle('Up-regulated TFs in OOD')
    fig.tight_layout()
    ("")


# In[ ]:


if visualize:
    ncols = n_markers
    nrows = len(test_conds)

    fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize = (5.1*ncols, 5.1*nrows))

    de_res = de_res_neg.copy()
    for i, cond in enumerate(test_conds):
        stim, ct = cond.split('^')
        tf_adata_ = tf_adata_predicted[(tf_adata_predicted.obs.seurat_annotations == ct)].copy()

        top_markers = de_res[ct].tolist()

        tf_adata_ = tf_adata_[:, top_markers].copy()

        viz_df = tf_adata_.to_df()
        viz_df['condition'] = pd.Categorical(tf_adata_.obs.batch.astype(str) + '^' + tf_adata_.obs.stim.astype(str).tolist(), 
                                    ordered = True,
                                    categories = ['actual^' + rev_stim[stim], 
                                                  'actual^' + stim, 
                                                  'predicted^' + stim]
                                   )
        for j, marker in enumerate(top_markers):
            sns.violinplot(data = viz_df, y = marker, x = 'condition', ax = ax[i,j])
            ax[i,j].set_title(ct)

    fig.suptitle('Down-regulated TFs in OOD')
    fig.tight_layout()
    ("")


# In[ ]:


clear_memory()


# In[19]:


import papermill as pm
from nbconvert import HTMLExporter
import nbformat
import os

input_notebook = 'visualize_scheduler.ipynb' # in the current directory
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


# Clear variables:

# In[ ]:


# Delete all user-defined variables
for name in dir():
    if not name.startswith("_"):
        if not name in ['In', 'Out', 'exit', 'get_ipython', 'open', 'quit']:
            del globals()[name]


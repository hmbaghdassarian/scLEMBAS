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

#python test_run.py --index 20v3 --run_type E --bn_weights_lambda_L2 1e-7 --uniform_lambda_L2 1e-7 --cat_max_norm 100 --global_bias_lambda_L2 0 --cat_bias_lambda_L2 1e-4 --vae_scaling_KL 1e-3 --global_bias_lambda_L1 0 --cat_bias_lambda_L1 0 --vae_prior_mu 0 --vae_prior_sigma 1 --adj_scaling_KL 0 --adj_prior_mu 0 --adj_prior_sigma 0.2 --loss_type MSE --per_condition_loss true --cat_max_penalty_weight 12 --cat_b_adv 2 --pert_max_penalty_weight 8 --pert_b_adv 3.5 --network_noise_scale 0.01 --min_network_noise 0.0025 --include_gradient_noise_vae true --include_gradient_noise_embedding true --constant_gradient_noise true --gradient_noise_scale 1e-9 --lr_period 4 --reset_state false --train_batch 500 --initialize_fc true --generator_dropout_rate 0.7 --cat_discriminator_dropout_rate 0.1 --pert_discriminator_dropout_rate 0.1 --discriminator_batch_momentum 0 --spectral_norm false --discriminator_lambda_L2 1e-3 --discriminator_bionet_activation false --smooth_labels true --gradient_ascent true --n_adversarial_start 200 --n_discriminator_train 5 --vae_lambda_l2 1e-5 --min_cat_adv_penalty 0.1 --min_pert_adv_penalty 0.1 --main_max_lr 1e-3 --generator_max_lr 5e-4 --cat_max_lr 1e-3 --pert_max_lr 1e-3 --lr_decay 0.9 --cat_bias_orthogonality_scaler 100 --cp_method orthogonality --cp_include_adjacency false --cp_per_label false


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
# main_max_lr = 1e-3
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


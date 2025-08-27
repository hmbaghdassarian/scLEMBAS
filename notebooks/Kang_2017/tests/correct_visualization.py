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


########################################################################
args = parser.parse_args()
fn = str(args.index)
run_type = 'E'
fn += run_type
# seed = args.seed
# loo = args.loo

#python test_run.py --index 2v2 --run_type E --bn_weights_lambda_L2 1e-7 --uniform_lambda_L2 1e-7 --cat_max_norm 100 --global_bias_lambda_L2 0 --cat_bias_lambda_L2 1e-4 --vae_scaling_KL 1e-3 --global_bias_lambda_L1 0 --cat_bias_lambda_L1 0 --vae_prior_mu 0 --vae_prior_sigma 1 --adj_scaling_KL 0 --adj_prior_mu 0 --adj_prior_sigma 0.2 --loss_type MSE --per_condition_loss true --cat_bias_orthogonality_scaler 100 --cat_max_penalty_weight 12 --cat_b_adv 2 --pert_max_penalty_weight 8 --pert_b_adv 3.5 --network_noise_scale 0.01 --min_network_noise 0.0025 --include_gradient_noise_vae true --include_gradient_noise_embedding true --constant_gradient_noise true --gradient_noise_scale 1e-9 --lr_period 4 --reset_state false --train_batch 500 --initialize_fc true --generator_dropout_rate 0.7 --discriminator_dropout_rate 0.3 --discriminator_batch_momentum 0 --spectral_norm false --discriminator_lambda_L2 1e-3 --discriminator_bionet_activation false --smooth_labels true --gradient_ascent true --n_adversarial_start 200 --n_discriminator_train 5 --vae_lambda_l2 1e-5 --min_cat_adv_penalty 0.1 --min_pert_adv_penalty 0.1 --main_max_lr 1e-3 --generator_max_lr 1e-3 --cat_max_lr 1e-3 --pert_max_lr 1e-3 --lr_decay 0.9


# 

# In[15]:


# fn = 'dev'#'trash'
# run_type = 'E'
# fn += run_type
# # seed = 3

# #loo = False

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


# main_max_lr = 1e-3 
# generator_max_lr = 1e-3 
# cat_max_lr = 1e-3 
# pert_max_l =r 1e-3


# In[16]:


run_types = {'A': (1, True),
            'B': (3, True), 
            'C': (7, True), 
            'D': (10, True), 
            'E': (1, False)
            }
seed, loo = run_types[run_type]




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


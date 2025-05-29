#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


#python test_MAIN.py --index dev --run_type E --bn_weights_lambda_L2 1e-7 --uniform_lambda_L2 1e-7 --cat_max_norm 100 --global_bias_lambda_L2 0 --cat_bias_lambda_L2 0 --vae_scaling_KL 1e-2 --global_bias_lambda_L1 0 --cat_bias_lambda_L1 0 --vae_prior_mu 0 --vae_prior_sigma 1 --adj_scaling_KL 0 --adj_prior_mu 0 --adj_prior_sigma 0.2 --loss_type MSE --per_condition_loss true --cat_bias_orthogonality_scaler 0 --cat_max_penalty_weight 8 --cat_b_adv 1.5 --pert_max_penalty_weight 20 --pert_b_adv 2 --network_noise_scale 0.01 --min_network_noise 0.0025 --include_gradient_noise_vae true --include_gradient_noise_embedding true --constant_gradient_noise true --gradient_noise_scale 1e-9 --lr_period 4 --reset_state true


# 

# In[3]:


run_types = {'A': (1, True),
            'B': (3, True), 
            'C': (7, True), 
            'D': (10, True), 
            'E': (1, False)
            }
seed, loo = run_types[run_type]


# In[4]:


visualize = False
mod_type = 'default'
subset = False
short_run = False
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


# In[5]:


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


# In[6]:


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


# In[7]:


import sys
sclembas = '/home/hmbaghda/Projects/scLEMBAS'
sys.path.insert(1, os.path.join(sclembas))
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

sys.path.insert(1, '/home/hmbaghda/Projects/scLEMBAS/notebooks/Kang_2017/')
from Kang_utils import (rev_stim, stim_map, rev_stim_map, adata_dimviz_bias, clear_memory,
                        get_prediction, adata_dimviz_prediction, prepare_for_metrics, get_loss)


# In[8]:


n_cores = 12
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)

data_path = '/nobackup/users/hmbaghda/scLEMBAS/analysis'
device = "cuda" if torch.cuda.is_available() else "cpu"


# In[9]:


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


# # 1. Create a novel train-test split:

# # Checkpoint: load the object

# In[10]:


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


# In[ ]:


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
        
for col in ['pert_discriminator_learning_rate', 'pert_discriminator_loss_total',
           'pert_discriminator_loss_prediction', 'pert_discriminator_param_reg_loss']:
    if col not in train_stats_df.columns:
        train_stats_df[col] = 0
        
# if 'test' in trainer.stats:
#     test_stats_df = trainer.stats['test'].copy()
#     test_stats_df = test_stats_df.groupby('epoch').mean().reset_index() # DELETE THIS


# # 4. Look at the loss curves:

# In[155]:


cf_map = {'in_distribution': train_cells}
counterfactual_types = list(cf_map.keys()) + ['opposite']


# ## 5.1 Test Biases

# In[156]:





from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# set up the outputs to classify
pred_types = {}
# global bias output from train data only (no counterfactual)
global_bias = io.read_pickled_object(os.path.join(data_path, 'trash', fn + '_feb_clustered_biases_train.pickle'))
global_bias = global_bias['opposite'][0]
pred_types['train_generator'] = global_bias

# forward pass with no ProjectOutput
# global bias Y_full
global_bias = get_prediction(mod = mod,
                              train_cells = train_cells,
                              tf_adata = tf_adata,
                              train_mode = True,
                              counterfactual = False,
                              remove_type = ['adj', 'categorical_bias'],
                              return_bias = False,
                                    max_cells = 2000, 
                     return_full = True)
pred_types['train_bionet'] = global_bias

# full forward pass on train data only (no counterfactual) using only the global bias
tf_res_train = io.read_pickled_object(os.path.join(data_path, 'trash', fn + '_predictions_train.pickle'))
global_bias = tf_res_train['adj_categorical_bias']
global_bias = global_bias[global_bias.obs.batch == 'predicted',:].copy()
pred_types['train_fullforward'] = global_bias

# global bias output from test  (counterfactuals, meaning input gene expression is still in train)
global_bias = io.read_pickled_object(os.path.join(data_path, 'trash', fn + '_feb_clustered_biases.pickle'))
global_bias = global_bias['opposite'][0]
pred_types['test'] = global_bias


probe_res = pd.DataFrame(columns=['category_name', 'accuracy', 'prediction_type'])
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

            probe = LogisticRegression(max_iter=2000, solver='lbfgs', multi_class='auto', random_state=seed_split)
            probe.fit(X_train, y_train)

            y_pred = probe.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            probe_res.loc[counter, 'category_name'] = category_name
            probe_res.loc[counter, 'accuracy'] = accuracy
            probe_res.loc[counter, 'prediction_type'] = pred_type
            counter += 1

probe_res = probe_res.pivot_table(
    index='prediction_type',
    columns='category_name',
    values='accuracy',
    aggfunc=list  # or use list/np.array if you want to keep all fold values
).reset_index().explode(['cell_type', 'perturbation']).reset_index(drop=True)
probe_res.columns.name = None
probe_res.to_csv(os.path.join(data_path, 'trash', fn + '_probe.csv'))

clear_memory()


# In[ ]:





# In[19]:


import papermill as pm
from nbconvert import HTMLExporter
import nbformat
import os

input_notebook = 'visualize.ipynb' # in the current directory
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


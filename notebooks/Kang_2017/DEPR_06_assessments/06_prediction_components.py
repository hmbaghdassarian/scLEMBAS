#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

from tqdm import tqdm
import json
import pandas as pd
import scanpy as sc
import numpy as np

import torch

import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(1, '../.')
from Kang_utils import get_prediction, prepare_for_metrics, get_best_hyperparams, adata_dimviz_bias 


# In[2]:


sclembas = '/home/hmbaghda/Projects/scLEMBAS'
sys.path.insert(1, os.path.join(sclembas))
from scLEMBAS import io
# from scLEMBAS.preprocess import embed_tf_activity


# In[3]:


seed = 888
device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = '/nobackup/users/hmbaghda/scLEMBAS/analysis'

author = 'Kang'


# In[4]:


n_cores = 20
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)


# Load data:

# In[5]:


tf_adata = io.read_tfad(os.path.join(data_path, 'processed', 'Kang_tf_activity.h5ad'))
tf_adata.obs['condition'] = tf_adata.obs['stim'].astype(str) + '^' + tf_adata.obs['seurat_annotations'].astype(str)

adata = sc.read_h5ad(os.path.join(data_path, 'processed', 'kang_expr_scored.h5ad'))
sn_ppis = pd.read_csv(os.path.join(data_path, 'processed', 'Kang_sn_ppis.csv'), index_col = 0)


# In[6]:


source_label = 'source_genesymbol'
target_label = 'target_genesymbol'
weight_label = 'mode_of_action'
stimulation_label = 'consensus_stimulation'
inhibition_label = 'consensus_inhibition'


# In[7]:


trainer = io.read_pickled_object(os.path.join(data_path, 'processed', 'Kang_fullbest_trainer.pickle'))
mod = trainer.mod

test_cells = open(os.path.join(data_path, 'processed', 'data_split_barcodes', 'kang_test.txt')).read().splitlines()
test_conds = sorted(tf_adata.obs.loc[test_cells, 'condition'].unique())
train_cells_all = [barcode for barcode in tf_adata.obs_names if barcode not in test_cells]

# test_cells = trainer.X_test.index.tolist()
# train_cells_all = trainer.X_train.index.tolist()
# test_conds = sorted(tf_adata.obs.loc[test_cells, 'condition'].unique())


# In[8]:


cf_map = {'in_distribution': train_cells_all}
counterfactual_types = list(cf_map.keys()) + ['opposite']


# # 0. Parameters

# In[9]:


calculation_type = 'project' # project data rather than embed
n_neighbors = 15 # clustering as before
run_umap = True # also get the prediction umaps in addition to pca
best_resolution = tf_adata.uns['leiden']['params']['resolution']


# # 1. Predictions

# In[9]:


mod.signaling_network.vae.seed = seed


# In[11]:








# ## 1.2 Let's also get the losses on the individual fold models:

# In[10]:


models_path = os.path.join(data_path, 'processed', 'models')

res_folds = pd.read_csv(os.path.join(data_path, 'processed', 'Kang_k_fold_validation_results.csv'), index_col = 0)
best_emd_mean, best_hyperparams, best_emd = get_best_hyperparams(res_folds)
trainers_best = {k: io.read_pickled_object(os.path.join(models_path, 'Kang_best_trainer_' + str(k) + '.pickle')) for k in range(best_emd.shape[0])}


# In[12]:


loss_res_fold = {}

for k, trainer_k in tqdm(trainers_best.items()):
    mod_k = trainer_k.mod
    mod_k.signaling_network.vae.seed = seed # REMOVE
    train_cells_all_k = trainer_k.X_train.index.tolist()
    cf_map_k = {'in_distribution': train_cells_all_k}
    
    loss_res_fold[k] = {}
    
    for counterfactual_type in counterfactual_types:#['opposite']:
        loss_res_fold[k][counterfactual_type] = {}
        for remove_type in ['none', 'global_bias', 'categorical_bias', 'total_bias', 'adj']:
            tf_adata_predicted, tot_loss = get_prediction(mod = mod_k, 
                                                          tf_adata = tf_adata, 
                                                          counterfactual_type = counterfactual_type, 
                                                          cf_map = cf_map_k, 
                                                          train_cells_all = train_cells_all_k, 
                                                          test_conds = test_conds, 
                                                          remove_type = remove_type,
                                                          return_bias = False, 
                                                          return_loss = True, 
                                                         test_cells = test_cells) 
            loss_res_fold[k][counterfactual_type][remove_type] = tot_loss
            
            
loss_res_fold_formatted = dict(zip(['in_distribution', 'opposite'], [[], []]))
for k, inner_dict in loss_res_fold.items():
    final_out = {'fold': k}
    for counterfactual_type, inner_dict_2 in inner_dict.items():
        for include_type, val in inner_dict_2.items():
            final_out[include_type] = val
        loss_res_fold_formatted[counterfactual_type].append(final_out)
        
with open(os.path.join(data_path, 'processed', author + '_prediction_loss_folds.json'), 'w') as json_file:
    json.dump(loss_res_fold_formatted, json_file, indent=4)

#!/usr/bin/env python
# coding: utf-8

# Here, we ask how well the global distribution of our predicted data matches our real data. To do so, we will draw on metrics from batch correction benchmarking studies, namely [here](https://doi.org/10.1038/s41587-020-00748-9). This study presents two key metrics:
# 
# 1. Clusterability: this shows how well biologically distinct groups separate in the data. We will use the NMI metric to quantify clusterability. 
# 2. Mixability: this shows how well biologically similar groups align in the data. With regards to batch correction, this typically asks how well the same cell type from different batches clusters together. In our case, we use it to see how well predicted and actual data group; so predicted and actual data are analogous to batches. To quantify mixability, we use the modified alignment score. The original score was developed [here](https://doi.org/10.1038/nbt.4096) and the modified version can be found [here](https://doi.org/10.1038/s41587-020-00748-9).
# 
# Notes:
# - While the batch correction metrics use this for cell type, we use it for the combination of cell type and stimulation condition. 
# - In this case, our "batch" is our predicted vs actual data. 
# 
# Baselines: We need some baseline metrics to compare our predicted values to. For each baseline, we first duplicate the values of our actual test/OOD cells and concatenate them to our actual data but labeled as "predicted". 
# 
# As the number of cells in each condition can effect the metrics and the model is generative such that this can vary from counterfactual to counterfactual, or fold to fold, we generate a different baseline for each instance. The number of values we duplicate per condition matches that of the specific counterfactual (the specific prediction). For each condition: If there are a fewer number of cells in the prediction than the actual data, we take a random subset without replacement to match it. If there are more cells in the prediction than the actual data, we randomly choose with replacement.  
# 
# 1. baseline NMI: NMI of the concatenated dataset
# 2. Alignment score: Here, we have two baselines based on two expectations. Expectation 1: our actual and predicted OOD will have high alignment, as they represent the same conditions. Expectation 2: our actual train data and predicted OOD data will have low alignment, as they represent different conditions. Thus, we calculate the alignment score between actual and predicted OOD of the concatenated, and between in-distribution actual and OOD predicted. 

# In[15]:


import os
from typing import Literal
from typing import List

import pandas as pd
import scanpy as sc
import numpy as np

import torch

from sklearn.metrics import normalized_mutual_info_score
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


import sys
sclembas = '/home/hmbaghda/Projects/scLEMBAS'
sys.path.insert(1, os.path.join(sclembas))
from scLEMBAS import io
from scLEMBAS.preprocess import embed_tf_activity, get_alignment_score


# In[17]:


seed = 888
device = "cuda" if torch.cuda.is_available() else "cpu"
author = 'Kang'
data_path = '/nobackup/users/hmbaghda/scLEMBAS/analysis'


# In[18]:


# params
calculation_type = 'project' # project data rather than embed
n_neighbors = 15


# In[19]:


n_cores = 20
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)


# Load the model and associated data:

# In[20]:


tf_adata = io.read_tfad(os.path.join(data_path, 'processed', 'Kang_tf_activity.h5ad'))
tf_adata.obs['condition'] = tf_adata.obs['stim'].astype(str) + '^' + tf_adata.obs['seurat_annotations'].astype(str)

adata = sc.read_h5ad(os.path.join(data_path, 'processed', 'kang_expr_scored.h5ad'))
sn_ppis = pd.read_csv(os.path.join(data_path, 'processed', 'Kang_sn_ppis.csv'), index_col = 0)

best_resolution = tf_adata.uns['leiden']['params']['resolution']


# In[21]:


source_label = 'source_genesymbol'
target_label = 'target_genesymbol'
weight_label = 'mode_of_action'
stimulation_label = 'consensus_stimulation'
inhibition_label = 'consensus_inhibition'


# In[22]:


trainer = io.read_pickled_object(os.path.join(data_path, 'processed', 'Kang_fullbest_trainer.pickle'))
mod = trainer.mod

test_cells = open(os.path.join(data_path, 'processed', 'data_split_barcodes', 'kang_test.txt')).read().splitlines()
test_conds = sorted(tf_adata.obs.loc[test_cells, 'condition'].unique())
train_cells_all = [barcode for barcode in tf_adata.obs_names if barcode not in test_cells]

test_cells = trainer.X_test.index.tolist()
train_cells_all = trainer.X_train.index.tolist()
test_conds = sorted(tf_adata.obs.loc[test_cells, 'condition'].unique())


# #### Get the predictions:
# 
# Given the flexibility of the counterfactual, we can make the predictions of the OOD cells from a number of gene expression inputs. Specifically, we can predict from the following gene expression inputs:
# - in-distribution: all train cells
# - opposite: for each test condition, we predict from the same cell type but opposite stimulation condition (these are all in-distribution as well)
# 
# Additional options not explored:
# - OOD: test cells only
# - all: all cells (in-distribution + OOD)
# - stimulated: in-distribution stimulated cells
# - unstimulated: in-distribution control cells
# - cell type: prediction from a specific in-distribution cell type
# - condition: prediction from a specific in-distribution cell type and stimulation

# In[23]:


rev_stim = {'STIM': 'CTRL', 'CTRL': 'STIM'}

stim_map = {'STIM': 1, 'CTRL': 0}
rev_stim_map = {v:k for k,v in stim_map.items()}


# In[24]:


def get_prediction(mod, tf_adata, counterfactual_type, cf_map):
    """Gets and formats the model predictions from the counterfactual"""
    cov_idx_map = dict(zip(mod.signaling_network.covariates['seurat_annotations'], 
                           mod.signaling_network.covariates_idx['seurat_annotations']))
    cov_rev_map = {v:k for k,v in cov_idx_map.items()}
    
    
    full_expr, full_X, full_covariates = None, None, None

    for cond in test_conds:
        stim, ct = cond.split('^')

        if counterfactual_type == 'opposite':
            train_cells_cond = tf_adata.obs[(tf_adata.obs['condition'] == rev_stim[stim] + '^' + ct)].index.tolist()
            if len(set(train_cells_cond).difference(train_cells_all)) != 0:
                raise ValueError('Something went wrong in the counterfactual')
        else:
            train_cells_cond = cf_map[counterfactual_type]

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
        
    obs = pd.DataFrame(full_covariates.detach().cpu().numpy())
    obs.columns = ['seurat_annotations']
    obs.seurat_annotations = obs.seurat_annotations.map(cov_rev_map)
    obs['stim'] = pd.Series(full_X.detach().cpu().numpy().reshape(-1)).map(rev_stim_map)
    obs['condition'] = obs['stim'].astype(str) + '^' + obs['seurat_annotations'].astype(str)
    
    y_predicted = pd.DataFrame(y_predicted.detach().cpu().numpy())
    y_predicted.columns = mod.y_out.columns
    tf_adata_predicted = sc.AnnData(X = y_predicted, obs = obs)
    
    return tf_adata_predicted

def make_baseline(tf_adata, tf_adata_predicted):
    """Makes baseline model based on # of cells in each condition of the predictd values"""
    barcodes = []
    for cond, n_cells in dict(tf_adata_predicted.obs.condition.value_counts()).items():
        actual_cells = tf_adata.obs[tf_adata.obs.condition == cond].index.tolist()
        if len(actual_cells) >= n_cells:
            np.random.seed(seed)
            actual_cells = np.random.choice(actual_cells, n_cells, replace = False)
        else:
            actual_cells = np.random.choice(actual_cells, n_cells, replace = True)
        barcodes += list(actual_cells)

    tf_adata_base = sc.AnnData(X = tf_adata.to_df().loc[barcodes,:],
                               obs = tf_adata.obs.loc[barcodes, :])
    tf_adata_base.obs_names_make_unique()
    
    return tf_adata_base


def prepare_for_metrics(tf_adata, 
                tf_adata_predicted, 
                resolution,
                calculation_type: Literal['embed', 'project'] = 'project',
                n_neighbors: int = 15
                       ):
    """Combine predictions with actual values and then recalculate neighbors/clusters. 
    Project will project predictions to space calculated on actual values
    Embed will jointly embed the actual and predicted values."""
    

    tf_adata_actual = tf_adata.copy()
    tf_adata_actual.obs['batch'] = 'actual'

    tf_adata_predicted = tf_adata_predicted.copy()
    tf_adata_predicted.obs['batch'] = 'predicted'
    
    tf_adata_ = sc.concat([tf_adata_actual, tf_adata_predicted])
    tf_adata_.obs['barcode'] = tf_adata_.obs.index.tolist()

    if len(set(tf_adata_.obs_names)) < len(tf_adata_.obs_names):
        tf_adata_.obs_names_make_unique()
    
    if calculation_type == 'project': # project the predicted data into the actual data space
        pc_rank = tf_adata.uns["pca"]['pca_rank']
        pca_mod = tf_adata.uns['pca']['pca_mod']
        tf_adata_.obsm['X_pca'] = pca_mod.transform(tf_adata_.to_df().values)
        tf_adata_.uns['pca'] = tf_adata.uns['pca'].copy()
        
        embed_tf_activity(tf_adata = tf_adata_,
                          scanpy_pca = None,
                          cluster_col_name = 'new_TF_clusters',
                          n_components = None,
                          pc_rank = None,
                          resolution = resolution,
                          n_neighbors = n_neighbors,
                          nmi_label = None,
                          run_pca = False, 
                          run_umap = False, 
                          cluster_data = True)

    elif calculation_type == 'embed': # embed the combined data again
        embed_tf_activity(tf_adata = tf_adata_,
                          scanpy_pca = True,
                          cluster_col_name = 'new_TF_clusters',
                          n_components = 50,
                          pc_rank = 'automate',
                          resolution = resolution,
                          n_neighbors = n_neighbors,
                          nmi_label = None,
                          run_pca = True, 
                          run_umap = False, 
                         cluster_data = True)
    return tf_adata_

def get_metrics(tf_adata_, train_cells, n_neighbors = 15):
    md = tf_adata_.obs.copy()
    
    # alignment between OOD actual and prediction
    as_cells = md[md.condition.isin(test_conds)].index.tolist()
    
    # sanity checks
    train_md = md[md.barcode.isin(train_cells)]
    if sorted(train_cells) != sorted(train_md.index.tolist()):
        raise ValueError('TF adata object contains different train cells than input')
    if train_md.batch.unique().tolist() != ['actual']:
        raise ValueError('TF adata object contains train cells annotated as predicted')
    if len(set(as_cells).intersection(train_cells)) != 0:
        raise ValueError('OOD cells intersect with train cells')
    
    nmi = normalized_mutual_info_score(tf_adata_.obs.condition, tf_adata_.obs.new_TF_clusters)
    as_OOD = get_alignment_score(adata = tf_adata_[as_cells, :], 
                       batch_key = 'batch', 
                       k = n_neighbors, 
                       normalize = True)

    # alignment between actual train and predicted OOD
    as_cells = md[md.barcode.isin(train_cells)].index.tolist() + md[(md.condition.isin(test_conds)) & (md.batch == 'predicted')].index.tolist()
    as_ID = get_alignment_score(adata = tf_adata_[as_cells, :], 
                       batch_key = 'batch', 
                       k = n_neighbors, 
                       normalize = True)
    
    return nmi, as_OOD, as_ID


# ##### Combined model

# In[25]:


cf_map = {'in_distribution': train_cells_all}
counterfactual_types = list(cf_map.keys()) + ['opposite']


# In[26]:


res = pd.DataFrame(columns = ['calculation_type', 'counterfactual_type', 
                              'nmi', 'as_OOD', 'as_ID', 
                             'baseline_nmi', 'baseline_as_OOD', 'baseline_as_ID'])

# for calculation_type in ['project', 'embed']:

tf_res = {}
for counterfactual_type in counterfactual_types:
    tf_adata_predicted = get_prediction(mod = mod, 
                                        tf_adata = tf_adata, 
                                        counterfactual_type = counterfactual_type, 
                                        cf_map = cf_map)
    tf_adata_base = make_baseline(tf_adata, tf_adata_predicted) # emulate predicted values, but with actual test values
    
    tf_adata_predicted = prepare_for_metrics(tf_adata, 
                                       tf_adata_predicted, 
                                       resolution = best_resolution,
                                       calculation_type = calculation_type, 
                                      n_neighbors = n_neighbors)
    nmi, as_OOD, as_ID = get_metrics(tf_adata_predicted, 
                                     train_cells = train_cells_all, 
                                     n_neighbors = n_neighbors)
    
    tf_adata_base = prepare_for_metrics(tf_adata, 
                                   tf_adata_base, 
                                   resolution = best_resolution,
                                   calculation_type = calculation_type, 
                                  n_neighbors = n_neighbors)
    baseline_nmi, baseline_as_OOD, baseline_as_ID = get_metrics(tf_adata_base, 
                                                                train_cells = train_cells_all, 
                                                                n_neighbors = n_neighbors)
    
    
    tf_res[counterfactual_type] = tf_adata_predicted
    
    
    res.loc[res.shape[0], :] = [calculation_type, counterfactual_type, 
                                nmi, as_OOD, as_ID, 
                               baseline_nmi, baseline_as_OOD, baseline_as_ID]
res.to_csv(os.path.join(data_path, 'processed', author + '_all_clusterability_mixability.csv'))
io.write_pickled_object(tf_res, os.path.join(data_path, 'interim', author + '_clusterability_ojects.pickle'))
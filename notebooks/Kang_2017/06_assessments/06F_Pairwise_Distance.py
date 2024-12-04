#!/usr/bin/env python
# coding: utf-8

# Clusterability and mixability looked at the global distribution of cells in PC space. EMD loss compared to baseline models similarly looked at the accuracy of the predicted global distribution. Here, we like pairwise distances between single-cells in the full TF activity space. We use euclidean distance as our distance metric.  
# 
# We have two expectations:
# 1. Distances within a cell type + stimulation condition are smaller than distances within a cell type across stimulations. 
# 2. Distances within a cell type + stimulation condition are smaller than a null distribution of distances between a random subset of cells of the same size. 
# 
# For each cell type, we will compared the predicted values of a given stimulation  to the actual values of the opposite stimulation. 
# 
# We will also need to set up baselines which use the actual data of the same condition as the predicted, to assess that our expectations are in fact met and how closely the predicted values compare.  

# In[20]:


import os
import itertools
import json

from tqdm import tqdm 

import pandas as pd
import scanpy as sc
import numpy as np

import torch

from scipy import stats
from statsmodels.stats.multitest import multipletests
from cliffs_delta import cliffs_delta

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import seaborn as sns


# In[2]:


import sys
sclembas = '/home/hmbaghda/Projects/scLEMBAS'
sys.path.insert(1, os.path.join(sclembas))
from scLEMBAS import io
from scLEMBAS.preprocess import calculate_pairwise_distances, get_upper_triangle, quantify_cluster_distance

sys.path.insert(1, '../.')
from Kang_utils import get_prediction, rev_stim


# In[3]:


seed = 888
device = "cuda" if torch.cuda.is_available() else "cpu"

author = 'Kang'
data_path = '/nobackup/users/hmbaghda/scLEMBAS/analysis'


# In[4]:


n_cores = 30
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)


# Params:

# In[5]:


distance_metric = 'euclidean'


# Load data:

# In[6]:


tf_adata = io.read_tfad(os.path.join(data_path, 'processed', 'Kang_tf_activity.h5ad'))
tf_adata.obs['condition'] = tf_adata.obs['stim'].astype(str) + '^' + tf_adata.obs['seurat_annotations'].astype(str)

adata = sc.read_h5ad(os.path.join(data_path, 'processed', 'kang_expr_scored.h5ad'))
sn_ppis = pd.read_csv(os.path.join(data_path, 'processed', 'Kang_sn_ppis.csv'), index_col = 0)


# In[7]:


source_label = 'source_genesymbol'
target_label = 'target_genesymbol'
weight_label = 'mode_of_action'
stimulation_label = 'consensus_stimulation'
inhibition_label = 'consensus_inhibition'


# In[8]:


trainer = io.read_pickled_object(os.path.join(data_path, 'processed', 'Kang_fullbest_trainer.pickle'))
mod = trainer.mod

test_cells = open(os.path.join(data_path, 'processed', 'data_split_barcodes', 'kang_test.txt')).read().splitlines()
test_conds = sorted(tf_adata.obs.loc[test_cells, 'condition'].unique())
train_cells_all = [barcode for barcode in tf_adata.obs_names if barcode not in test_cells]

# test_cells = trainer.X_test.index.tolist()
# train_cells_all = trainer.X_train.index.tolist()
# test_conds = sorted(tf_adata.obs.loc[test_cells, 'condition'].unique())


# #### Get the predictions and concordance:
# 
# As described in Notebook 06E, for each cell type, we get the counterfactual from all in-distribution or within a cell type ("opposite").

# In[13]:


cf_map = {'in_distribution': train_cells_all}
counterfactual_types = list(cf_map.keys()) + ['opposite']

tf_predicted_res = {}
for counterfactual_type in counterfactual_types:
    tf_adata_predicted = get_prediction(mod = mod, 
                                        tf_adata = tf_adata, 
                                        counterfactual_type = counterfactual_type, 
                                        cf_map = cf_map, 
                                        train_cells_all = train_cells_all, 
                                        test_conds = test_conds
                                        )
    tf_predicted_res[counterfactual_type] = tf_adata_predicted



# #### Expectation 2: Within stimulation distances < random distances

# For conditions with greater than 2000 cells (occurs for the in-distribution predictions), we randomly subset to 2000 cells for computation time

# In[74]:


comp_subset_size = int(2e3)


# In[54]:


tf_adata = io.read_tfad(os.path.join(data_path, 'processed', 'Kang_tf_activity.h5ad'))
tf_adata.obs['condition'] = tf_adata.obs['stim'].astype(str) + '^' + tf_adata.obs['seurat_annotations'].astype(str)


# In[12]:


for counterfactual_type in ['opposite', 'in_distribution']:
    tf_adata_predicted = tf_predicted_res[counterfactual_type]
    print(counterfactual_type)
    tf_adata_predicted.obs['batch'] = 'predicted'
    tf_adata_actual = tf_adata.copy()
    tf_adata_actual.obs['batch'] = 'actual'
    tf_adata_all = sc.concat([tf_adata_predicted, tf_adata_actual])
    tf_adata_all.obs['condition'] = tf_adata_all.obs['batch'].astype(str) + '&' + tf_adata_all.obs['condition'].astype(str)

    label = 'condition'
    comparison_combination_subset = []
    for comp in itertools.combinations_with_replacement(tf_adata_all.obs['condition'].unique(), 2):
        if comp[0].split('&')[1] and comp[1].split('&')[1] in test_conds:
            if comp[0] == comp[1]:
                comparison_combination_subset.append(comp)
    
    # subset for reasonable computation time
    included_conds = list(set(itertools.chain.from_iterable(comparison_combination_subset)))
    vc = tf_adata_all.obs.condition.value_counts().loc[included_conds]
    drop_cells = []
    for ic in vc[vc > comp_subset_size].index.tolist():
        all_cells = tf_adata_all.obs[tf_adata_all.obs.condition == ic].index.tolist()
        np.random.seed(seed)
        drop_cells += list(np.random.choice(all_cells, size = vc.loc[ic] - comp_subset_size, replace = False))
    tf_adata_all = tf_adata_all[~tf_adata_all.obs_names.isin(drop_cells)]

    predicted_cells = tf_adata_all.obs[tf_adata_all.obs.batch == 'predicted'].index.tolist() 



    distances_df = quantify_cluster_distance(tf_adata = tf_adata_all,
                                     label = 'condition', 
                                     include_self = True,
                                     comparison_combination_subset = comparison_combination_subset,
                                     alternative = 'less',
                                     exclude_null_cells = predicted_cells, 
                                     distance_metric = distance_metric,
                                     n_cores = min(len(comparison_combination_subset), n_cores),
                                     seed = seed,
                                     n_perm = 1000,  

                                     comparison_subset = None,
                                     label_subset = None,
                                     feature_subset = None,
                                     normal = True,
                                     use_pcs = False,
                                     rank = None 
                         )
    distances_df.to_csv(os.path.join(data_path, 'interim', author + '_' + counterfactual_type + '_null_distances.csv'))



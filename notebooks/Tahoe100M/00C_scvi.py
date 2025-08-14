#!/usr/bin/env python
# coding: utf-8

# Here, we apply normalization/dimensionality reduction/batch correections to the counts matrix:

# In[1]:


import time
import os

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.decomposition import PCA

import scanpy as sc

import matplotlib.pyplot as plt
import seaborn as sns

import sys
sclembas_path = '/home/hmbaghda/Projects/scLEMBAS'
sys.path.insert(1, os.path.join(sclembas_path))
from scLEMBAS import io
from scLEMBAS import preprocess as pp
import scLEMBAS.utilities as utils


# In[2]:


n_cores = 30
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)


# In[3]:


data_path = '/nobackup/users/hmbaghda/scLEMBAS/analysis'
author = 'Tahoe100M'
seed = 888


# Load the raw counts matrix (already filtered for low QC cells):

# ## Covariate Associations
# 
# Use the first n_pcs to understand which covariates are most associated with PC variance:

# In[43]:


adata = sc.read_h5ad(os.path.join(data_path, 'interim', author + '_normalized_counts.h5ad'))


# In[44]:


adata.shape


# ### Cell Cycle Deep Dive
# 
# Looks like drug is not a particularly strong source of variance. We will correct for these various other variance sources to capture those that we care about. First, let's look at the relationship between "phase", "G2M_score", and "S_score" since these are all indicators of cell cycle. If they're all closely associated, we probably only need to correct for one.

# Let's use a PC component that captures all 3 metrics:

# In[45]:


phase_dummies = pd.get_dummies(adata.obs['phase'], prefix='phase').astype(int)
X = pd.concat([adata.obs[['S_score', 'G2M_score']], phase_dummies], axis=1)

pca_mod = PCA(n_components=1, random_state = seed)
pca_mod.fit(X) # fit to all 3 metrics

var_explained_pc1 = pca_mod.explained_variance_ratio_[0]
print(f"Variance explained by PC1 of aggregated cell cycle metrics: {var_explained_pc1:.3%}")

cell_cycle_pc1 = pca_mod.transform(X)
adata.obs.loc[X.index, 'cell_cycle_PC1'] = cell_cycle_pc1[:, 0]


# Looks like the PC well accounts for all 3 cell cycle metrics, atleast linearly. So, we will use this as our cell cycle covariate.

# ## Covariate correction

# In[46]:


# adata = sc.read_h5ad(os.path.join(data_path, 'interim', author + '_normalized_counts.h5ad'))
# adata.obs.loc[X.index, 'cell_cycle_PC1'] = cell_cycle_pc1[:, 0]


# Nonlinear:

# In[4]:


import scvi
from scipy import sparse
from lightning.pytorch.loggers import CSVLogger

# from lightning.pytorch.strategies import DDPStrategy

scvi.settings.seed = seed
scvi._settings.ScviConfig.dl_num_workers = n_cores


# Initialize and train the model:

# In[ ]:


# scvi requires raw counts
adata_raw = sc.read_h5ad(os.path.join(data_path, 'interim', author + '_filtered_counts.h5ad'))
adata.layers['counts'] = adata_raw.X.copy()

scvi.model.SCVI.setup_anndata(
    adata,
    layer = 'counts', 
    #strategy=DDPStrategy(find_unused_parameters=True),
    categorical_covariate_keys=['plate'],
    continuous_covariate_keys=['pcnt_mito', 'cell_cycle_PC1'],
)
scvi_mod = scvi.model.SCVI(adata, n_layers=2, n_latent=30, gene_likelihood="nb")

n_gpus = 1

logger = CSVLogger(
    save_dir=os.path.join(data_path, 'interim'),
    version = 'overwrite_2',
    name=author + '_scvi_mod')

if n_gpus <= 1 or n_gpus is None:
    scvi_mod.train(
        max_epochs = 400, 
        accelerator = 'gpu', 
        devices = 1,  
        early_stopping = True, 
        early_stopping_patience = 30, 
        early_stopping_monitor = 'reconstruction_loss_validation', # rather than elbo
        batch_size = 4098, 
        plan_kwargs={'lr': 1e-3, # default 1e-3 did not decrease loss        
                     'reduce_lr_on_plateau': True,
                     'lr_scheduler_metric': 'reconstruction_loss_validation', # rather than elbo
                     'lr_patience': 8,
                     'lr_factor': 0.6, 
                     'max_kl_weight': 0.5, # default of 1 makes model focus just on KL
                    },
        logger = logger,
    )
else:
    scvi_mod.train(
        max_epochs = 400, 
        accelerator = 'gpu', 
        devices = n_gpus,  
        early_stopping = True, 
        early_stopping_patience = 30, 
        early_stopping_monitor = 'reconstruction_loss_validation', # rather than elbo
        batch_size = 4098, 
        plan_kwargs={'lr': 1e-3, # default 1e-3 did not decrease loss        
                     'reduce_lr_on_plateau': True,
                     'lr_scheduler_metric': 'reconstruction_loss_validation', # rather than elbo
                     'lr_patience': 8,
                     'lr_factor': 0.6, 
                     'max_kl_weight': 0.5, # default of 1 makes model focus just on KL
                    },
        logger = logger,
        strategy='ddp_find_unused_parameters_true', 
    )

print('Training complete')
utils.clear_memory()

print('Save scvi model')
# qzm, qzv = scvi_mod.get_latent_representation(give_mean=False, return_dist=True)
# scvi_mod.adata.obsm["X_latent_qzm"] = qzm
# scvi_mod.adata.obsm["X_latent_qzv"] = qzv

# scvi_mod.minify_adata()

scvi_mod.save(os.path.join(data_path, 'processed', author + '_scvi_mod.scvi'), overwrite=True)
utils.clear_memory()


# In[67]:


print('Get counts')
adata.layers['normalized_counts'] = adata.X.copy()
scvi_counts = scvi_mod.get_normalized_expression(adata, library_size=1e6).astype('float32').values
adata.X = sparse.csr_matrix(scvi_counts)
utils.clear_memory()

print('Get latent')
adata.obsm['X_scVI'] = scvi_mod.get_latent_representation(adata)
utils.clear_memory()

print('Save anndata object')
adata.write_h5ad(os.path.join(data_path, 'interim', author + '_scvi_counts.h5ad'))


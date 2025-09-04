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

import Tahoe_utils as Tu
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

adata = sc.read_h5ad(os.path.join(data_path, 'interim', author + '_normalized_counts.h5ad'))
models, assessment, X_pls = Tu.pls_da(adata,
       n_components = 5,
       control_confounders = False,
          assess = True,
          return_components = True,
          seed = 888, 
          enc_X = None, 
          enc_Y = None)
pls_model = models['pls_model']


# In[ ]:


with open("trash_norm.json", "w") as f:
    json.dump(assessment, f, indent=4)


# In[ ]:


proj_df = pd.DataFrame(X_pls)
proj_df.columns = ['PLS{}'.format(i+1) for i in range(proj_df.shape[1])]

res = []
for cov_ in ['drug', 'cell_line']:
    cov = adata.obs[cov_].astype(str)
    enc = OneHotEncoder(drop="first", sparse_output=False)
    cov_encoded = enc.fit_transform(cov.values.reshape(-1, 1))

    r2_scores = []
    for pls_idx in trange(proj_df.shape[1]):
        y = proj_df.iloc[:, pls_idx]
        model = LinearRegression().fit(cov_encoded, y)
        r2 = model.score(cov_encoded, y)
        r2_scores.append({"PLS": pls_idx + 1, "R2": r2})

    r2_df = pd.DataFrame(r2_scores)
    r2_df['covariate'] = cov_
    res.append(r2_df)
    
r2_df = pd.concat(res, axis=0, ignore_index=True)
r2_df.to_csv('trash_norm.csv')





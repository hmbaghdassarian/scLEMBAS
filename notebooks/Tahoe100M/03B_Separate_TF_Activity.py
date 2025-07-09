#!/usr/bin/env python
# coding: utf-8

# # Drug Separation
# 
# These are clearly separating much more strongly by cell line than perturbation. here, we will attempt to achieve an embedding that demonstrates good drug perturbation separation.

# In[1]:


import os
import ast
import json
import joblib
from joblib import Parallel, delayed
import time

from tqdm import tqdm
from tqdm import trange

import numpy as np
import pandas as pd

from scipy.stats import f_oneway, kruskal
from scipy import stats
from sklearn.metrics import normalized_mutual_info_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

from kneed import KneeLocator

import scanpy as sc
import umap

import matplotlib.pyplot as plt
import seaborn as sns

import sys
sclembas = '/home/hmbaghda/Projects/scLEMBAS'
sys.path.insert(1, os.path.join(sclembas))
from scLEMBAS import io


# In[2]:


n_cores = 30
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)

seed = 888
data_path = '/nobackup/users/hmbaghda/scLEMBAS/analysis'
author = 'Tahoe100M'


# In[3]:


tf_adata = io.read_tfad(os.path.join(data_path, 'processed', author + '_consensus_tf_activity.h5ad'))


# Step 1: Fit a PLS-DA model of drug ~ TF activity , including confounders (cell line, plate, cell cycle scores) as X covariates to remove those sources of variance.
# - since only cell line information will be available in the generative model, we only control for this (we need to be able to project the predicted data into this space)
# #- categorical confounders (cell line and plate) are one-hot encoded
# #- continuous confounders (cell cycle scores) are scaled
# - X (TF activity) is not scaled because it is the consensus Z-score from decoupler
# 
# We will identify an optimal n components by assessing the following:
# 1) Mean accuracy score across 5-fold CV of a logistic regression classifier trained on the PLS components (cannot directly use PLS model CV because it is technically a regression model on the one-hot encodings, and the high R^2 does not necessarily mean high classification accuracy)
# 2) Explained variance in the X and y blocks

# In[4]:


def ss_explained_var(Y, Y_pred):
    ss_res = np.sum((Y - Y_pred) ** 2)
    ss_tot = np.sum((Y - np.mean(Y, axis=0)) ** 2)
    return 1 - ss_res / ss_tot


def prepare_input_matrix(tf_adata, 
                         control_confounders = True, 
                        enc_X = None):
    
    """Prepares input for PLSR, and fits one-hot encodings of 
    confounding categorical covariates (cell line and plate)
    
    *Note, noticed that adding in cell cycle scores doesn't make a difference, but including
    the plate does make a substantial difference in assessment metrics. 
    """
    X = tf_adata.X
    if control_confounders:
#         # covariates
        cell_line = tf_adata.obs['cell_line'].astype(str).values
        plate = tf_adata.obs['plate'].astype(str).values

        if enc_X is None:
            enc_X = OneHotEncoder(sparse_output=False, drop='first')  # drop to avoid collinearity
            enc_X.fit(np.stack([cell_line, plate], axis=1))
            covariates = enc_X.transform(np.stack([cell_line, plate], axis=1)) 

#         cell_cycle_scores = np.concatenate([
#             tf_adata.obs['S_score'].values.reshape(-1, 1),
#             tf_adata.obs['G2M_score'].values.reshape(-1, 1)
#         ], axis=1)

#         scaler = StandardScaler()
#         cell_cycle_scaled = scaler.fit_transform(cell_cycle_scores)

#         X = np.concatenate([X, covariates, cell_cycle_scaled], axis=1)
        X = np.concatenate([X, covariates], axis=1)
    return X, enc_X


def pls_da(tf_adata, 
           n_components: int, 
          control_confounders: bool = True, 
          assess: bool = True, 
           return_components: bool = True,
          seed: int = 888, 
          enc_X = None, 
          enc_Y = None):
    """Creates a PLS-DA model (drug ~ TF activity) with confounding covariates

    Parameters
    ----------
    tf_adata : _type_
        TF activity anndata object
    n_components : int
        Number of PLS components to use
    control_confounders : bool, optional
        controls for various confounders in the X-block of PLS-DA, by default True
    assess : bool, optional
        gets assessment metrics for PLS fit, by default True
    return_components : bool, optional
        returns the X PLS components, by default True
    seed : int, optional
        random state, by default 888
    enc_X: 
        fit model for encoding. If not provided (when fitting on actual data) will fit an encoding. 
        If provided (when projecting predicted data), will use the fit encoding to transform the input
    """
    

    y = tf_adata.obs['drug'].astype(str).values.reshape(-1,1)
    if enc_Y is None:
        enc_Y = OneHotEncoder(sparse_output=False, drop=None)    
        enc_Y.fit(y)
    Y = enc_Y.transform(y)

    X, enc_X = prepare_input_matrix(tf_adata = tf_adata,
                                    control_confounders = control_confounders, 
                                   enc_X = enc_X)

    pls_model = PLSRegression(n_components=n_components)
    pls_model.fit(X, Y)
    
    X_pls = None
    if assess or return_components:
        X_pls = pls_model.transform(X)
    
    assessment = None
    if assess: 
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        accuracy_score = cross_val_score(clf, X_pls, y.ravel(), 
                                         cv=StratifiedKFold(5, random_state=seed, shuffle=True), 
                                         scoring='accuracy').mean()
        
#         explained_var_y = np.var(pls_model.y_scores_, axis=0, ddof=0) / np.var(Y, axis=0, ddof=0).sum()
#         

        Y_pred = pls_model.predict(X)
        explained_var_y = ss_explained_var(Y, Y_pred) #1 - np.sum((Y - Y_pred) ** 2) / np.sum((Y - Y.mean(axis=0)) ** 2)
        
        
#         X_pred = pls_model.x_scores_ @ pls_model.x_loadings_.T
#         explained_var_x = ss_explained_var(X, X_pred)
#         explained_var_x = np.var(pls_model.x_scores_, axis=0) / np.var(X, axis=0).sum()
#         cum_explained_x = np.cumsum(explained_var_x)[-1]

        assessment = {
            'n_components': n_components,
            'accuracy': accuracy_score,
            'explained_y': explained_var_y}  
    
    models = {'pls_model': pls_model, 
             'encoder_x': enc_X, 
             'encoder_y': enc_Y}

    return models, assessment, X_pls


# In[5]:


assessment_df = []
pls_components_max = 25
for n_components in tqdm(range(pls_components_max, 0, -1)): #(1, pls_components_max + 1):
    _, assessment, _= pls_da(tf_adata = tf_adata, 
                                 n_components = n_components ,
                                 control_confounders = True, 
                                 assess = True,
                             return_components = True, 
                                 seed = seed, 
                            enc_X = None, 
                            enc_Y = None)
    assessment_df.append(assessment)
    
assessment_df = pd.DataFrame(assessment_df)
assessment_df = assessment_df.sort_values(by = 'n_components').reset_index(drop = True)
assessment_df.to_csv(os.path.join(data_path, 'interim', author + '_PLS_drug_scores.csv'))

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import scanpy as sc
import pandas as pd


rev_stim = {'STIM': 'CTRL', 'CTRL': 'STIM'}

stim_map = {'STIM': 1, 'CTRL': 0}
rev_stim_map = {v:k for k,v in stim_map.items()}

def get_prediction(mod, tf_adata, counterfactual_type, cf_map, 
                   train_cells_all, test_conds):
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


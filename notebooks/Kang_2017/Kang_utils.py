"""Common functions used across multiple scripts"""


import torch
import scanpy as sc
import pandas as pd

from typing import List, Literal


rev_stim = {'STIM': 'CTRL', 'CTRL': 'STIM'}

stim_map = {'STIM': 1, 'CTRL': 0}
rev_stim_map = {v:k for k,v in stim_map.items()}

def get_prediction(mod, tf_adata: List[str], 
                   counterfactual_type: Literal['in_distribution', 'opposite'], cf_map, 
                   train_cells_all, test_conds, return_bias: bool = False):
    """Get prediction from a model given a counterfactual

    Parameters
    ----------
    mod : _type_
        _description_
    tf_adata : _type_
        all the actual data
    counterfactual_type : Literal['in_distribution', 'opposite']
        in distribution will be all train cells to each test cond
        opposite will be within the same cell type, predicting the test cond from the opposit stimulation condition
    cf_map : _type_
        keys are the label for the counterfactual, values are the list of cells in that counterfactual that aren't 'opposite' ('opposite' calculated internally)
    train_cells_all : List[str]
        all cells the model was trained on 
    test_conds : List[str]
        cell type^stimulation to predict
    return_bias : bool, optional
        whether to return bias terms (True) or prediction (False), by default False
    """
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
        if return_bias:
            bias_global, bias_mu, bias_log_sigma_squared = biases
            bias_sigma = torch.exp(bias_log_sigma_squared/2.) + mod.signaling_network.vae.var_min

            # add in categorical information
            bias_cats = torch.zeros_like(bias_global.T, device = mod.device, dtype = mod.dtype)
            for cat_group_idx in range(full_covariates.shape[1]):
                cat_group = mod.signaling_network._cat_group_idx[cat_group_idx]
                mod.signaling_network.cat_embeddings[cat_group].weight.data.masked_fill_(mask = mod.signaling_network.cat_embeddings_mask[cat_group], 
                                                                            value = 0.0)
                bias_cats += mod.signaling_network.cat_embeddings[cat_group](full_covariates[:,cat_group_idx]).T
            bias_tot = bias_global.T + bias_cats
            
            obs = pd.DataFrame(full_covariates.detach().cpu().numpy())
            obs.columns = ['seurat_annotations']
            obs.seurat_annotations = obs.seurat_annotations.map(cov_rev_map)
            
            return bias_global, bias_mu, bias_sigma, bias_cats, bias_tot, obs
        
    obs = pd.DataFrame(full_covariates.detach().cpu().numpy())
    obs.columns = ['seurat_annotations']
    obs.seurat_annotations = obs.seurat_annotations.map(cov_rev_map)
    obs['stim'] = pd.Series(full_X.detach().cpu().numpy().reshape(-1)).map(rev_stim_map)
    obs['condition'] = obs['stim'].astype(str) + '^' + obs['seurat_annotations'].astype(str)
    
    y_predicted = pd.DataFrame(y_predicted.detach().cpu().numpy())
    y_predicted.columns = mod.y_out.columns
    tf_adata_predicted = sc.AnnData(X = y_predicted, obs = obs)
    
    return tf_adata_predicted



def get_best_hyperparams(res, 
                         iter_cols = ['max_epochs', 'max_lr', 'train_batch_size'], 
                        loss_col = 'emd_loss_total'):
    
    res_nans = res.groupby(iter_cols)[loss_col].apply(lambda x: x.isna().any()).reset_index()
    res_nans = res_nans[res_nans[loss_col]]
    res_nans.set_index(iter_cols, inplace = True)

    mean_emd = res.groupby(iter_cols).emd_loss_total.mean().reset_index()
    mean_emd = mean_emd.set_index(iter_cols).drop(res_nans.index)
    best_emd_loss = mean_emd.emd_loss_total.min()
    best_emd = mean_emd[mean_emd.emd_loss_total == best_emd_loss]
    best_emd_mean = best_emd.copy()
    
    best_hyperparams = dict(zip(best_emd.index.names, best_emd.index[0]))
    best_emd = res[(res.max_epochs == best_hyperparams['max_epochs']) & 
                   (res.max_lr == best_hyperparams['max_lr']) & 
                   (res.train_batch_size == best_hyperparams['train_batch_size'])]
    
    return best_emd_mean, best_hyperparams, best_emd

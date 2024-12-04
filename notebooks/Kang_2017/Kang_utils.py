"""Common functions used across multiple scripts"""


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

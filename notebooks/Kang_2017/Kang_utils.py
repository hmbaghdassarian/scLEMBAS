"""Common functions used across multiple scripts"""

import gc

import torch
import scanpy as sc
import pandas as pd
import numpy as np

from sklearn.metrics import normalized_mutual_info_score

from geomloss import SamplesLoss

from typing import List, Literal, Optional

import sys
import os
sclembas = '/home/hmbaghda/Projects/scLEMBAS'
sys.path.insert(1, os.path.join(sclembas))
from scLEMBAS.preprocess import embed_tf_activity

device = "cuda" if torch.cuda.is_available() else "cpu"

rev_stim = {'STIM': 'CTRL', 'CTRL': 'STIM'}

stim_map = {'STIM': 1, 'CTRL': 0}
rev_stim_map = {v:k for k,v in stim_map.items()}

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.reset_peak_memory_stats()

def setup_prediction(mod, 
                     train_cells,
                     tf_adata, 
                     train_mode, 
                     counterfactual,
                     ):

    cov_idx_map = dict(zip(mod.signaling_network.covariates['seurat_annotations'], 
                        mod.signaling_network.covariates_idx['seurat_annotations']))
    cov_rev_map = {v:k for k,v in cov_idx_map.items()}
    full_expr, full_X, full_covariates = None, None, None

    train_conds = tf_adata.obs.loc[train_cells, 'condition'].unique()
    test_conds = tf_adata[~tf_adata.obs.condition.isin(train_conds), ].obs.condition.unique()

    if not counterfactual and not train_mode:
        raise ValueError('Trying to predict test cells without a counterfactual')

    iterable_conds = train_conds if train_mode else test_conds
    for cond in sorted(iterable_conds): 
        stim, ct = cond.split('^') # stim and ct of the prediction
        
        if counterfactual: 
            predict_cells_from = tf_adata.obs[(tf_adata.obs['condition'] == rev_stim[stim] + '^' + ct)].index.tolist()
        else:
            predict_cells_from = tf_adata.obs[tf_adata.obs.condition == cond].index.tolist()
        n_predictions = len(predict_cells_from)
            
        # generate model inputs
        # input gene expression: counterfactual or not
        expr_in = mod.df_to_tensor(mod.expr.loc[predict_cells_from, :])
        
        # input stimulation
        X_in = pd.DataFrame(data = {'IFNB1': [stim_map[stim]]*n_predictions})
        X_in = mod.df_to_tensor(X_in)
        
        # input ct
        covariates_in = torch.tensor([cov_idx_map[ct]]*n_predictions,
                                        device = mod.device, dtype = torch.int64).view(-1,1)
        
        full_expr = expr_in if full_expr is None else torch.cat((full_expr, expr_in), dim = 0)
        full_X = X_in if full_X is None else torch.cat((full_X, X_in), dim = 0)
        full_covariates = covariates_in if full_covariates is None else torch.cat((full_covariates, covariates_in), dim = 0)
        
        clear_memory()
    
    # metadata setup
    obs = pd.DataFrame(full_covariates.detach().cpu().numpy())
    obs.columns = ['seurat_annotations']
    obs.seurat_annotations = obs.seurat_annotations.map(cov_rev_map)
    obs['stim'] = pd.Series(full_X.detach().cpu().numpy().reshape(-1)).map(rev_stim_map)
    obs['condition'] = obs['stim'].astype(str) + '^' + obs['seurat_annotations'].astype(str)

    return full_expr, full_X, full_covariates, obs


def run_prediction(mod, 
                   remove_type, 
                   return_bias, 
                   X_in, 
                   covariates_idx, 
                   expr, 
                  obs, 
                  return_full):

    # ----------------CHECKS----------------
    if type(remove_type) != list:
        remove_type = [remove_type]
    if len(remove_type) not in [1,2]:
        raise ValueError('Cannot remove more than two components at once')
    if len(set(remove_type).difference(['none', 'global_bias', 'categorical_bias', 'total_bias', 'adj'])) > 0:
        raise ValeuError('Incorrect remove_type specified')
    if len(remove_type) == 2:
        if sorted(remove_type) not in [['adj', 'categorical_bias'], ['adj', 'global_bias']]:
            raise ValueError('Can only specify multiple remove types when ')
    if remove_type != ['none'] and return_bias:
        raise ValueError('Have not considered looking at the bias components without the full forward pass')
    
    # ----------------FORWARD PASS----------------
    mod.eval()
    with torch.inference_mode():
        X_full = mod.input_layer(X_in) # input ligands to signaling network

        bias_cats = torch.zeros_like(X_full.T, device = mod.signaling_network.device, dtype = mod.signaling_network.dtype)
        # add categorical covariates
        for cat_group_idx in range(covariates_idx.shape[1]):
            cat_group = mod.signaling_network._cat_group_idx[cat_group_idx]
            bias_cats += mod.signaling_network.cat_embeddings[cat_group](covariates_idx[:,cat_group_idx]).T

        bias_mu, bias_log_sigma_squared, bias_global = mod.signaling_network.vae(expr)
        bias_global.data.masked_fill_(mask = mod.signaling_network.bias_mask.T.expand(bias_global.shape[0], -1), value = 0.0) # apply bias mask

#         if 'bias_global_scaler' in mod.signaling_network.bionet_params:
#             bias_global /= mod.signaling_network.bionet_params['bias_global_scaler']

#         # this is equivalent to dividing bias_tot by the scalers, but since we get out individual components 
#         # we should scale each individual component
#         elif 'bias_tot_scaler' in mod.signaling_network.bionet_params:
#             bias_global /= mod.signaling_network.bionet_params['bias_tot_scaler']
#             bias_cats /= mod.signaling_network.bionet_params['bias_tot_scaler']

        if return_bias:
            bias_tot = bias_global.T + bias_cats
            bias_sigma = torch.exp(bias_log_sigma_squared/2.) + mod.signaling_network.vae.var_min
            clear_memory()
            return bias_global, bias_mu, bias_sigma, bias_cats, bias_tot, obs

        if remove_type == ['none'] or remove_type == ['adj']:
            bias_tot = bias_global.T + bias_cats # include all biases
        elif 'categorical_bias' in remove_type:
            bias_tot = bias_global.T # don't include categorical bias
        elif 'global_bias' in remove_type:
            bias_tot = bias_cats # don't include global bias
        elif remove_type == ['total_bias']:
            bias_tot = torch.zeros_like(X_full.T, device = mod.signaling_network.device, dtype = mod.signaling_network.dtype) # don't include bias    
        else:
            raise ValueError('Incorrect remove_type specified')

        X_bias = X_full.T + bias_tot # this is the bias with the projection_amplitude included
        X_new = torch.zeros_like(X_bias) #initialize hidden state values at 0

        if 'adj' in remove_type: 
            X_new = mod.signaling_network.activation(X_bias,
                                                     mod.signaling_network.bionet_params['leak'])
            # this is the equivalen of setting the signaling network weights to 0 in the 
            # iteration below because this makes X_new = 0 at every element in the forward pass

            # see commented out remove_type == 'adj' below for the equivalent
        else:
            for t in range(mod.signaling_network.bionet_params['max_steps']): # like an RNN, updating from previous time step
                X_old = X_new

    #             if remove_type == 'adj':
    #                 X_new = torch.mm(torch.zeros(mod.signaling_network.weights.shape,
    #                             device = mod.signaling_network.device, 
    #                             requires_grad=False), X_new)
                if 'signaling_weights_scaler' in mod.signaling_network.bionet_params: #DEV
                     X_new = torch.mm(mod.signaling_network.weights*mod.signaling_network.bionet_params['signaling_weights_scaler'],
                                      X_new) # scale matrix by edge weights
                else:
                    X_new = torch.mm(mod.signaling_network.weights, X_new) # scale matrix by edge weights

                X_new = X_new + X_bias  # add original values and bias       
                X_new = mod.signaling_network.activation(X_new, mod.signaling_network.bionet_params['leak'])

                if (t % 10 == 0) and (t > 20):
                    diff = torch.max(torch.abs(X_new - X_old))    
                    if diff.lt(mod.signaling_network.bionet_params['tolerance']):
                        break
        
        Y_full = X_new.T
        
        if return_full:
            return sc.AnnData(X = Y_full.detach().cpu().numpy(), obs = obs)

        y_predicted = mod.output_layer(Y_full)
        
        clear_memory()
    if remove_type == ['none']:
        consistent_forward = True
        y_predicted_, Y_full_, biases_ = mod(X_in, covariates_idx, expr)
        for tt_1, tt_2 in zip([y_predicted, Y_full, bias_global, bias_mu, bias_log_sigma_squared], 
                              [y_predicted_, Y_full_, *biases_]):
            if not torch.equal(tt_1, tt_2):
                consistent_forward = False
        if not consistent_forward:
            raise ValueError('Prediction here does not match forward pass')
        del y_predicted_, Y_full_, biases_

    y_predicted = pd.DataFrame(y_predicted.detach().cpu().numpy())
    y_predicted.columns = mod.y_out.columns
    tf_adata_predicted = sc.AnnData(X = y_predicted, obs = obs)

    del X_full, X_in, covariates_idx, expr
    del bias_mu, bias_log_sigma_squared, bias_global
    del X_bias, X_new, Y_full
    if 'adj' not in remove_type:
        del X_old

    clear_memory()

    return tf_adata_predicted

def get_prediction(mod,
                   train_cells,
                   tf_adata,
                   train_mode: bool = False,
                   counterfactual: bool = True,
                   remove_type: Literal['none', 'global_bias', 'categorical_bias', 'total_bias', 'adj'] = 'none',
                   return_bias: bool = False, 
                  max_cells = None, 
                  return_full: bool = False):
    """Get prediction from a model given a counterfactual

    Parameters
    ----------
    mod : _type_
        _description_
    tf_adata : _type_
        all the actual data
    train_cells : List[str]
        all cells the model was trained on 
    train_mode : bool
        predicts the training conditions, as a negative control to see how model predictions perform on training data
    return_bias : bool, optional
        whether to return bias terms (True) or prediction (False), by default False
    counterfactual : bool, optional
        whether to calculate from opposite stimulation condition (True) or same one (False)
    remove_type : Literal['none', 'global_bias', 'categorical_bias', 'total_bias', 'adj'], optional
        can be a string or a list of strings
        which components of bias/adj matrix to include in the prediction, by default 'all_bias'; 
        only incorporated if return_bias = False
        any bias component includes the full adjacency matrix
        - 'none': includes all components in the prediction
        - 'categorical_bias': includes global but excludes categorical bias in the prediction
        - 'global_bias': includes categorical but excludes global bias in the prediction
        - 'total_bias': does not include bias in the prediction (just input and signaling weights)
        - 'adj': includes all bias but sets signaling weights to 0
        the only list of strings are combining either categorical or global bias with adj, since in these cases just removing one of the two bias components still leaves two components in the model, making it hard to decouple effects. 
    test_cells : 
        the list of test cell barcodes, necessary if return_loss = True
    max_cells : int
        the max cells in a forward pass; for cuda memory, will break up into chunks
    return_full : bool, optional
        whether to return model output prior to ProjectOutput transformation (True) or after (False), by default False
    """
    
    max_cells = np.inf if max_cells is None else max_cells
    
    expr, X_in, covariates_idx, obs = setup_prediction(mod, 
                                                       train_cells,
                                                       tf_adata, 
                                                       train_mode, 
                                                       counterfactual
                                                              )

    if expr.shape[0] < max_cells:
        res = run_prediction(mod, 
                             remove_type, 
                             return_bias, 
                             X_in, 
                             covariates_idx, 
                             expr, 
                            obs, 
                            return_full)

    else:
        res = []

        # split into chunks for cuda memory
        expr_chunks = torch.split(expr, max_cells)
        X_in_chunks = torch.split(X_in, max_cells)
        covariates_idx_chunks = torch.split(covariates_idx, max_cells)
        obs = obs.copy()
        obs.index = obs.index.astype(str)
        obs_index = obs.index
        obs_chunks = [obs_index[i:i+max_cells] for i in range(0, len(obs_index), max_cells)]

        for chunk_idx in range(len(expr_chunks)):
            res_ = run_prediction(mod, 
                                 remove_type, 
                                 return_bias, 
                                 X_in_chunks[chunk_idx], 
                                 covariates_idx_chunks[chunk_idx], 
                                 expr_chunks[chunk_idx], 
                                obs.loc[obs_chunks[chunk_idx], :], 
                                 return_full)
            res.append(res_)

        if not return_bias:
            res = sc.concat(res)
            res.obs_names_make_unique()
        else:
            bias_global_chunks, bias_mu_chunks, bias_sigma_chunks, bias_cats_chunks, bias_tot_chunks, obs_chunks = zip(*res)

            obs = pd.concat(obs_chunks, axis=0)
            bias_global = torch.cat(bias_global_chunks, dim=0)
            bias_mu = torch.cat(bias_mu_chunks, dim=0)
            bias_sigma = torch.cat(bias_sigma_chunks, dim=0)
            bias_cats = torch.cat(bias_cats_chunks, dim=1)
            bias_tot = torch.cat(bias_tot_chunks, dim=1)

            res = (bias_global, bias_mu, bias_sigma, bias_cats, bias_tot, obs)
    return res



def get_loss(tf_adata, tf_adata_predicted, device = device):
    """Calculates the loss between predicted and actual data per condition. 
    geom_loss by default normalizes to sample size so that doesn't need to be done. 
    Does take average across conditions for the total loss (rather than simply summing), so that it does not change with 
    the number of conditions 
    """
    loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05).to(device)
    loss = {}
    conds = tf_adata_predicted.obs.condition.unique()
    for cond in conds:
        y_predicted = torch.tensor(tf_adata_predicted[tf_adata_predicted.obs.condition == cond, ].to_df().values).to(device)
        y_actual = torch.tensor(tf_adata[tf_adata.obs.condition == cond, ].to_df().values).to(device)
        loss[cond] = loss_fn(y_predicted, y_actual)
        clear_memory()
    loss['EMD Loss'] = sum(loss.values())/len(loss) # averaged across condition to not scale with n_conditions
    
    
    return {k: float(v.cpu().numpy()) for k,v in loss.items()}



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


def adata_dimviz_bias(adata, reduction_type, cat, subset_size = None, seed = 888):
    viz_df = pd.DataFrame(adata.obsm['X_' + reduction_type])
    viz_df = pd.concat([viz_df, pd.DataFrame(adata.obs[cat]).reset_index(drop = True)], ignore_index = True, axis = 1)

    reduction_type_ = 'pc' if reduction_type == 'pca' else reduction_type
    viz_df.columns = [reduction_type_.upper() + str(i+1) for i in range(viz_df.shape[1])]
    viz_df.columns = viz_df.columns[:-1].tolist() + [cat]
    
    nmi = normalized_mutual_info_score(adata.obs.leiden, adata.obs[cat])
    
    if subset_size is not None:
        cell_prop = viz_df[cat].value_counts()/viz_df.shape[0]
        index_to_keep = []
        for cell_type in viz_df[cat].unique():
            np.random.seed(seed)
            index_to_keep += np.random.choice(viz_df[viz_df[cat] == cell_type].index, 
                                              size = int(np.round(cell_prop.loc[cell_type]*subset_size)), 
                                              replace = False).tolist()
        viz_df = viz_df.loc[index_to_keep, :]
    
    # shuffle
    viz_df = viz_df.sample(frac=1, random_state = seed).reset_index(drop=True)
    return viz_df,nmi

def adata_dimviz_prediction(adata, reduction_type, cats, max_condition_size = None, seed = 888):
    viz_df = pd.DataFrame(adata.obsm['X_' + reduction_type])
    for cat in cats:
        viz_df = pd.concat([viz_df, pd.DataFrame(adata.obs[cat]).reset_index(drop = True)], ignore_index = True, axis = 1)
    if reduction_type=='umap':
        viz_df.columns = [reduction_type.upper() + str(i+1) for i in range(viz_df.shape[1])]
    elif reduction_type=='pca':
        viz_df.columns = [reduction_type.upper()[:-1] + str(i+1) for i in range(viz_df.shape[1])]
    viz_df.columns = viz_df.columns[:-len(cats)].tolist() + cats
    
    if max_condition_size is not None:
#         cell_prop = viz_df.condition.value_counts()/viz_df.shape[0]
#         index_to_keep = []
#         for cond in viz_df.condition.unique():
#             np.random.seed(seed)
#             index_to_keep += np.random.choice(viz_df[viz_df.condition == cond].index, 
#                                               size = int(np.round(cell_prop.loc[cond]*subset_size)), 
#                                               replace = False).tolist()
#         viz_df = viz_df.loc[index_to_keep, :]
        
        
        vc = viz_df.condition.value_counts()
        subset_conds = vc[vc > max_condition_size].index.tolist()

        drop_idx = []
        for subset_cond in subset_conds: 
            np.random.seed(seed)
            drop_idx += list(np.random.choice(viz_df[viz_df.condition == subset_cond].index, 
                             vc[subset_cond] - subset_size, 
                            replace = False))
        viz_df.drop(index = drop_idx, inplace = True)

    # shuffle
    
    viz_df = viz_df.sample(frac=1, random_state = seed).reset_index(drop=True)
    return viz_df


def prepare_for_metrics(tf_adata, 
                tf_adata_predicted, 
                resolution,
                calculation_type: Literal['embed', 'project'] = 'project',
                n_neighbors: int = 15, 
                run_umap = True, 
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
        # project new data into PCA space
        pc_rank = tf_adata.uns["pca"]['pca_rank']
        pca_mod = tf_adata.uns['pca']['pca_mod']
        tf_adata_.obsm['X_pca'] = pca_mod.transform(tf_adata_.to_df().values)
        tf_adata_.uns['pca'] = tf_adata.uns['pca'].copy()
        
        # neihgbors/clustering
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
        
        # project from PCA space into UMAP space
        if run_umap:
            from scanpy.tools._utils import _choose_representation
            from scanpy._utils import NeighborsView
            
            neighbors = NeighborsView(tf_adata_, 'neighbors')
            X_pca = _choose_representation(
                tf_adata_,
                use_rep=neighbors['params'].get("use_rep", None),
                n_pcs=neighbors['params'].get("n_pcs", None),
                silent=True,
            )

            tf_adata_.obsm['X_umap'] = tf_adata_actual.uns['umap']['umap_mod'].transform(X_pca)
            tf_adata_.uns['umap'] = tf_adata_actual.uns['umap'].copy()
        

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
                          run_umap = run_umap, 
                         cluster_data = True)
    return tf_adata_

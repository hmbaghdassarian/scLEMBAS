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


def get_prediction(mod, tf_adata: List[str], 
                   counterfactual_type: Literal['in_distribution', 'opposite'], cf_map, 
                   train_cells_all, test_conds, 
                   return_bias: bool = False,
                  remove_type: Literal['none', 'global_bias', 'categorical_bias', 'total_bias', 'adj'] = 'none', 
#                   return_loss: bool = True, 
                  test_cells: Optional[List[str]] = None, 
                  train_mode: bool = False):
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
    train_mode : bool
        predicts the training conditions, as a negative control to see how model predictions perform on training data
    """
    
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

    
    # ----------------SETUP----------------
    cov_idx_map = dict(zip(mod.signaling_network.covariates['seurat_annotations'], 
                           mod.signaling_network.covariates_idx['seurat_annotations']))
    cov_rev_map = {v:k for k,v in cov_idx_map.items()}


    full_expr, full_X, full_covariates = None, None, None

    if not train_mode:
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
    else:
        train_conds = tf_adata.obs.loc[train_cells_all, 'condition'].unique()
        for cond in train_conds: # predict the training condition from the training condition
            stim, ct = cond.split('^')
            train_cells_cond = tf_adata.obs[tf_adata.obs.condition == cond].index.tolist()

            expr_test = mod.df_to_tensor(mod.expr.loc[train_cells_cond, :])

            X_test_df = pd.DataFrame(data = {'IFNB1': [stim_map[stim]]*len(train_cells_cond)})
            X_test = mod.df_to_tensor(X_test_df)

            covariates_idx_test = torch.tensor([cov_idx_map[ct]]*len(train_cells_cond), 
                                           device = mod.device, dtype = torch.int64).view(-1,1)

            full_expr = expr_test if full_expr is None else torch.cat((full_expr, expr_test), dim = 0)
            full_X = X_test if full_X is None else torch.cat((full_X, X_test), dim = 0)
            full_covariates = covariates_idx_test if full_covariates is None else torch.cat((full_covariates, covariates_idx_test), dim = 0)


    # metadata setup
    obs = pd.DataFrame(full_covariates.detach().cpu().numpy())
    obs.columns = ['seurat_annotations']
    obs.seurat_annotations = obs.seurat_annotations.map(cov_rev_map)
    obs['stim'] = pd.Series(full_X.detach().cpu().numpy().reshape(-1)).map(rev_stim_map)
    obs['condition'] = obs['stim'].astype(str) + '^' + obs['seurat_annotations'].astype(str)

    # ----------------FORWARD PASS----------------
    X_in, covariates_idx, expr = full_X, full_covariates, full_expr

    clear_memory()
    
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

        if 'bias_global_scaler' in mod.signaling_network.bionet_params:
            bias_global /= mod.signaling_network.bionet_params['bias_global_scaler']

        # this is equivalent to dividing bias_tot by the scalers, but since we get out individual components 
        # we should scale each individual component
        elif 'bias_tot_scaler' in mod.signaling_network.bionet_params:
            bias_global /= mod.signaling_network.bionet_params['bias_tot_scaler']
            bias_cats /= mod.signaling_network.bionet_params['bias_tot_scaler']

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

#     if return_loss:
#         if not train_mode:
#             if test_cells is None:
#                 # TODO: do not make test_cells an argument in the function, this line suffices
#                 test_cells = tf_adata.obs[tf_adata.obs.condition.isin(test_conds)].index.tolist()
#             tot_loss = 0
#             for cond in test_conds:
#                 stim, ct = cond.split('^')

#                 test_cells_cond = tf_adata.obs[(tf_adata.obs.index.isin(test_cells)) & (tf_adata.obs['condition'] == cond)].index.tolist()
#                 # ^ this should work as just one or the other of the two conditions, but keep as is since it works

#                 y_test = mod.df_to_tensor(tf_adata.to_df().loc[test_cells_cond, :])  

#                 loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05).to(mod.device)
#                 tot_loss += loss_fn(y_predicted[obs[obs.condition == cond].index,:], y_test).detach().cpu().item() 
#                 clear_memory()
#         else:
#             tot_loss = 0
#             for cond in train_conds:
#                 stim, ct = cond.split('^')

#                 train_cells_cond = tf_adata.obs[(tf_adata.obs['condition'] == cond)].index.tolist()
#                 y_test = mod.df_to_tensor(tf_adata.to_df().loc[train_cells_cond, :])  

#                 loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05).to(mod.device)
#                 tot_loss += loss_fn(y_predicted[obs[obs.condition == cond].index,:], y_test).detach().cpu().item() 
#                 clear_memory()
#     else:
#         tot_loss = None

    y_predicted = pd.DataFrame(y_predicted.detach().cpu().numpy())
    y_predicted.columns = mod.y_out.columns
    tf_adata_predicted = sc.AnnData(X = y_predicted, obs = obs)

    del X_full, full_expr, full_X, full_covariates
    del bias_mu, bias_log_sigma_squared, bias_global
    del X_bias, X_new, Y_full
    if 'adj' not in remove_type:
        del X_old

    clear_memory()

    return tf_adata_predicted

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

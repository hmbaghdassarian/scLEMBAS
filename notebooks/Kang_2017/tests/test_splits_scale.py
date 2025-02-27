#!/usr/bin/env python
# coding: utf-8

# In[52]:


subset = False


# In[53]:


import os
from typing import Optional, Dict, List, Literal
import copy
import itertools

import pandas as pd
import numpy as np
import scanpy as sc

from sklearn.metrics import normalized_mutual_info_score

import torch
from geomloss import SamplesLoss
import torch.nn as nn

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split


# In[33]:


import sys
sclembas = '/home/hmbaghda/Projects/scLEMBAS'
sys.path.insert(1, os.path.join(sclembas))
from scLEMBAS import io
from scLEMBAS.model.scl_dev import SignalingModel as SignalingModelDev
from scLEMBAS.model.scl import SignalingModel

from scLEMBAS.model.train import TrainSC
from scLEMBAS.preprocess import discriminator_weight_curve, embed_tf_activity, get_alignment_score


# In[13]:


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
            
            if 'bias_tot_scaler' in mod.signaling_network.bionet_params:
                bias_tot /= mod.signaling_network.bionet_params['bias_tot_scaler']
            
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


# In[14]:


n_cores = 12
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)

data_path = '/nobackup/users/hmbaghda/scLEMBAS/analysis'
device = "cuda" if torch.cuda.is_available() else "cpu"


# In[15]:


tf_adata = io.read_tfad(os.path.join(data_path, 'processed', 'Kang_tf_activity.h5ad'))
tf_adata.obs['condition'] = tf_adata.obs['stim'].astype(str) + '^' + tf_adata.obs['seurat_annotations'].astype(str)
tf_adata = tf_adata[:, sorted(tf_adata.var_names)] 

adata = sc.read_h5ad(os.path.join(data_path, 'processed', 'kang_expr_scored.h5ad'))
sn_ppis = pd.read_csv(os.path.join(data_path, 'processed', 'Kang_sn_ppis.csv'), index_col = 0)

source_label = 'source_genesymbol'
target_label = 'target_genesymbol'
weight_label = 'mode_of_action'
stimulation_label = 'consensus_stimulation'
inhibition_label = 'consensus_inhibition'


# Loop through:

# In[16]:


def generate_lr_params(n_epochs, max_lr, lr_scaling_factor = 10, lr_decay = 0.9):
    lr_period = 3 if n_epochs < 500 else 4
    lr_params = {'max_epochs': n_epochs, 
                 'maximum_learning_rate': max_lr, 
                 'minimum_learning_rate': max_lr/lr_scaling_factor,
                 'lr_restart_epoch': int(n_epochs/lr_period), 
                 'reset_optimizer_epoch': int(n_epochs/3), 
                'lr_decay': lr_decay, 
                 'lr_restart_factor': 1, 
                 'warmup_epochs': int(n_epochs/10)}
    return lr_params

def generate_discriminator_params(n_epochs, max_lr, discriminator_penalty_weight, 
                                  lr_scaling_factor = 10, lr_decay = 0.9):
    general_params = generate_lr_params(n_epochs, max_lr, lr_scaling_factor = lr_scaling_factor, 
                                        lr_decay = lr_decay)
    
    keys_to_keep = ['maximum_learning_rate', 'minimum_learning_rate', 'lr_restart_epoch', 
                   'warmup_epochs', 'lr_decay', 'reset_optimizer_epoch']
    discriminator_params = {'batch_momentum': 0.01,
                            'layer_norm': False,
                            'dropout_rate': 0.1,
                            'activation_fn': nn.LeakyReLU,
                            'n_hidden_nodes': [768, 512, 256],
                            'lr_restart_factor': 1,
                            'optimizer': torch.optim.Adam,
                            'discriminator_lambda_L2': 1e-3,
                            'discriminator_penalty_weight': discriminator_penalty_weight}
    discriminator_params = {**discriminator_params, 
                           **{k:v for k,v in general_params.items() if k in keys_to_keep}}
    
    return discriminator_params


# In[17]:


# hyperparameters
n_layers_vae = 2
n_nodes = len(set(sn_ppis[source_label].tolist() + sn_ppis[target_label].tolist()))
vae_n_hidden_nodes = list(np.round(np.linspace(adata.shape[1], n_nodes, n_layers_vae + 2)).astype(int)[1:-1])

# linear scaling of inputs/outputs
projection_amplitude_in = 10
projection_amplitude_out = 1

# other parameters
bionet_params = {'target_steps': 100, 
                 'max_steps': 120, 
                 'exp_factor':50, 
                 'tolerance': 1e-5, 
                 'leak':1e-2, 
                'cat_max_norm': 100}
vae_params = {'vae_batch_momentum': 0.01, 'vae_layer_norm': False, 'vae_dropout_rate': 0.1,
              'vae_activation_fn': nn.LeakyReLU,
              'vae_n_hidden_nodes': vae_n_hidden_nodes, 
              'vae_var_min': 1e-4}
bionet_params = {**bionet_params, **vae_params}

bionet_params_dev = bionet_params.copy()
bionet_params_dev['bias_tot_scaler'] = 2
bionet_params_dev['signaling_weights_scaler'] = 10


# In[20]:


# training parameters
other_params_default = {'network_noise_scale': 10, 'gradient_noise_scale': 1e-9, 
               'test_batch_size': np.nan}
spectral_radius_params = {'n_probes_spectral': 5, 
                          'power_steps_spectral': 5, #50, 
                          'subset_n_spectral': 5} #10}
target_spectral_radius = 0.9

regularization_params_default = {'input_lambda_L2': 0, # doesn't matter if setting the requires grad to False
                         'bn_weights_lambda_l2': 1e-7, 
                         'bn_bias_lambda_L2': 0, # don't incorporate because of KL divergence
                         'output_weights_lambda_L2': 1e-7,
                         'output_bias_lambda_L2': 1e-7,
                         'moa_lambda_L1': 1e2,  
                         'uniform_lambda_L2': 1e-7,#, 1e-5,
                         'uniform_min': 0,
                         'uniform_max': 1, 
                         'spectral_loss_factor': 1e-6,
                        'vae_lambda_l2': 1e-7, 
                        'vae_scaling_KL': 1e-2}


max_penalty_weight = 7.75
b = 0.6
max_epochs = 600


max_lr = 0.001

if subset:
    batch_factor = 2 if n_fraction <= 0.1 else 3
    train_batch = int(np.round(n_train/batch_factor))
else:
    train_batch = 1024

discriminator_penalty_weight = discriminator_weight_curve(n_epochs = max_epochs,
                                                          min_penalty_weight = 0.1,
                                                          max_penalty_weight = max_penalty_weight,
                                                          a = 1,
                                                          b = b, 
                                                          curve_type = 'power')


lr_params = generate_lr_params(n_epochs = max_epochs, max_lr = max_lr, lr_scaling_factor = 10, lr_decay = 0.9)
discriminator_params = generate_discriminator_params(n_epochs = max_epochs, max_lr = max_lr, 
                      discriminator_penalty_weight = discriminator_penalty_weight, 
                      lr_scaling_factor = 10, lr_decay = 0.9)

regularization_params = regularization_params_default.copy()
other_params = {**other_params_default,
                **{'train_batch_size': train_batch,
                   'validation_batch_size': np.nan}}
training_params = {**lr_params, **other_params, **regularization_params, **spectral_radius_params}


# In[21]:


def ood_split(tf_adata, 
             train_frac: float, 
              stim_col: str = 'stim', 
              context_col: str = 'seurat_annotations', 
              context_bins: Optional[pd.DataFrame] = None, 
              context_bins_frac: float = 1, 
             max_iter: int = 1000, 
             seed: int = 888, 
             deviation_thresh: float = 0.025, 
             include_train_cond: Optional[Dict[str, List]] = None):
    """Generate an OOD train test split, where both the condition data and the single-cells have approximately
    the specified split.
    
    Rules: 
    1. cell split and condition split is similar (conditions is exactly the split, 
    cells is approximate by deviation_thresh)
    2. Each component (condition_cols) of the condition is seen atleast once in the training
    3. Each of the stimulation column is seen atleast once in the test
    4. The test context (cell type) needs to contain atleast one from context_frac of the total context_bins 

    Parameters
    ----------
    tf_adata : 
        AnnData object of TF activity
    train_frac : float
        the fraction of the data going to training
    stim_col : List[str]
        the metadata column defining the stimulation condition
    context_col: str
        the metadata column of the context, together with stim col defines the OOD split
    context_bins: Optional[pd.DataFrame]
        binning of the context categories, by default None
    context_bins_frac: float
        the number of bins that should be included in the test data from the context bins
    max_iter : int, optional
        total iterations to try to identify a suitable split, by default 1000
    seed : int, optional
        random state variable, by default 888
    deviation_thresh : float, optional
        extent to which to allow the single-cell split to deviate from the specified `train_frac split`, by default 0.025
    include_train_cond : Optional[Dict[str, List]]
        a dictionary with keys for each of stim_col and context_col and keys as a list that represents a subset
        of values in that column that must be included in the training set. 
    """
    test_frac = 1 - train_frac

    condition_cols = [stim_col, context_col]
    condition_combs = tf_adata.obs[condition_cols].apply(lambda row: '^'.join(row.astype(str)), axis=1)
    unique_conditions = condition_combs.drop_duplicates(keep = 'first', inplace = False).tolist()

    # define the conditions for stopping
    train_frac_deviation = np.inf
    zero_shot_bool = False

    test_stim = False
    n_stims = tf_adata.obs['stim'].nunique()

    if context_bins is not None:
        if context_bins_frac > 1 or context_bins_frac < 0:
            raise ValueError('The fraction must be between 0 and 1')
        n_contexts = np.floor(context_bins_frac*context_bins.nunique())
        context_present = False
    else:
        context_present = True

    if include_train_cond is not None:
        if sorted(set(include_train_cond).intersection(condition_cols)) != sorted(condition_cols):
            raise ValueError('Keys for `include_train_cond` must be the same as `condition_cols`')
        for cond, cond_vals in include_train_cond.items():
            if not set(cond_vals).issubset(tf_adata.obs[cond].tolist()):
                raise ValueError('The conditions to include in training for ' + cond + ' are not present in the metadata.')
        ict_bool = False
    else:
        ict_bool = True

    counter = 0
    while (train_frac_deviation > deviation_thresh or not zero_shot_bool or not ict_bool or not test_stim or not context_present) and (counter < max_iter):
        # ood at categorical level
        train_cond, test_cond = train_test_split(unique_conditions, test_size = test_frac, random_state = seed + counter, shuffle = True)
        train_cells = condition_combs[condition_combs.isin(train_cond)].index.tolist()
        test_cells = condition_combs[condition_combs.isin(test_cond)].index.tolist()

        train_frac_actual = len(train_cells)/tf_adata.shape[0]
        train_frac_deviation = abs(train_frac_actual - train_frac)

        # ensure that all individual conditions are seen atleast once (it's the combination of conditions that's unique)
        train_cond_map = {}
        for idx, cond in enumerate(condition_cols):
            train_cond_map[cond] = {tc.split('^')[idx] for tc in train_cond}

        test_cond_map = {}
        for idx, cond in enumerate(condition_cols):
            test_cond_map[cond] = {tc.split('^')[idx] for tc in test_cond}

        zero_shots = [len(test_cond_map[cond].difference(train_cond_map[cond])) for cond in condition_cols]    
        zero_shot_bool = all(value == 0 for value in zero_shots)

        test_stim = (len(test_cond_map['stim']) == n_stims)


        if context_bins is not None:
            context_present = context_bins.loc[list(test_cond_map[context_col])].nunique() >= n_contexts

        if include_train_cond is not None:
            ict_conds = [len(include_train_cond[cond].difference(cond_vals)) for cond, cond_vals in train_cond_map.items()]
            ict_bool = any(value == 0 for value in ict_conds)

        counter += 1

    if counter < max_iter:
        return train_cells, test_cells, train_cond, test_cond
    else:
        return None, None, None, None


# In[22]:


from typing import Literal
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

def adata_dimviz(adata, reduction_type, cats):
    viz_df = pd.DataFrame(adata.obsm['X_' + reduction_type])
    for cat in cats:
        viz_df = pd.concat([viz_df, pd.DataFrame(adata.obs[cat]).reset_index(drop = True)], ignore_index = True, axis = 1)
    if reduction_type=='umap':
        viz_df.columns = [reduction_type.upper() + str(i+1) for i in range(viz_df.shape[1])]
    elif reduction_type=='pca':
        viz_df.columns = [reduction_type.upper()[:-1] + str(i+1) for i in range(viz_df.shape[1])]
    viz_df.columns = viz_df.columns[:-len(cats)].tolist() + cats
    
    return viz_df


# In[23]:


contingency_table = pd.crosstab(tf_adata.obs['stim'], tf_adata.obs['seurat_annotations'], 
                                margins=True, margins_name="Total")
contingency_table = contingency_table.T.sort_values(by = 'Total').T
bins = pd.qcut(contingency_table.T.Total, q = 4, labels = False)


# In[24]:


seed = 1
max_iter = 1000

folds = 10

train_frac, test_frac = 0.8, 0.2
condition_cols = ['stim', 'seurat_annotations']

best_resolution = tf_adata.uns['leiden']['params']['resolution']
calculation_type = 'project' # project data rather than embed
n_neighbors = 15

unique_conditions = sorted(set(tf_adata.obs.condition))

if os.path.isfile(os.path.join(data_path, 'trash', 'res_all.csv')):
    res_all = pd.read_csv(os.path.join(data_path, 'trash', 'res_all.csv'), index_col = 0)
else:
    res_all = pd.DataFrame(columns = ['fold', 'seed', 'train_conditions', 'test_conditions',
                              'NMI_global_default', 'NMI_categorical_default', 
                                 'NMI_global_dev', 'NMI_categorical_dev'])


# In[ ]:


unique_conditions.remove('CTRL^DC')
unique_conditions.insert(0, 'CTRL^DC')


# In[ ]:


for (k, test_cond) in enumerate(unique_conditions):
    seed = (k*max_iter)
    
    tf_adata = io.read_tfad(os.path.join(data_path, 'processed', 'Kang_tf_activity.h5ad'))
    tf_adata.obs['condition'] = tf_adata.obs['stim'].astype(str) + '^' + tf_adata.obs['seurat_annotations'].astype(str)
    tf_adata = tf_adata[:, sorted(tf_adata.var_names)] 
    
    adata = sc.read_h5ad(os.path.join(data_path, 'processed', 'kang_expr_scored.h5ad'))
    sn_ppis = pd.read_csv(os.path.join(data_path, 'processed', 'Kang_sn_ppis.csv'), index_col = 0)
    
    # split data
#     train_cells, test_cells, train_cond, test_cond = ood_split(tf_adata,
#                                                                        train_frac = 0.8,
#                                                                        stim_col = condition_cols[0], 
#                                                                        context_col = condition_cols[1], 
#                                                                        context_bins = bins, 
#                                                                        context_bins_frac = 1, 
#                                                                        max_iter = max_iter, 
#                                                                        seed = seed, 
#                                                                        deviation_thresh = 0.025, 
#                                                                        include_train_cond = None)
#     test_conds = test_cond


    test_cond = [test_cond]
    fn_ = os.path.join(data_path, 'trash', test_cond[0])
    train_cond = set(unique_conditions).difference(test_cond)
    
    test_conds = test_cond
    
    test_cells = tf_adata.obs[tf_adata.obs.condition.isin(test_cond)].index.tolist()
    train_cells = tf_adata.obs[tf_adata.obs.condition.isin(train_cond)].index.tolist()

    res_all.loc[test_cond[0], ['fold', 'seed', 'train_conditions', 'test_conditions']] = pd.Series([k, seed, train_cond, test_cond], 
                                                                                                   index = ['fold', 'seed', 'train_conditions', 'test_conditions'],
                                                                                                   dtype = object)
 
    # build model
    mod_ = SignalingModel(net = sn_ppis,
                     X_in = pd.DataFrame(tf_adata.obs.stim.cat.codes, columns = ['IFNB1']),
                     y_out = tf_adata.to_df().copy(), 
                     expr = adata.to_df().copy(), 
                     covariates = tf_adata.obs.copy(),
                     categorical_covariate_keys = ['seurat_annotations'],
                     projection_amplitude_in = projection_amplitude_in, 
                     projection_amplitude_out = projection_amplitude_out,
                     weight_label = weight_label, source_label = source_label, target_label = target_label,
                     bionet_params = bionet_params, 
                     dtype = torch.float32, device = device, seed = seed)

    mod_.input_layer.weights.requires_grad = False # don't learn scaling factors for the ligand input concentrations
    mod_.signaling_network.prescale_weights(target_radius = target_spectral_radius) # spectral radius
    

    mod_dev = SignalingModelDev(net = sn_ppis,
                     X_in = pd.DataFrame(tf_adata.obs.stim.cat.codes, columns = ['IFNB1']),
                     y_out = tf_adata.to_df().copy(), 
                     expr = adata.to_df().copy(), 
                     covariates = tf_adata.obs.copy(),
                     categorical_covariate_keys = ['seurat_annotations'],
                     projection_amplitude_in = projection_amplitude_in, 
                     projection_amplitude_out = projection_amplitude_out,
                     weight_label = weight_label, source_label = source_label, target_label = target_label,
                     bionet_params = bionet_params_dev, 
                     dtype = torch.float32, device = device, seed = seed)

    mod_dev.input_layer.weights.requires_grad = False # don't learn scaling factors for the ligand input concentrations
    mod_dev.signaling_network.prescale_weights(target_radius = target_spectral_radius) # spectral radius
    
    # train model
    mod_res = {}
    mods = {'default': mod_, 
           'dev': mod_dev}
    for mod_type, mod in mods.items():
        if not os.path.isfile(fn_ + '_' + mod_type + '.png'):
            print(test_cond[0] + '_' + mod_type)

            trainer = TrainSC(mod = mod,
                               prediction_optimizer = torch.optim.Adam,
                               prediction_loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05).to(device), #torch.nn.MSELoss(reduction='mean'),
                              discriminator_params = discriminator_params,
                               hyper_params = training_params,
                               train_split = {'train': train_cells, 'test': test_cells, 'validation': None}, 
                               train_seed = seed, 
                               track_test = False,
                               track_validation = False)
            mod_res[mod_type] = trainer.train_model(verbose = False)
            io.write_pickled_object(trainer,
                                    '_'.join([fn_, mod_type, 'trainer.pickle']))

            # get global bias quant
            train_stats_df = trainer.stats['train'].copy()
            train_stats_df = train_stats_df.groupby('epoch').mean().reset_index() # delete this

            cf_map = {'in_distribution': train_cells}
            counterfactual_types = list(cf_map.keys()) + ['opposite']

            biases_res = {}
            for counterfactual_type in ['opposite']:#, 'in_distribution']:
                biases_res[counterfactual_type] = {}
                biases = get_prediction(mod = mod, 
                                                             tf_adata = tf_adata, 
                                                             counterfactual_type = counterfactual_type, 
                                                             cf_map = cf_map, 
                                                             train_cells_all = train_cells, 
                                                             test_conds = test_conds, 
                                                             return_bias = True)
                bias_global, bias_mu, bias_sigma, bias_cats, bias_tot, obs = biases

                biases_res[counterfactual_type]['adverserial'] = (bias_global, bias_mu, bias_sigma, bias_cats, bias_tot)
                biases_res[counterfactual_type]['obs'] = obs
                del biases
                torch.cuda.empty_cache()

            biases_clustered = {}
            for counterfactual_type, br in biases_res.items():
                print(counterfactual_type)
                bias_global, _, _, _, bias_tot = br['adverserial']
                obs = br['obs']

                # full model
                bias_adata = sc.AnnData(X = bias_global.detach().cpu().numpy(), obs = obs)
                embed_tf_activity(bias_adata, scanpy_pca = False, cluster_col_name = 'leiden', resolution = 1)

                # full model -- categorical information added
                bias_tot = sc.AnnData(X = bias_tot.detach().cpu().numpy().T, obs = obs)
                embed_tf_activity(bias_tot, scanpy_pca = False, cluster_col_name = 'leiden', resolution = 1)

                biases_clustered[counterfactual_type] = (bias_adata, bias_tot)

            counterfactual_type = 'opposite'
            bias_adata, bias_tot = biases_clustered[counterfactual_type]

            nmis = []
            for adata in [bias_adata,bias_tot]:
                nmis.append(normalized_mutual_info_score(adata.obs.leiden, adata.obs.seurat_annotations))

            res_all.loc[test_cond[0], 
                        [nmi_col + '_' + mod_type for nmi_col in ['NMI_global', 'NMI_categorical']]] = nmis
            res_all.to_csv(os.path.join(data_path, 'trash', 'res_all.csv'))

            tf_res = {}
            for counterfactual_type in ['opposite']: #counterfactual_types:
                tf_adata_predicted = get_prediction(mod = mod, 
                                                    tf_adata = tf_adata, 
                                                    counterfactual_type = counterfactual_type, 
                                                    cf_map = cf_map, 
                                                    train_cells_all = train_cells, 
                                                    test_conds = test_conds)
        #         tf_adata_base = make_baseline(tf_adata, tf_adata_predicted) # emulate predicted values, but with actual test values

                tf_adata_predicted = prepare_for_metrics(tf_adata, 
                                                   tf_adata_predicted, 
                                                   resolution = best_resolution,
                                                   calculation_type = calculation_type, 
                                                  n_neighbors = n_neighbors)
        #         nmi, as_OOD, as_ID = get_metrics(tf_adata_predicted, 
        #                                          train_cells = train_cells, 
        #                                          n_neighbors = n_neighbors)

        #         tf_adata_base = prepare_for_metrics(tf_adata, 
        #                                        tf_adata_base, 
        #                                        resolution = best_resolution,
        #                                        calculation_type = calculation_type, 
        #                                       n_neighbors = n_neighbors)
        #         baseline_nmi, baseline_as_OOD, baseline_as_ID = get_metrics(tf_adata_base, 
        #                                                                     train_cells = train_cells, 
        #                                                                     n_neighbors = n_neighbors)


                tf_res[counterfactual_type] = tf_adata_predicted


    #         res.loc[res.shape[0], :] = [calculation_type, counterfactual_type, 
    #                                     nmi, as_OOD, as_ID, 
    #                                    baseline_nmi, baseline_as_OOD, baseline_as_ID]
# visualize first 2 pcs
#             fig, ax = plt.subplots(ncols = 1, figsize = (5, 5))
#             counterfactual_type_title = {'in_distribution': 'In Distribution', 'opposite': 'Opposite'}
#             title_coords = [0.92, 0.47]

#             subset_size = 1000

#             cell_types = [tc.split('^')[1] for tc in test_conds]
#             len(cell_types)
#             counterfactual_type = 'opposite'
#             tf_adata_all = tf_res[counterfactual_type]
#             # for i, (counterfactual_type, tf_adata_all) in enumerate(tf_res.items()):

#             tf_adata_viz = tf_adata_all[tf_adata_all.obs[tf_adata_all.obs.seurat_annotations.isin(cell_types)].index.tolist(),:].copy()

#             tf_adata_viz = tf_adata_viz.copy()  
#             tf_adata_viz.obs['condition'] = (
#                 tf_adata_viz.obs['stim'] + '^' + 
#                 tf_adata_viz.obs['seurat_annotations'] + '^' + 
#                 tf_adata_viz.obs['batch']
#             )

#             np.random.seed(seed)
#             tf_adata_viz = tf_adata_viz[np.random.permutation(tf_adata_viz.obs_names), :]

#             viz_df = adata_dimviz(adata = tf_adata_viz, reduction_type = 'pca', cats = ['seurat_annotations', 'batch', 'stim', 'condition'])
#             viz_df.seurat_annotations = pd.Categorical(viz_df.seurat_annotations, 
#                                                       categories = cell_types, ordered=True)

#             # only 1000 cells to visualize
#             vc = viz_df.condition.value_counts()
#             subset_conds = vc[vc > subset_size].index.tolist()

#             drop_idx = []
#             for subset_cond in subset_conds: 
#                 np.random.seed(seed)
#                 drop_idx += list(np.random.choice(viz_df[viz_df.condition == subset_cond].index, 
#                                  vc[subset_cond] - subset_size, 
#                                 replace = False))
#             viz_df.drop(index = drop_idx, inplace = True)

#             cell_type = cell_types[0]
#             viz_df_ = viz_df[viz_df.seurat_annotations == cell_type]

#             stim_pred = viz_df_[viz_df_.batch == 'predicted'].stim.iloc[0]
#             order = [stim_pred + '^' + cell_type + '^' + 'predicted', 
#                     stim_pred + '^' + cell_type + '^' + 'actual', 
#                     rev_stim[stim_pred] + '^' + cell_type + '^' + 'actual']
#             viz_df_ = viz_df_.copy()
#             viz_df_.condition = pd.Categorical(viz_df_.condition, 
#                                                   categories = order, ordered=True)

#             np.random.seed(seed)
#             viz_df_ = viz_df_.loc[np.random.permutation(viz_df_.index),:]
#             sns.scatterplot(data = viz_df_, x = 'PC1', y = 'PC2', hue = 'condition', 
#                     s=10, ax = ax)
#             ax.set_title(cell_type)

#             fig.tight_layout(rect=[0,0,1,0.9])
            # visuzlie first 5 pcs
            n_pcs = 5
            pc_combs = list(itertools.combinations(range(1, n_pcs + 1), 2))

            ncols = 5
            nrows = int(np.ceil(len(pc_combs)/ncols))

            fig, axes = plt.subplots(ncols = ncols, nrows = nrows, figsize = (5.1*ncols, 5.1*nrows))
            ax = axes.flatten()

            counterfactual_type_title = {'in_distribution': 'In Distribution', 'opposite': 'Opposite'}
            title_coords = [0.92, 0.47]

            subset_size = 1000

            cell_types = [tc.split('^')[1] for tc in test_conds]
            len(cell_types)
            counterfactual_type = 'opposite'
            tf_adata_all = tf_adata_predicted
            # for i, (counterfactual_type, tf_adata_all) in enumerate(tf_res.items()):

            tf_adata_viz = tf_adata_all[tf_adata_all.obs[tf_adata_all.obs.seurat_annotations.isin(cell_types)].index.tolist(),:].copy()

            tf_adata_viz = tf_adata_viz.copy()  
            tf_adata_viz.obs['condition'] = (
                tf_adata_viz.obs['stim'] + '^' + 
                tf_adata_viz.obs['seurat_annotations'] + '^' + 
                tf_adata_viz.obs['batch']
            )

            np.random.seed(seed)
            tf_adata_viz = tf_adata_viz[np.random.permutation(tf_adata_viz.obs_names), :]

            viz_df = adata_dimviz(adata = tf_adata_viz, reduction_type = 'pca', cats = ['seurat_annotations', 'batch', 'stim', 'condition'])
            viz_df.seurat_annotations = pd.Categorical(viz_df.seurat_annotations, 
                                                      categories = cell_types, ordered=True)

            # only 1000 cells to visualize
            vc = viz_df.condition.value_counts()
            subset_conds = vc[vc > subset_size].index.tolist()

            drop_idx = []
            for subset_cond in subset_conds: 
                np.random.seed(seed)
                drop_idx += list(np.random.choice(viz_df[viz_df.condition == subset_cond].index, 
                                 vc[subset_cond] - subset_size, 
                                replace = False))
            viz_df.drop(index = drop_idx, inplace = True)

            cell_type = cell_types[0]
            viz_df_ = viz_df[viz_df.seurat_annotations == cell_type]

            stim_pred = viz_df_[viz_df_.batch == 'predicted'].stim.iloc[0]
            order = [stim_pred + '^' + cell_type + '^' + 'predicted', 
                    stim_pred + '^' + cell_type + '^' + 'actual', 
                    rev_stim[stim_pred] + '^' + cell_type + '^' + 'actual']
            viz_df_ = viz_df_.copy()
            viz_df_.condition = pd.Categorical(viz_df_.condition, 
                                                  categories = order, ordered=True)

            np.random.seed(seed)
            viz_df_ = viz_df_.loc[np.random.permutation(viz_df_.index),:]

            for i,comb in enumerate(pc_combs):
                sns.scatterplot(data = viz_df_, 
                                x = 'PC{}'.format(comb[0]), 
                                y = 'PC{}'.format(comb[1]), hue = 'condition', 
                        s=10, ax = ax[i])
                ax[i].set_title(cell_type)

            fig.tight_layout()
            plt.savefig(fn_ + '_' + mod_type + '.png', dpi=300, bbox_inches='tight')

    


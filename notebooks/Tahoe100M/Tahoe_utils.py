#!/usr/bin/env python
# coding: utf-8

import random
from collections import defaultdict
import os
from typing import Literal, List
import joblib

from tqdm import tqdm, trange


import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression 
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
import scipy

import torch
from geomloss import SamplesLoss

import sys
sclembas_path = '/home/hmbaghda/Projects/scLEMBAS'
sys.path.insert(1, os.path.join(sclembas_path))

from scLEMBAS import utilities as utils
from scLEMBAS import preprocess as pp
from scLEMBAS import latent_separation as ls


device = "cuda" if torch.cuda.is_available() else "cpu"


def Tahoe100M_split(tf_adata, 
                   train_frac: float = 0.8, 
                   min_drug_frac: float = 0.6, 
                   min_cell_line_frac: float = 0.6, 
                    exclude_control: bool = False,
                   max_attempts: int = 1000, 
                   seed: int = 888 
                   ):
    """Create a train-test split for the Tahoe 100M data. 
    Splits by condition (drug + cell line), which leads to even splits in barcodes given balanced classes. 
    Also ensures a minimum fraction of total of each drug / cell line in conditions is in the train split 
    (this is separate and less than the condition split). 

    Parameters
    ----------
    tf_adata : _type_
        _description_
    train_frac : float, optional
        fraction of conditions to split into train, by default 0.8
    min_drug_frac : float, optional
        minimum fraction of conditions in train that should include each drug, by default 0.6
    min_cell_line_frac : float, optional
        minimum fraction of conditions in train that should include each cell line, by default 0.6
    exclude_control : bool, optional
        whether to ensure all control conditions are in the train (True) or not (False), by default False
    max_attempts : int, optional
        maximum number of tries to achieve a split that meets these requirements, by default 1000
    seed : int, optional
        random state, by default 888
    """
    
    if min_drug_frac > train_frac or min_cell_line_frac > train_frac:
        raise ValueError('The minimum perturbation/cell line fraction must be less than or equal to the train fraction')
    
    obs = tf_adata.obs.copy()
    unique_conditions = obs[['cell_line', 'drug', 'condition']].drop_duplicates().reset_index(drop = True)
    if unique_conditions.cell_line.value_counts().nunique() != 1 or unique_conditions.drug.value_counts().nunique() != 1:
        raise ValueError('This function was written expecting even drug and cell line counts')
        
    tot_drug_counts = unique_conditions.drug.value_counts().min()    
    min_train_count_drug = int(np.round(min_drug_frac*tot_drug_counts))
    min_train_count_cell_line = int(np.round(min_cell_line_frac*unique_conditions.cell_line.value_counts().min()))

    rng = np.random.default_rng(seed)
    
    test_frac = 1 - train_frac
    # used to calculate minimum train thresholds
    n_drugs = obs['drug'].nunique()
    n_cell_lines = obs['cell_line'].nunique()
    n_conds = obs['condition'].nunique()
    n_test = int(np.round(n_conds * test_frac))


    for attempt in range(max_attempts):
        # shuffle conditions
        unique_conditions = unique_conditions.sample(frac=1.0, random_state=rng.integers(0, 1e6)).reset_index(drop=True)


        test_conds = unique_conditions.iloc[:n_test]
        train_conds = unique_conditions.iloc[n_test:]

        train_drug_counts = train_conds['drug'].value_counts()
        train_cell_line_counts = train_conds['cell_line'].value_counts()

        a = np.all(train_drug_counts > min_train_count_drug)
        b = np.all(train_cell_line_counts > min_train_count_cell_line)
        c = True
        if exclude_control:
            c = (train_drug_counts['DMSO_TF'] == tot_drug_counts)
            

        if (a and b and c):
            break

    if attempt == max_attempts - 1:
        raise ValueError('Maximum attempts reached')
    
    train_split = {'conditions': train_conds, 
                  'drug_counts': train_drug_counts,
                  'cell_line_counts': train_cell_line_counts, 
                  'barcodes': obs[obs.condition.isin(train_conds.condition)].index.tolist()}
    
    test_split = {'conditions': test_conds, 
                 'drug_counts': test_conds['drug'].value_counts(), 
                 'cell_line_counts': test_conds['cell_line'].value_counts(), 
                 'barcodes': obs[obs.condition.isin(test_conds.condition)].index.tolist()}

    
    if len(set(train_conds.condition).intersection(test_conds.condition)) != 0:
        raise ValueError('Train and test conditions overlap')
    
    return train_split, test_split


def Tahoe100M_kfold(tf_adata, 
                    n_splits: int = 5,
                    min_drug_frac: float = 0.6,
                    min_cell_line_frac: float = 0.6,
                    exclude_control: bool = False, 
                    max_attempts: int = 1000,
                    seed: int = 888):
    """Creates k-fold cross-validation splits in the fashion of `Tahoe100M_split`.
    
    Splits by condition (drug + cell line), which leads to even splits in barcodes given balanced classes. 
    Also ensures a minimum fraction of total of each drug / cell line in conditions is in the train split 
    (this is separate and less than the condition split). 


    Parameters
    ----------
    tf_adata : _type_
        _description_
    n_splits : float, optional
        k-splits to create
    min_drug_frac : float, optional
        minimum fraction of conditions in train that should include each drug, by default 0.6
    min_cell_line_frac : float, optional
        minimum fraction of conditions in train that should include each cell line, by default 0.6
    exclude_control : bool, optional
        whether to ensure all control conditions are in the train (True) or not (False), by default False
    max_attempts : int, optional
        maximum number of tries to achieve a split that meets these requirements, by default 1000
    seed : int, optional
        random state, by default 888
    """
    
    train_frac = 1 - (1/n_splits)

    if min_drug_frac > train_frac or min_cell_line_frac > train_frac:
        raise ValueError('The minimum perturbation/cell line fraction must be less than or equal to the train fraction')
    
    obs = tf_adata.obs.copy()
    unique_conditions = obs[['cell_line', 'drug', 'condition']].drop_duplicates().reset_index(drop = True)
    if unique_conditions.cell_line.value_counts().nunique() != 1 or unique_conditions.drug.value_counts().nunique() != 1:
        raise ValueError('This function was written expecting even drug and cell line counts')
        
    tot_drug_counts = unique_conditions.drug.value_counts().min()    
    min_train_count_drug = int(np.round(min_drug_frac*tot_drug_counts))
    min_train_count_cell_line = int(np.round(min_cell_line_frac*unique_conditions.cell_line.value_counts().min()))

    rng = np.random.default_rng(seed)
    
    test_frac = 1 - train_frac
    # used to calculate minimum train thresholds
    n_drugs = obs['drug'].nunique()
    n_cell_lines = obs['cell_line'].nunique()
    n_conds = obs['condition'].nunique()
    n_test = int(np.round(n_conds * test_frac))
    


    for attempt in trange(max_attempts):
        folds = []
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=rng.integers(1e6))

        for train_idx, test_idx in kf.split(unique_conditions):
            train_conds = unique_conditions.iloc[train_idx]
            test_conds = unique_conditions.iloc[test_idx]
            # shuffle conditions
            unique_conditions = unique_conditions.sample(frac=1.0, random_state=rng.integers(0, 1e6)).reset_index(drop=True)


            test_conds = unique_conditions.iloc[:n_test]
            train_conds = unique_conditions.iloc[n_test:]

            train_drug_counts = train_conds['drug'].value_counts()
            train_cell_line_counts = train_conds['cell_line'].value_counts()

            a = np.all(train_drug_counts > min_train_count_drug)
            b = np.all(train_cell_line_counts > min_train_count_cell_line)
            c = True
            if exclude_control:
                c = (train_drug_counts['DMSO_TF'] == tot_drug_counts)


            if (a and b and c):
                folds.append({'train': train_conds, 
                             'test': test_conds})

        if len(folds) == n_splits:
            break
    if attempt == max_attempts - 1:
        raise ValueError('Maximum attempts reached')
        
    train_splits = []
    test_splits = []
    for k in folds:
        for split_type, conds in k.items():
            items = {'conditions': conds, 
                     'drug_counts': conds['drug'].value_counts(),
                     'cell_line_counts': conds['cell_line'].value_counts(), 
                     'barcodes': obs[obs.condition.isin(conds.condition)].index.tolist()}

            if split_type == 'train':
                train_splits.append(items)
            else:
                test_splits.append(items)

        

    return train_splits, test_splits

# ----------------------------------------USAGE----------------------------------------

# train_split, test_split = Tahoe100M_split(tf_adata,
#                                           train_frac = 0.9, 
#                                           min_drug_frac = 0.7, 
#                                           min_cell_line_frac = 0.7,
#                                           exclude_control = True,
#                                           max_attempts = 1000, 
#                                           seed = 888)

# train_splits, test_splits = Tahoe100M_kfold(tf_adata, 
#                                             n_splits=10, 
#                                             min_drug_frac = 0.6, 
#                                             min_cell_line_frac = 0.6, 
#                                           exclude_control = True,
#                                             max_attempts = 1000, 
#                                             seed = 888)
# ----------------------------------------USAGE----------------------------------------

def match_test_cell_lines(train_conds, test_conds, seed=888):
    """
    Matches each test cell line to a different train cell line, such that
    for every test condition <test_cell^perturbation>, the matched train
    cell line also has the same perturbation in the training conditions.
    
    For use in counterfactual predictions across cell lines. 

    Parameters
    ----------
    train_conds : list of str
        List of training conditions in the format "<cell_line>^<perturbation>".

    test_conds : list of str
        List of test conditions in the format "<cell_line>^<perturbation>".

    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        A dictionary mapping each test cell line (str) to a matched train
        cell line (str) that has the same perturbation in the training data.
    """


    # mapping of perturbations to train cells
    pert_to_train_cells = defaultdict(set)
    for train_cond in train_conds:
        train_cell, train_pert = train_cond.split('^')
        pert_to_train_cells[train_pert].add(train_cell)

    # mapping of trest cells to train cells that have the same pertrubation
    test_to_train_map = {}
    if seed is not None:
        random.seed(seed)
    for test_cond in test_conds:
        test_cell, test_pert = test_cond.split('^')
        eligible_train_cells = pert_to_train_cells[test_pert].difference(test_cell)
        test_to_train_map[test_cond] = random.choice(list(eligible_train_cells))

    return test_to_train_map


def setup_prediction(mod, 
                     train_cells,
                     test_cells,
                     tf_adata, 
                     cat_col: str, 
                     pert_col: str, 
                     ctrl_pert: str, 
                     counterfactual: Literal['perturbation', 'category', None] = 'perturbation',
                     cat_counterfactual_map: dict = None,
                     ):
    
    """Creates model inputs for running a prediction with counterfactuals

    Parameters
    ----------
    mod :
        initialized scLEMBAS model
    train_cells : 
        the barcodes for cells used in training
    test_cells : 
        the barcodes for cells assigned to test
    tf_adata : 
        TF AnnData object
    cat_col : str
        the categorical column in `obs`
    pert_col : str
        the perturbation column in `obs`
    ctrl_pert : str
        the control perturbation in `obs[pert_col]`
    counterfactual : Literal['perturbation', 'category', None], optional
        currently the counterfactual will only cross between perturbations or category, keeping the 
        other constant (e.g., within a cell type b/w perturbations or within a perturbation b/w cell types)
        by default 'perturbation'
        - 'perturbation': predicts the test condition from the control pertrubation + same category in train
        - 'category': predicts the test condition from a different cell line + the same perturbation in train.
                      the different cell line is specified by `cat_counterfactual_map`
        - None: iterates through train conditions and predicts without a counterfactual
    cat_counterfactual_map : dict, optional
        a dictionary mapping each test condition to a  different cell line in 
        the train data that has the same perturbation from which to ask the 'category' counterfactual
        only needed when setting `counterfactual`  = 'category', by default None
    """

    if counterfactual == 'category' and cat_counterfactual_map is None:
        raise ValueError('Need to specify a mapping from test to train cells for counterfactual across category')
        
    print('Set up inputs for prediction')

    train_conds = tf_adata.obs.loc[train_cells, 'condition'].unique()
    test_conds = tf_adata.obs.loc[test_cells, 'condition'].unique()

    cov_idx_map = dict(zip(
        mod.signaling_network.covariates[cat_col],
        mod.signaling_network.covariates_idx[cat_col]
    ))
    cov_rev_map = {v:k for k,v in cov_idx_map.items()}

    pert_columns = tf_adata.obs[pert_col].cat.categories.tolist()
    ctrl_idx = pert_columns.index(ctrl_pert)

    full_expr, full_X, full_covariates = None, None, None
    
    iterable_conds = test_conds if counterfactual is not None else train_conds
    
    counterfactual_cond_map = {} # map from control to prediction
    
    plates = []
    for cond in tqdm(sorted(iterable_conds)):
        ct, pert = cond.split('^')

        if counterfactual == 'perturbation': # predict test perturbation from ctrl pert of same cell line 
            ctrl_cond = ct + '^' + ctrl_pert
        elif counterfactual == 'category':
            ctrl_cond = cat_counterfactual_map[cond] + '^' + pert
        elif counterfactual == None: # no counterfactual, just predict the same thing
            ctrl_cond = cond
            
        counterfactual_cond_map[cond] = ctrl_cond

        if not ctrl_cond in train_conds:
            raise ValueError('The control condition {} for asking the counterfactual is not in the train conditions'.format(ct))

        # get the input g (counterfactual)
        predict_cells_from = tf_adata[train_cells, :].obs.condition == ctrl_cond
        predict_cells_from = predict_cells_from[predict_cells_from].index.tolist()
        expr_in = mod.df_to_tensor(mod.expr.loc[predict_cells_from, :])  
        # record the corresponding plate for running PLSR if needed
        plates += tf_adata.obs.loc[predict_cells_from, 'plate'].tolist()
        

        n_predictions = expr_in.shape[0]

        X_in = pd.Categorical([pert]*n_predictions,
                              categories = pert_columns)
        X_in = pd.get_dummies(X_in).astype(int)
        X_in.drop(columns = ctrl_pert, inplace = True)
        X_in = mod.df_to_tensor(X_in)

        covariates_in = torch.tensor([cov_idx_map[ct]]*n_predictions,
                                        device = mod.device, dtype = torch.int64).view(-1,1)

        full_expr = expr_in if full_expr is None else torch.cat((full_expr, expr_in), dim = 0)
        full_X = X_in if full_X is None else torch.cat((full_X, X_in), dim = 0)
        full_covariates = covariates_in if full_covariates is None else torch.cat((full_covariates, covariates_in), dim = 0)

        utils.clear_memory()

        # format metadata
        obs = pd.DataFrame(full_covariates.detach().cpu().numpy())
        obs.columns = [cat_col]
        obs[cat_col] = obs[cat_col].map(cov_rev_map)

        pert_vals = pd.DataFrame(full_X.detach().cpu().numpy(), 
                                columns = [i for i in pert_columns if i != ctrl_pert])
        pert_vals.insert(ctrl_idx, ctrl_pert, 0.0)
        obs[pert_col] = pert_vals.idxmax(axis=1)

        # add in ctrl pertrubation
        ctrl_cells = pert_vals[pert_vals.sum(axis = 1) == 0].index.tolist()
        obs.loc[ctrl_cells, pert_col] = ctrl_pert

    obs['condition'] = obs[cat_col].astype(str) + '^' + obs[pert_col].astype(str)
    obs['counterfactual_condition'] = obs.condition.map(counterfactual_cond_map)
    obs['plate'] = plates
    
    for col in obs:
        if col in tf_adata.obs and isinstance(tf_adata.obs[col].dtype, pd.CategoricalDtype):
            obs[col] = pd.Categorical(obs[col], 
                             categories = tf_adata.obs[col].cat.categories)
            obs[col] = obs[col].cat.remove_unused_categories()

    unique_conds = sorted(obs.condition.unique())

    if counterfactual is not None:
        if unique_conds != sorted(test_conds):
            raise ValueError("Something went wrong in adding test conditions")
    else:
        if unique_conds != sorted(train_conds):
            raise ValueError("Something went wrong in adding train conditions")

    return full_expr, full_X, full_covariates, obs


def run_prediction(mod, 
                   remove_type: Literal['none', 'adj', 'categorical_bias', 'global_bias'], 
                   return_bias: bool, 
                   X_in: torch.tensor, 
                   covariates_idx: torch.tensor, 
                   expr: torch.tensor, 
                   obs: pd.DataFrame, 
                   return_full: bool = False):
    """Gets the model prediction.

    Parameters
    ----------
    mod : 
        trained scLEMBAS model
    remove_type : Literal['none', 'adj', 'categorical_bias', 'global_bias', 'total_bias], optional
        can be a string or a list of strings
        which components of bias/adj matrix to remove when running the prediction, by default 'none'; 
        only incorporated if `return_bias` = False
        any bias component includes the full adjacency matrix
        - 'none': includes all components in the prediction
        - 'categorical_bias': includes global but excludes categorical bias in the prediction
        - 'global_bias': includes categorical but excludes global bias in the prediction
        - 'total_bias': does not include bias in the prediction (just input and signaling weights)
        - 'adj': includes all bias but sets signaling weights to 0
        the only list of strings are combining either categorical or global bias with adj, since in these cases just removing, by default 'none'
    return_bias : bool, optional
        whether to return bias terms (True) or prediction (False), by default False
    X_in : torch.tensor
        input to model forward pass, generated by `setup_prediction`
    covariates_idx : torch.tensor
        input to model forward pass, generated by `setup_prediction`
    expr : torch.tensor
        input to model forward pass, generated by `setup_prediction`
    obs : pd.DataFrame
        input to model forward pass, generated by `setup_prediction`
    return_full : bool, optional
        whether to return model output prior to ProjectOutput transformation (True) or after (False), by default False

    Returns
    -------
    tf_adata_predicted
        AnnData object of the predicted TF activity

    """

    # ----------------CHECKS----------------
    if type(remove_type) != list:
        remove_type = [remove_type]
    if len(remove_type) not in [1,2]:
        raise ValueError('Cannot remove more than two components at once')
    if len(set(remove_type).difference(['none', 'global_bias', 'categorical_bias', 'total_bias', 'adj'])) > 0:
        raise ValueError('Incorrect remove_type specified')
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

        if return_bias:
            bias_tot = bias_global.T + bias_cats
            bias_sigma = torch.exp(bias_log_sigma_squared/2.) + mod.signaling_network.vae.var_min
            utils.clear_memory()
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
        
        utils.clear_memory()
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

    utils.clear_memory()

    return tf_adata_predicted


def get_prediction(mod,
                   train_cells,
                   test_cells, 
                   tf_adata,
                     cat_col: str, 
                     pert_col: str, 
                     ctrl_pert: str, 
                     counterfactual: Literal['perturbation', 'category', None] = 'perturbation',
                     cat_counterfactual_map: dict = None,
                   remove_type: Literal['none', 'global_bias', 'categorical_bias', 'total_bias', 'adj'] = 'none',
                   return_bias: bool = False, 
                  max_cells = None, 
                  return_full: bool = False):
    """Get prediction from a model given a counterfactual

    Parameters
    ----------
    mod : _type_
        trained scLEMBAS model
    tf_adata : _type_
        TF AnnData object
    train_cells : List[str]
        the barcodes for cells used in training
    test_cells : 
        the barcodes for cells assigned to test 
    cat_col : str
        the categorical column in `obs`
    pert_col : str
        the perturbation column in `obs`
    ctrl_pert : str
        the control perturbation in `obs[pert_col]`
    counterfactual : Literal['perturbation', 'category', None], optional
        currently the counterfactual will only cross between perturbations or category, keeping the 
        other constant (e.g., within a cell type b/w perturbations or within a perturbation b/w cell types)
        by default 'perturbation'
        - 'perturbation': predicts the test condition from the control pertrubation + same category in train
        - 'category': predicts the test condition from a different cell line + the same perturbation in train.
                      the different cell line is specified by `cat_counterfactual_map`
        - None: iterates through train conditions and predicts without a counterfactual
    cat_counterfactual_map : dict, optional
        a dictionary mapping each test condition to a  different cell line in 
        the train data that has the same perturbation from which to ask the 'category' counterfactual
        only needed when setting `counterfactual`  = 'category', by default None
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
    max_cells : int
        the max cells in a forward pass; for cuda memory, will break up into chunks
    return_full : bool, optional
        whether to return model output prior to ProjectOutput transformation (True) or after (False), by default False
    """
    
    max_cells = np.inf if max_cells is None else max_cells
    
    expr, X_in, covariates_idx, obs = setup_prediction(
        mod = mod, 
        train_cells = train_cells,
        test_cells = test_cells, 
        tf_adata = tf_adata, 
        cat_col = cat_col, 
        pert_col = pert_col, 
        ctrl_pert = ctrl_pert, 
        counterfactual = counterfactual, 
        cat_counterfactual_map = cat_counterfactual_map
    )

    print('Get the predictions')
    if expr.shape[0] < max_cells:
        res = run_prediction(mod = mod, 
                             remove_type = remove_type, 
                             return_bias = return_bias, 
                             X_in = X_in, 
                             covariates_idx = covariates_idx, 
                             expr = expr, 
                            obs = obs, 
                            return_full = return_full)

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

        for chunk_idx in trange(len(expr_chunks)):
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


# ----------------------------------------USAGE----------------------------------------
# expr, X_in, covariates_idx, obs = setup_prediction(
#     mod = mod, 
#     train_cells = trainer.X_train.index.tolist(),
#     test_cells = trainer.X_test.index.tolist(),
#     tf_adata = tf_adata, 
#     cat_col = cat_col, 
#     pert_col = pert_col, 
#     ctrl_pert = 'DMSO_TF', 
#     counterfactual = 'perturbation', 
#     cat_counterfactual_map = None
# )


# cat_counterfactual_map = Tu.match_test_cell_lines(
#     train_conds = train_conds,                               
#     test_conds = test_conds,                                    
#     seed = seed
# )
# full_expr, full_X, full_covariates, obs = Tu.setup_prediction(
#     mod = mod, 
#     train_cells = trainer.X_train.index.tolist(),
#     test_cells = trainer.X_test.index.tolist(),
#     tf_adata = tf_adata, 
#     cat_col = cat_col, 
#     pert_col = pert_col, 
#     ctrl_pert = 'DMSO_TF', 
#     counterfactual = 'category', 
#     cat_counterfactual_map = cat_counterfactual_map
# )

# res = run_prediction(mod = mod, 
#                      remove_type = 'none', 
#                      return_bias = False, 
#                      X_in = X_in, 
#                      covariates_idx = covariates_idx, 
#                      expr = expr, 
#                     obs = obs, 
#                     return_full = False)


# # combined
# res = get_prediction(
#     mod = mod,
#     train_cells = trainer.X_train.index.tolist(),
#     test_cells = trainer.X_test.index.tolist(), 
#     tf_adata = tf_adata,
#     cat_col = cat_col,
#     pert_col = pert_col ,
#     ctrl_pert = 'DMSO_TF', 
#     counterfactual = 'perturbation',
#     cat_counterfactual_map = None,
#     remove_type = 'none',
#     return_bias = False, 
#     max_cells = int(5e3), 
#     return_full = False
# )
# ----------------------------------------USAGE----------------------------------------

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
        utils.clear_memory()
    loss['Mean EMD Loss'] = sum(loss.values())/len(loss) # averaged across condition to not scale with n_conditions
    
    
    return {k: float(v.cpu().numpy()) for k,v in loss.items()}


def _per_condition_projection(tf_adata, X_in, per_condition_models, ctrl_pert, counterfactual, cat_counterfactual_map):
    X_lp = []
    X_umap = []
    coords = [] # will repeat coords for control condition
    control_for_labels = [] # need to track which perturbaiton the control was used for

    predicted_conds = tf_adata.obs.condition.unique().tolist()
    predicted_conds = [tc for tc in predicted_conds if tc.split('^')[1] != ctrl_pert]
    for test_cond in tqdm(predicted_conds):
        cat, pert = test_cond.split('^')
        if counterfactual == 'perturbation':
            ctrl_cond = '^'.join([cat, ctrl_pert]) # include control condition
        else:
            ctrl_cond = '^'.join([cat_counterfactual_map[cat], pert])
        coords_cond = np.where(tf_adata.obs.condition.isin([test_cond, ctrl_cond]))[0]
        
        # record what perturbation the control is with respect to (since re-calculated per perturbation)
        new_control_for_labels = tf_adata[coords_cond, :].obs.condition.astype(str).copy()
        if counterfactual == 'perturbation':
            ctrl_cond_label = pert #ctrl_cond + '| CONTROLFOR{}'.format(pert)
        else:
            ctrl_cond_label = cat #ctrl_cond + '| CONTROLFOR{}'.format(cat)
        new_control_for_labels[new_control_for_labels == ctrl_cond] = ctrl_cond_label 
        new_control_for_labels[new_control_for_labels == test_cond] = None

        control_for_labels += new_control_for_labels.tolist()

        X_in_cond = X_in[coords_cond, :]
        X_lp_cond = per_condition_models[test_cond][linear_projection + '_mod'].transform(X_in_cond)

        X_lp.append(X_lp_cond)
        coords.append(coords_cond)

        if project_umap: 
            umap_mod = per_condition_models[test_cond][umap_key + '_mod'] 
            X_umap_cond = umap_mod.transform(X_lp_cond[:, :lp_rank])

            X_umap.append(X_umap_cond)

    X_umap = None
    if project_umap:
        X_umap = np.vstack(X_umap)
    X_lp = np.vstack(X_lp)
    coords = np.concatenate(coords, axis = 0)

    # expand with redundant values for controls since repeating for each PLS specific model
    tf_adata = tf_adata[coords, :].copy() 
    tf_adata.obs['control_for'] = pd.Categorical(control_for_labels)

    return tf_adata, X_lp, X_umap

def _per_condition_projection(tf_adata, 
                              X_in, 
                              linear_projection, 
                              per_condition_models, 
                              ctrl_pert, 
                              counterfactual, 
                              cat_counterfactual_map, 
                              project_umap
                              ):
    X_lp = []
    X_umap = []
    coords = [] # will repeat coords for control condition
    control_for_labels = [] # need to track which perturbaiton the control was used for

    predicted_conds = tf_adata.obs.condition.unique().tolist()
    predicted_conds = [tc for tc in predicted_conds if tc.split('^')[1] != ctrl_pert]
    for test_cond in tqdm(predicted_conds):
        cat, pert = test_cond.split('^')
        if counterfactual == 'perturbation':
            ctrl_cond = '^'.join([cat, ctrl_pert]) # include control condition
        else:
            ctrl_cond = '^'.join([cat_counterfactual_map[cat], pert])
        coords_cond = np.where(tf_adata.obs.condition.isin([test_cond, ctrl_cond]))[0]
        
        # record what perturbation the control is with respect to (since re-calculated per perturbation)
        new_control_for_labels = tf_adata[coords_cond, :].obs.condition.astype(str).copy()
        if counterfactual == 'perturbation':
            ctrl_cond_label = pert #ctrl_cond + '| CONTROLFOR{}'.format(pert)
        else:
            ctrl_cond_label = cat #ctrl_cond + '| CONTROLFOR{}'.format(cat)
        new_control_for_labels[new_control_for_labels == ctrl_cond] = ctrl_cond_label 
        new_control_for_labels[new_control_for_labels == test_cond] = None

        control_for_labels += new_control_for_labels.tolist()

        X_in_cond = X_in[coords_cond, :]
        X_lp_cond = per_condition_models[test_cond][linear_projection + '_mod'].transform(X_in_cond)

        X_lp.append(X_lp_cond)
        coords.append(coords_cond)

        if project_umap: 
            umap_mod = per_condition_models[test_cond][umap_key + '_mod'] 
            X_umap_cond = umap_mod.transform(X_lp_cond[:, :lp_rank])

            X_umap.append(X_umap_cond)

    X_umap = None
    if project_umap:
        X_umap = np.vstack(X_umap)
    X_lp = np.vstack(X_lp)
    coords = np.concatenate(coords, axis = 0)

    # expand with redundant values for controls since repeating for each PLS specific model
    tf_adata = tf_adata[coords, :].copy() 
    tf_adata.obs['control_for'] = pd.Categorical(control_for_labels)

    return tf_adata, X_lp, X_umap

def project_prediction(
    tf_adata_actual, 
    tf_adata_predicted, 
    linear_projection: Literal['pca', 'pls'],
    per_condition_models = None,
    ctrl_pert: str = 'DMSO_TF', # only needed for per_condition_models
    counterfactual: Literal['perturbation', 'category'] = 'perturbation', 
    cat_counterfactual_map: dict = None,
    project_umap: bool = True, 
    merge: bool = True
):
    """Projects predictions into existing latent space. 

    Parameters
    ----------
    tf_adata_actual : _type_
        the actual TF activity data, with necessary embedding objects (if not using per_condition_models)
    tf_adata_predicted : _type_
       the predicted TF activity data
    linear_projection : Literal['pca', 'pls']
         the linear embedding to project into
    per_condition_models : Dict, optional
        allows for projections into models calculated on a subset of the data specific to condition, by default None
        if None, assumes model fit on all data is stored in relevant `tf_adata_actual.uns` keys
        otherwise, keys of dictionary should match test condition values that are a subset of those in `tf_adata_actual.obs.condition`. Will then project both predicted and actual data using the models specific to each condition.
        Note, because we include the 
        model predictions from the training counterfactual as well, this will make tf_adata_predicted larger with repeated values in the 
        `tf_adata_predicted.X`. For example, if cell_type_A^perturbation_1 in a test condition, we include the predictions for 
        cell_type_A^perturbation_CTRL. Consequently, each retrieved X_PLS separating perturbation was transformed including the ctrl, 
        and if multiple cell type A perturbations are in the test conditions, we will calculate multiple DISTINCT projections 
        of cell_type_A^perturbation_CTRL. 
    ctrl_pert : str, optional
        the control perturbation, by default 'DMSO_TF'. Only needed if using per_condition_models. 
    counterfactual : Literal['perturbation', 'category'], optional
        currently the counterfactual will only cross between perturbations or category, keeping the 
        other constant (e.g., within a cell type b/w perturbations or within a perturbation b/w cell types)
        by default 'perturbation'
        - 'perturbation': predicts the test condition from the control pertrubation + same category in train
        - 'category': predicts the test condition from a different cell line + the same perturbation in train.
                      the different cell line is specified by `cat_counterfactual_map`
        only needed if using per_condition_models
    cat_counterfactual_map : dict, optional
        a dictionary mapping each test condition to a  different cell line in 
        the train data that has the same perturbation from which to ask the 'category' counterfactual
        only needed when setting `counterfactual`  = 'category', by default None
    project_umap : bool, optional
        whether to run umap on the linear embedding output, by default True
    merge: bool, optional
        whether to combine the predicted AnnData object with the actual, by default True
    """

    # filter to predicted conditions -- only necessary for using per_condition_models, but set here for consistency in output
    predicted_conds = tf_adata_predicted.obs.condition.unique().tolist()
    tf_adata_actual = tf_adata_actual[tf_adata_actual.obs.condition.isin(predicted_conds),:].copy()

    if linear_projection == 'pca':
        X_in_pred = tf_adata_predicted.to_df().values
    elif linear_projection == 'pls':
        if tf_adata_actual.uns['pls']['encoder_x'] is not None:
            msg = 'Internal: Need to account for confounder controlling -- store params in tf_adata_actual to run '
            msg += 'prepar_input_matrix_plsda in an automated manner'
            raise ValueError(msg)

        X_in_pred, _ = ls.prepare_input_matrix_plsda(adata = tf_adata_predicted,
                            control_confounders = [],
                            enc_X = tf_adata_actual.uns['pls']['encoder_x'])

    if project_umap:
        umap_key = 'umap' if linear_projection == 'pca' else 'umap_pls'

    lp_rank = tf_adata_actual.uns[linear_projection][linear_projection + '_rank']
    if per_condition_models is None:
        lp_mod = tf_adata_actual.uns[linear_projection][linear_projection + '_mod']
        X_lp_pred = lp_mod.transform(X_in_pred)
        X_lp_actual = tf_adata_actual.obsm['X_' + linear_projection]

        if project_umap:
            umap_mod = tf_adata_actual.uns[umap_key][umap_key + '_mod']
            X_umap_pred = umap_mod.transform(X_lp_pred[:, :lp_rank])
            X_umap_actual = tf_adata_actual.obsm['X_' + umap_key]

    else:
        if counterfactual == 'category':
            raise ValueError('Internal: This needs to be checked')
        
        # clear slots for condition - specific fits
        for tf_adata_ in [tf_adata_actual, tf_adata_predicted]:
            if linear_projection in tf_adata_.uns and linear_projection + '_mod' in  tf_adata_.uns[linear_projection]:
                del tf_adata_.uns[linear_projection][linear_projection + '_mod']
            if 'X_' + linear_projection in tf_adata_.obsm:
                del tf_adata_.obsm['X_' + linear_projection]
            if project_umap and 'X_' + umap_key in tf_adata_.obsm:
                del tf_adata_.obsm['X_' + umap_key]
            
        tf_adata_predicted, X_lp_pred, X_umap_pred = _per_condition_projection(tf_adata = tf_adata_predicted, 
                                                      X_in = X_in_pred,
                                                                               linear_projection = linear_projection,
                                                      per_condition_models = per_condition_models,
                                                      ctrl_pert = ctrl_pert, 
                                                      counterfactual = counterfactual, 
                                                      cat_counterfactual_map = cat_counterfactual_map, 
                                                      project_umap = project_umap,
                                                      )
        tf_adata_actual, X_lp_actual, X_umap_actual  = _per_condition_projection(tf_adata = tf_adata_actual, 
                                                   X_in = tf_adata_actual.X,
                                                                                 linear_projection = linear_projection,
                                                   per_condition_models = per_condition_models,
                                                      ctrl_pert = ctrl_pert, 
                                                      counterfactual = counterfactual, 
                                                      cat_counterfactual_map = cat_counterfactual_map, 
                                                      project_umap = project_umap,
                                                      )
        

    tf_adata_predicted.obs['batch'] = 'predicted'
    if merge:
        tf_adata_actual.obs['batch'] = 'actual'
        tf_adata_actual.obs['counterfactual_condition'] = None # to retain this column in the predicted object

        tf_adata_merged = sc.concat([tf_adata_actual, tf_adata_predicted])
        tf_adata_merged.obs['barcode'] = tf_adata_merged.obs.index.tolist()

        tf_adata_merged.obs_names_make_unique()

        tf_adata_merged.obsm['X_' + linear_projection] = np.concatenate(
            [X_lp_actual, X_lp_pred],
            axis = 0
        )

        if project_umap:
            tf_adata_merged.obsm['X_' + umap_key] = np.concatenate(
                    [X_umap_actual, X_umap_pred],
                    axis = 0
                )

        del tf_adata_actual

        return tf_adata_merged
    else:
        tf_adata_predicted.obsm['X_' + linear_projection] = X_lp_pred

        if project_umap:
            tf_adata_predicted.obsm['X_' + umap_key] = X_umap_pred

        return tf_adata_predicted
    
    
# # how to check projections as they are formatted now (09/12/25):
# # get tf_adata_predicted on it's own

# md = tf_adata_predicted.obs
# mask = (md.condition.isin([test_cond, ctrl_cond]))
# X_pls_predicted_correct = pls_models[test_cond]['pls_mod'].transform(tf_adata_predicted[mask, :].X)


# md = tf_adata.obs
# mask = (md.condition.isin([test_cond, ctrl_cond]))
# X_pls_actual_correct = pls_models[test_cond]['pls_mod'].transform(tf_adata[mask, :].X)

# # project into embedding (PLS for perturbation counterfactual, PCA otherwise) and combine with actual anndata
# tf_adata_merged = project_prediction(
#     tf_adata, 
#     tf_adata_predicted, 
#     linear_projection = 'pls', 
#     per_condition_models = pls_models,
#     ctrl_pert = 'DMSO_TF', # only needed for per_condition_models
#     counterfactual = 'perturbation', 
#     cat_counterfactual_map = cat_counterfactual_map, 
#     project_umap = False, 
#     merge = True
# )

# # CTRL: add control specification
# mask = [on.endswith('_predicted_ctrl') for on in tf_adata_merged.obs.barcode]
# tf_adata_merged.obs.loc[mask, 'batch'] = 'predicted_ctrl'


# md = tf_adata_merged.obs
# maskA = (md.condition == test_cond)
# maskB = (md.condition == ctrl_cond) &(md.control_for == test_pert)
# maskC = (md.batch.isin(['predicted', 'predicted_ctrl']))
# mask = (maskA | maskB) & maskC
# if not np.allclose(tf_adata_merged[mask, ].obsm['X_pls'], X_pls_predicted_correct):
#     raise ValueError('Mismatched PLS')


# md = tf_adata_merged.obs
# maskA = (md.condition == test_cond)
# maskB = (md.condition == ctrl_cond) &(md.control_for == test_pert)
# maskC = (md.batch.isin(['actual']))
# mask = (maskA | maskB) & maskC
# if not np.allclose(tf_adata_merged[mask, ].obsm['X_pls'], X_pls_actual_correct):
#     raise ValueError('Mismatched PLS')


def all_rows_close(arr, tol=1e-5):
    diffs = np.abs(arr[:, None, :] - arr[None, :, :])
    return np.all(diffs <= tol)

def merge_novar_predictions(tf_adata_predicted, remove_type, 
                           pert_col = 'drug', 
                           cat_col = 'cell_line', 
                           atol = 1e-5):
    """Checks that predictions without global bias don't have variance and merges."""

    if remove_type == 'total_bias':
        same_col = pert_col
    elif remove_type == ['adj', 'global_bias']:
        same_col = cat_col
    elif remove_type == 'global_bias':
        same_col = 'condition'
    else:
        raise ValueError('Only predictions without global bias should not vary between cells')

    # check that predictions are unique according to component removal
    same_group = tf_adata_predicted.obs[same_col].unique()
    for sg in same_group:
        sg_adata = tf_adata_predicted[tf_adata_predicted.obs[same_col] == sg, :]
        value_check = all_rows_close(sg_adata.to_df().values, tol = atol)
        if not value_check:
            raise ValueError('Expected predictions to be same by grouping for {}'.format(remove_type))
#         else:
#             keep_idx.append(sg_adata.obs_names[0])
            
    
    # repeate, this time merging non-unique values by condition 
    # instead of merging by "same_col" bc need a unique value per condition for visualizations
    keep_idx = []
    same_group = tf_adata_predicted.obs['condition'].unique()
    for sg in same_group:
        sg_adata = tf_adata_predicted[tf_adata_predicted.obs['condition'] == sg, :]
        keep_idx.append(sg_adata.obs_names[0])
    
    return tf_adata_predicted[keep_idx, :].copy()


reduction_type_map = {'pca': 'pc', 
                      'pls': 'pls', 
                      'umap': 'umap', 
                      'umap_pls': 'umap (pls)'}
def adata_dimviz(
        adata, 
        reduction_type: Literal['pca', 'pls', 'umap', 'umap_pls'], 
        cats: List[str], 
        subset_size: int = int(1e4), 
        seed: int = 888):
    """Formats for visualization of embedding

    Parameters
    ----------
    adata : _type_
        AnnData object
    reduction_type : Literal['pca', 'pls', 'umap', 'umap_pls']
        embedding to visualize
    cats : List[str]
        columns in adata.obs to retain
    subset_size : int, optional
        proportionally subsets across all cats, by default int(1e4)
    seed : int, optional
        random state, by default 888
    """


    reduction_type_ = reduction_type_map[reduction_type]

    if type(cats) != list:
        cats = [cats]

    viz_df = pd.DataFrame(adata.obsm['X_' + reduction_type])
    viz_df = pd.concat([viz_df, pd.DataFrame(adata.obs[cats]).reset_index(drop = True)], ignore_index = True, axis = 1)
    viz_df.columns = [reduction_type_.upper() + str(i+1) for i in range(viz_df.shape[1])][:-len(cats)] + cats


#     nmi = normalized_mutual_info_score(adata.obs.leiden, adata.obs[cat])

    if subset_size is not None and subset_size < viz_df.shape[0]:
        grouped = viz_df.groupby(cats, observed=False)
        cell_prop = viz_df[cats].value_counts(normalize = True)
        index_to_keep = []
        np.random.seed(seed)
        for cat_type, cat_df in grouped:
            mask = (viz_df[cats].values == cat_type)
            if len(cats) > 1:
                mask = mask.all(axis=1)
            all_barcodes = viz_df[mask].index
            
            cat_subset_size = np.round(max(1, np.round(cell_prop.loc[cat_type]*subset_size)))
            cat_subset_size = min(cat_subset_size, len(all_barcodes))
            cat_subset_size = int(cat_subset_size)

            
            index_to_keep += np.random.choice(all_barcodes, 
                         size = cat_subset_size,
                         replace = False).tolist()

        viz_df = viz_df.loc[index_to_keep, :]

    # shuffle
    viz_df = viz_df.sample(frac=1, random_state = seed).reset_index(drop=True)
    
    return viz_df
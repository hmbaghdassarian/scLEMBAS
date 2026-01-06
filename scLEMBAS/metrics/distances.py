"""Distance-based metrics"""

from typing import Literal

import pandas as pd
import scanpy as sc
import numpy as np

import sklearn

from .. import utilities as utils

from geomloss import SamplesLoss
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"



def get_EMD_loss(tf_adata_actual, tf_adata_predicted, device = device):
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
        y_actual = torch.tensor(tf_adata_actual[tf_adata_actual.obs.condition == cond, ].to_df().values).to(device)
        loss[cond] = loss_fn(y_predicted, y_actual)
        utils.clear_memory()
    loss['Mean EMD Loss'] = sum(loss.values())/len(loss) # averaged across condition to not scale with n_conditions
    
    
    return {k: float(v.cpu().numpy()) for k,v in loss.items()}

def rmse(u, v):
    return np.sqrt(np.mean((u - v) ** 2))

def rank_score(
    tf_adata_actual, 
    tf_adata_predicted, 
    pert_col: str, 
    distance_metric: str = 'manhattan', 
#     method: Literal['perturbench', 'virtual_cell_challenge'] = 'virtual_cell_challenge',
    **kwargs
):
    """Average perturbation rank score for a set of predicted perturbations, as described in Perturbench (https://arxiv.org/html/2408.10609v3)
    
    The rank metric falls between [0,1]. 1 indicates a perfect score, and 0.5 is the expected score of a random prediction. 
    
    Note, Perturbench normalizes to the number of *predicted* perturbations, so we do not include all perturbations in the dataset
    set (as contained in tf_adata_actual), but rather limit to those in the tf_adata_predicted 
    (e.g., likely only include test data rather than train + test): "For a given observed perturbation, 
    how close the model prediction is to the observation, compared to predictions made for other perturbations."

    Parameters
    ----------
    tf_adata_actual : _type_
        AnnData object containing actual data
    tf_adata_predicted : _type_
        AnnData object containing actual data
    pert_col : _type_
        column in `tf_adata_predicted.obs` containing perturbation information
    distance_metric : str, optional
        how to calculate the distance between psuedobulked perturbations, by default 'manhattan' (as in virtual cel challenge)
        options include any metric included in available to sklearn.metrics.pairwise_distances as well as rmse
    method : str, optional
        DEPRECATED: only difference is whether individual rank denominator is subtracted by 1
        whether to use the scoring based on the description in PerturBench (https://arxiv.org/html/2408.10609v3)
        or Virtual Cell Challenge (https://virtualcellchallenge.org/evaluation). Should be very similar.
    **kwargs
        key word arguments to pass to `sklearn.metrics.pairwise_distances`
        
        
    Returns
    ----------
    rank_score_average : float
        The rank score averaged across modelled perturbations
    rank_scores : dict
        A dictionary mapping the rank score of each modelled perturbation
    """

    pred_pert_mask = tf_adata_actual.obs[pert_col].isin(tf_adata_predicted.obs[pert_col])
    tf_adata_actual = tf_adata_actual[pred_pert_mask, :].copy()

    n_pert_predicted = tf_adata_predicted.obs[pert_col].nunique()
    if tf_adata_actual.obs[pert_col].nunique() != n_pert_predicted:
        raise ValueError('Not all perturbations present in actual data')

    psuedobulked_perts_actual = sc.get.aggregate(tf_adata_actual, by = pert_col, axis = 0, func = 'mean')
    psuedobulked_perts_predicted = sc.get.aggregate(tf_adata_predicted, by = pert_col, axis = 0, func = 'mean')

    # get pairwise distances as in virtual cell challenge
    if distance_metric == 'rmse':
        distance_metric = rmse
    
    distances = sklearn.metrics.pairwise_distances(
        psuedobulked_perts_actual.layers['mean'],
        psuedobulked_perts_predicted.layers['mean'],
        metric=distance_metric, 
    #     **kwargs
    )


    distances = pd.DataFrame(distances, 
                 index = psuedobulked_perts_actual.obs_names, 
                 columns = psuedobulked_perts_predicted.obs_names)

    rank_scores = {}
    for pred_pert in distances.columns:
        corresponding_distance = distances.loc[pred_pert, pred_pert] # self distance

        all_distances = distances.loc[:, pred_pert]
        pred_mask = (all_distances.index != pred_pert) # exclude self
        rank = (all_distances[pred_mask] <= corresponding_distance).sum()

        rank /= (n_pert_predicted - 1) # since excluding self

        rank_scores[pred_pert] = 1 - rank 

    rank_score_average = sum(rank_scores.values())/len(rank_scores)
    
    return rank_score_average, rank_scores




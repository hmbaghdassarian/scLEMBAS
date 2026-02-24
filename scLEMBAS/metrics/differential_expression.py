"""Pipeline to calculate concordance of DE between actual and predicted data. See `concordance_pipeline` function for all details."""

from typing import Literal
from joblib import Parallel, delayed


from tqdm.auto import tqdm as _tqdm

import scipy
from statsmodels.stats.multitest import multipletests

import pandas as pd
import numpy as np

from cliffs_delta import cliffs_delta

def tqdm_outer_only(iterable=None, **kwargs):
    # if a bar already exists, we're nested → disable this one
    return _tqdm(iterable, disable=bool(_tqdm._instances), **kwargs)


def cd_effect(array_comp, array_ctrl):
    """Arrays are observations x features. Both arrays should have the same features.
    
    Positive values means array_comp feature > array_ctrl, negative values are opposite. 
    
    """
    cd_vals, cd_quals = [],[]
    for feature_idx in range(array_comp.shape[1]):
        cd_val, cd_qual = cliffs_delta(array_comp[:, feature_idx], array_ctrl[:, feature_idx])
        cd_vals.append(cd_val)
        cd_quals.append(cd_qual)
    return cd_vals, cd_quals


def get_de(array_comp, array_ctrl, feature_names, alpha, rank_by):
    """DE of array_comp vs array_ctrl. 
    Filter for MWU FDR <= alpha. 
    Order by effect size (cliff's delta) or FDR signifiance.
    Cliff's delta sign is positive for those features higher in array_comp, and negative for those higher in array_ctrl.
    """
    pvals = scipy.stats.mannwhitneyu(array_comp, array_ctrl, alternative = 'two-sided', axis = 0).pvalue
    fdr = multipletests(pvals, method="fdr_bh")[1]
    cd_vals, _ = cd_effect(array_comp, array_ctrl)
    
    de_df = pd.DataFrame(data = {'cliffs_delta': cd_vals, 
                            'MWU_fdr': fdr, 
                                'features': feature_names})
    
    de_df['significant'] = False
    de_df.loc[(de_df.MWU_fdr <= alpha), 'significant'] = True
        
    
    if rank_by == 'significance':
        de_df = de_df.sort_values(by = 'MWU_fdr', ascending = True).reset_index(drop = True)
    elif rank_by == 'effect_size':
        de_df = de_df.sort_values(by="cliffs_delta", key=lambda x: x.abs(), ascending = False).reset_index(drop = True)
    else:
        raise ValueError('Incorrect rank_by specified')
        
    return de_df

def _split_de(de_df):
    de_df = de_df[de_df.significant].copy()
    de_df_pos = de_df[de_df.cliffs_delta > 0]
    de_df_neg = de_df[de_df.cliffs_delta < 0]
    
    return de_df_pos, de_df_neg

def jaccard_index(list1, list2):
    set1, set2 = set(list1), set(list2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if len(union) != 0 else 0
        
def de_score(actual_list, predicted_list):
    """Differential expression score as described in Virtual Cell Challenge.    
    """
    set1, set2 = set(actual_list), set(predicted_list)
    intersection = set1.intersection(set2)
    return len(intersection)/len(actual_list) 

def calculate_concordance(de_actual, de_predicted, concordance_metric, rank_by):
    """DE matrices should already be rank-ordered."""
    n_actual = de_actual.shape[0]
    n_predicted = de_predicted.shape[0]
    concordance = []
    if concordance_metric == 'jaccard':
        metric_function = jaccard_index 
    elif concordance_metric == 'de_score':
        if rank_by != 'effect_size' and n_predicted > n_actual:
            raise ValueError('Must sort by effect size to use DE score in this scenario')
        metric_function = de_score
    else:
        raise ValueError('Must specify jaccard or de_score as the concordance metric')


    for idx in range(1, min(n_actual, n_predicted) + 1):
        cv = metric_function(de_actual.iloc[:idx, :].features, 
                      de_predicted.iloc[:idx, :].features)
        concordance.append(cv)
    return concordance

def _format_concordance_list(concordance_list):
    concordance_list = pd.DataFrame(concordance_list).T
    concordance_list.index.name = 'rank'
    concordance_list.columns.name = 'iteration'
    return concordance_list

def concordance_subpipeline(test_data, predicted_data, ctrl_data, 
                         feature_names, alpha, rank_by, concordance_metric, n_permutation, seed):
    # get statistics and rank-order
    de_actual = get_de(test_data, ctrl_data, 
                       feature_names = feature_names, alpha = alpha, rank_by = rank_by)   
    de_predicted = get_de(predicted_data, ctrl_data, 
                       feature_names = feature_names, alpha = alpha, rank_by = rank_by)
    
    # split into positive and negative and filter for signifance
    de_actual_pos, de_actual_neg = _split_de(de_actual)
    de_predicted_pos, de_predicted_neg = _split_de(de_predicted)

    # calculate concordance
    pos_concordance = calculate_concordance(de_actual_pos, de_predicted_pos, concordance_metric = concordance_metric, rank_by = rank_by)
    neg_concordance = calculate_concordance(de_actual_neg, de_predicted_neg, concordance_metric = concordance_metric, rank_by = rank_by)

    
    if n_permutation is not None:
        rng = np.random.default_rng(seed) 
        de_random = de_actual.copy()
        pos_concordance_rand, neg_concordance_rand = [], []
        for _ in tqdm_outer_only(range(n_permutation)):
            de_random.features = rng.permutation(feature_names) #np.random.permutation(feature_names)
            de_random_pos, de_random_neg = _split_de(de_random)
            pos_cr = calculate_concordance(de_actual_pos, de_random_pos, concordance_metric = concordance_metric, rank_by = rank_by)
            neg_cr = calculate_concordance(de_actual_neg, de_random_neg, concordance_metric = concordance_metric, rank_by = rank_by)

            pos_concordance_rand.append(pos_cr)
            neg_concordance_rand.append(neg_cr)
            
        pos_concordance_rand = _format_concordance_list(pos_concordance_rand)
        neg_concordance_rand = _format_concordance_list(neg_concordance_rand)
    else:
        pos_concordance_rand, neg_concordance_rand = None, None
    
    
    return pos_concordance, neg_concordance, pos_concordance_rand, neg_concordance_rand


def _one_subsample_worker(
    one_seed: int,
    larger,
    smaller,
    k: int,
    test_is_larger: bool,
    ctrl_data,
    feature_names,
    alpha: float,
    rank_by: str,
    concordance_metric: str,
    n_permutation
):
    rng = np.random.default_rng(int(one_seed))
    idx = rng.choice(larger.shape[0], k, replace=False)
    larger_sub = larger[idx, :]

    test_in = larger_sub if test_is_larger else smaller
    pred_in = smaller if test_is_larger else larger_sub

    return concordance_subpipeline(
        test_in, pred_in, ctrl_data, feature_names,
        alpha, rank_by, concordance_metric,
        n_permutation, int(one_seed)
    )

def concordance_pipeline(
    tf_adata_actual, 
    tf_adata_predicted, 
    test_condition: tuple, 
    pert_col: str, 
    cat_col: str, 
    ctrl_var: str, 
    counterfactual: Literal['perturbation', 'category'] = 'perturbation', 
    n_subsample = 100, 
    n_permutation = 100, 
    alpha: float = 0.1, 
    rank_by: Literal['effect_size', 'significance'] = 'effect_size', 
    concordance_metric: Literal['de_score', 'jaccard'] = 'de_score', 
    seed: int = 888,
    n_cores: int = None
):
    """Calculates differential-expression concordance within a given condition. This is a comparison of comparisons: 
    (DE of actual test vs control) compared to (DE of predicted test vs control).


    Step 1: Filter data to condition of interest

    Step 2 [OPTIONAL]: IF n_obs predicted != n_obs test actual, subsample the larger dataset `n_subsample`  times to avoid having one 
    comparison be more powerd than the other. 

    Step 3: Run DE for each of predicted and actual, with control as baseline. Get BH FDR as the alpha signifiance threshold and 
    Cliff's Delta as the effect size metric. 

    Step 4: Once filtered for significance and rank-ordered, calculate either the Jaccard Index or the DE Score (see Virtual Cell Challenge) at each rank. Stop calculating at the smaller of the two DE sets. Do this separately for those with a positive or negative effect size (as measured by Cliff's Delta). 
    
    Note, because this is a running rank, unlike the native Virtual Cell Challenge DE Score, we do not continue to calculate it
    in instances where # of DE genes of the actual is > predicted. 
    
    Step 5 [OPTIONAL]: Create a null distribution of concordance at each rank by permuting feature labels prior to calculating concordance. 



    Parameters
    ----------
    tf_adata_actual : 
        AnnData object containing the actual data
    tf_adata_predicted :
        AnnData object containing the predicted data
    test_condition : tuple
        first value is a string of the category of interest, second is a value of the string of the perturbation of interest
    pert_col : str
        column in `adata.obs.X` containing the perturbation metadata
    cat_col : str
        column in `adata.obs.X` containing the categorical metadata
    ctrl_var : str
        the control perturbation or category to compare to 
    counterfactual : Literal['perturbation', 'category'], optional
        whether DE comparison is being made across perturbations or the categorical covariate, by default perturbation
    n_subsample : int, optional
        if predicted and test data contain a different number of observations, will subset to the minimum of the two 
        `n_subsample` times. This ensures one comparison is not more powered than another, by default 100
    n_permutation : int, optional
        if not None, generates a null distribution of concordance values at each rank by permuting features in the DE output, by default 100
        Note, when used in conjunction with n_subsample, will generate nested iterations. In this case, takes the median across permutations for each subsample. 
    alpha : float, optional
        BH FDR threshold to call a feature significant, by default 0.1
    rank_by : Literal['effect_size', 'significance'], optional
        whether to rank order genes by effect size (Cliff's Delta) or significance (BH FDR), by default 'effect_size'
    concordance_metric : Literal['de_score', 'jaccard'], optional
        what metric to use at each rank to quantify concordance, by default 'de_score'
            - 'de_score': takes the DE Score as described by the Virtual Cell Challenge
            - 'jaccard': takes Jaccard Index
    seed : int, optional
        random state to set, by default 888
    """

    test_cat, test_pert = test_condition
    if counterfactual == 'perturbation':
        test_var = test_pert 
    #     ctrl_var = counterfactual_map[test_cond].split('^')[1]
        main_col = pert_col

        ortho_var = test_cat
        ortho_col = cat_col

    elif counterfactual == 'category':
        raise ValueError('Need to check all this')
        test_var = test_cat
    #     ctrl_var = counterfactual_map[test_cond].split('^')[0]
        main_col = cat_col

        ortho_var = test_pert
        ortho_col = pert_col

    if not (tf_adata_actual.var_names == tf_adata_predicted.var_names).all():
        raise ValueError('Input AnnData objects should have the same features')
    feature_names = tf_adata_actual.var_names 

    ctrl_data = tf_adata_actual[(tf_adata_actual.obs[ortho_col] == ortho_var) & (tf_adata_actual.obs[main_col] == ctrl_var)].X
    test_data = tf_adata_actual[(tf_adata_actual.obs[ortho_col] == ortho_var) & (tf_adata_actual.obs[main_col] == test_var)].X
    predicted_data = tf_adata_predicted[(tf_adata_predicted.obs[ortho_col] == ortho_var) & (tf_adata_predicted.obs[main_col] == test_var)].X

    # subsamples to the same size, so that one does not have different power over the other
    if (n_subsample is not None) and (test_data.shape[0] != predicted_data.shape[0]):
        test_is_larger = test_data.shape[0] > predicted_data.shape[0]
        larger, smaller = (test_data, predicted_data) if test_is_larger else (predicted_data, test_data)
        k = smaller.shape[0]
        if n_cores is not None and n_cores > 1:
            feature_names = np.asarray(feature_names)
            ctrl_data = np.asarray(ctrl_data)
            test_data = np.asarray(test_data)
            predicted_data = np.asarray(predicted_data)

            rng = np.random.default_rng(seed)
            sub_seeds = rng.integers(0, 2**32 - 1, size=n_subsample, dtype=np.uint32)

            results = Parallel(
                n_jobs=min(n_cores, n_subsample),      
                prefer="processes",
                batch_size=1,
                max_nbytes="50M",                # memmap large arrays to reduce copying
                mmap_mode="r"
            )(
                delayed(_one_subsample_worker)(
                    int(s),
                    larger, smaller, k, test_is_larger,
                    ctrl_data, feature_names,
                    alpha, rank_by, concordance_metric,
                    n_permutation
                )
                for s in sub_seeds
            )

            pos_concordance = []
            neg_concordance = []
            pos_concordance_rand = []
            neg_concordance_rand = []

            for pos_c, neg_c, pos_c_rand, neg_c_rand in results:
                pos_concordance.append(pos_c)
                neg_concordance.append(neg_c)
                if n_permutation is not None:
                    pos_concordance_rand.append(pos_c_rand.median(axis=1).tolist())
                    neg_concordance_rand.append(neg_c_rand.median(axis=1).tolist())
        else:
            pos_concordance, neg_concordance = [], []
            pos_concordance_rand, neg_concordance_rand = [], []

            np.random.seed(seed)
            for i in tqdm_outer_only(range(n_subsample)):
                idx = np.random.choice(larger.shape[0], k, replace=False)
                larger_sub = larger[idx, :]

                # keep the pipeline’s (test, predicted) argument order
                test_in  = larger_sub if test_is_larger else smaller
                pred_in  = smaller if test_is_larger else larger_sub

                res = concordance_subpipeline(test_in, pred_in, ctrl_data, feature_names, alpha, rank_by, concordance_metric, 
                                                                   n_permutation, seed + i)
                pos_c, neg_c, pos_c_rand, neg_c_rand = res

                pos_concordance.append(pos_c)
                neg_concordance.append(neg_c)
                if n_permutation is not None:
                    pos_concordance_rand.append(pos_c_rand.median(axis = 1).tolist())
                    neg_concordance_rand.append(neg_c_rand.median(axis = 1).tolist())

        pos_concordance = _format_concordance_list(pos_concordance)
        neg_concordance = _format_concordance_list(neg_concordance)
        pos_concordance_rand = _format_concordance_list(pos_concordance_rand)
        neg_concordance_rand = _format_concordance_list(neg_concordance_rand)

    else:
        res = concordance_subpipeline(test_data, predicted_data, ctrl_data, feature_names, alpha, rank_by, concordance_metric, 
                                                               n_permutation, seed)
        pos_concordance, neg_concordance, pos_concordance_rand, neg_concordance_rand = res

        pos_concordance = _format_concordance_list([pos_concordance])
        neg_concordance = _format_concordance_list([neg_concordance])


    return pos_concordance, neg_concordance, pos_concordance_rand, neg_concordance_rand
    


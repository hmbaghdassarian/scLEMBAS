"""
Preprocessing functions for single-cell AnnData objects.
"""
import itertools
from itertools import repeat
import multiprocessing
from tqdm import tqdm, trange
import warnings
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind, rankdata
from scipy import stats, sparse
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import euclidean
import statsmodels.stats.multitest as smm
from cliffs_delta import cliffs_delta

import torch
from geomloss import SamplesLoss

from sklearn.decomposition import PCA

from anndata import AnnData
import scanpy as sc

default_device = "cuda" if torch.cuda.is_available() else "cpu"
default_emd_loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05).to(default_device)

grn_link = ppi_link = 'https://zenodo.org/records/11477837/files/grn_organism_06_04_24.csv'

def get_tf_activity(adata, organism: str, grn = 'collectri', 
                    verbose: bool = True, min_n: int = 5, use_raw: bool = False,
                    filter_pvals: bool = False, pval_thresh: float = 0.05,
                    hvg: bool = False, static: bool = True,
                    consensus: bool = True, 
                    **kwargs):
    """Wrapper of decoupler to estimate TF activity from single-cell transcriptomics data.

    Parameters
    ----------
    adata : AnnData
        Annotated single-cell data matrix 
    organism : str
        The organism of interest: either NCBI Taxonomy ID, common name, latin name or Ensembl name. 
        Organisms other than human will be translated from human data by orthology.
    grn : str, optional
        database to get the GRN, by default 'collectri'. Available options are ``collectri`` or ``dorothea``.
    min_n : int
        Minimum of targets per source. If less, sources are removed. By default 5.
    verbose : bool
        Whether to show progress.
    use_raw : bool
        Use AnnData its raw attribute.
    filter_pvals : bool
        whether to set TF activity estimates to 0 if it is insignificant according to pval_thresh
    pval_thresh : float
        significance threshold, by default 0.05. Used in conjunction with filter_pvals.
    hvg : bool
        whether to filter for HVGs (stored in `adata.var['highly_variable']`) prior to TF activity inference
    static : bool, optional
        whether to download a static version of the GRN DB or the most current, by default True
        for stable results and consistency with downstream analyses, recommended to use static = True
        only organism 'human', 'rat', and 'mouse' available for static
    consensus: bool, optional 
        whether to calculate the consensus score across methods, by default True
    kwargs : 
        passed to  `decoupler.decouple`.

    Returns
    -------
    estimate : DataFrame
        Consensus TF activity scores. Stored in `adata.obsm['consensus_estimate']`.
    pvals : DataFrame
        Obtained TF activity p-values. Stored in `adata.obsm['consensus_pvals']`.
    """
    import decoupler as dc
    #from decoupler.pre import extract


    if hvg:
        adata = adata[:, adata.var['highly_variable']].copy()
    else:
        adata = adata.copy()
    
    if not static:
        grn_map = {'collectri': dc.get_collectri, 'dorothea': dc.get_dorothea} # get_dorothea returns "A" confidence by default
        net = grn_map[grn](organism=organism, split_complexes=False) # builds on dorothea, used by Saez-Rodriguez lab
    else:
        net = pd.read_csv(grn_link.replace('grn', grn).replace('organism', organism), index_col = 0)
    # # reimplementation of dc.run_consensus, allowing all options in dc.decouple to be passed
    # dc.run_consensus(mat=adata, net=net, source='source', target='target', weight='weight', **kwargs)

    # m, r, c = extract(adata, use_raw=use_raw, verbose=verbose)
    if verbose:
        print('Running scores.')
    
    # # unnecessary, this is the default behavior    
    # if not kwargs:
    #     kwargs = {'methods': ['lm', 'ulm', 'wsum'], 
    #               'cns_metds': ['lm', 'ulm', 'wsum_norm']}
    # else:
    #     if 'methods' not in kwargs:
    #         kwargs['methods'] = ['lm', 'ulm', 'wsum']
    #     if 'cns_methods' not in kwargs and kwargs['methods'] == ['lm', 'ulm', 'wsum']:
    #         kwargs['cns_metds'] = ['lm', 'ulm', 'wsum_norm']

    dc.decouple(mat=adata, net=net, source='source', target='target', weight='weight', consensus = consensus,
                      min_n=min_n, verbose=verbose, use_raw=use_raw, **kwargs)

    if filter_pvals:
        estimate_key = [k for k in adata.obsm if k.endswith('_estimate')]
        pvals_key = [k for k in adata.obsm if k.endswith('_pvals')]
        
        if len(estimate_key) > 0 or len(pvals_key) > 0:
            warnings.warn('Multiple TF estimates/pvals, choosing first')
            
        adata.obsm[estimate_key[0]][adata.obsm[pvals_key[0]] > pval_thresh] = 0

    return adata

def transform_tf_activity(tf_estimate: pd.DataFrame) -> pd.DataFrame:
    """Scales TF activity scores between 0 and 1.

    Parameters
    ----------
    tf_estimate : pd.DataFrame
        original TF activity scores

    Returns
    -------
    pd.DataFrame
        scaled TF activity scores
    """
    return 1 / (1 + np.exp(-tf_estimate))

def tf_to_adata(adata: AnnData, estimate_key: str = 'consensus_estimate'):
    """Converts the TF activity results from `get_tf_activity` into its own AnnData object. 

    Parameters
    ----------
    adata : AnnData
        AnnData object with TF activity scores stored in `.obsm[estimate_key]`.
    estimate_key : str, optional
        `.obsm` key under which TF activity is stored, by default 'consensus_estimate'
    Returns
    -------
    tf_adata : AnnData
        AnnData object with input TF activity estimates stored in `.X`.
    """

    tf_adata = AnnData(X = adata.obsm[estimate_key], obs = adata.obs.copy())
    
    all_na = np.isnan(tf_adata.X).all(axis=0)
    n_dropped = len(np.where(all_na)[0])
    if n_dropped > 0:
        tf_adata = tf_adata[:,np.where(~all_na)[0]]
        warnings.warn('{} TFs with all NA scores were dropped'.format(n_dropped))
    
    return tf_adata        
    
#     return best_nmi, best_res
    
def pairwise_pearson_correlation(X, Y):
    # Center the data by subtracting the mean of each row
    X_centered = X - X.mean(axis=1, keepdims=True)
    Y_centered = Y - Y.mean(axis=1, keepdims=True)
    
    # Compute the dot product of the centered data
    dot_product = np.dot(X_centered, Y_centered.T)
    
    # Compute the norm (standard deviation) of each row
    X_norm = np.linalg.norm(X_centered, axis=1)
    Y_norm = np.linalg.norm(Y_centered, axis=1)
    
    # Normalize the dot product by the outer product of the norms
    correlation_matrix = dot_product / np.outer(X_norm, Y_norm)
    
    return correlation_matrix

def pairwise_spearman_correlation(X, Y):
    # Convert the rows of X and Y to ranks
    X_ranked = np.apply_along_axis(rankdata, 1, X)
    Y_ranked = np.apply_along_axis(rankdata, 1, Y)
    
    # Center the ranked data by subtracting the mean of each row
    X_centered = X_ranked - X_ranked.mean(axis=1, keepdims=True)
    Y_centered = Y_ranked - Y_ranked.mean(axis=1, keepdims=True)
    
    # Compute the dot product of the centered ranked data
    dot_product = np.dot(X_centered, Y_centered.T)
    
    # Compute the norm (standard deviation) of each row
    X_norm = np.linalg.norm(X_centered, axis=1)
    Y_norm = np.linalg.norm(Y_centered, axis=1)
    
    # Normalize the dot product by the outer product of the norms
    correlation_matrix = dot_product / np.outer(X_norm, Y_norm)
    
    return correlation_matrix

def pairwise_manhattan_distance(X, Y):
    # Get the number of rows in X and Y
    num_rows_X, num_rows_Y = X.shape[0], Y.shape[0]
    
    # Initialize an empty distance matrix
    distance_matrix = np.zeros((num_rows_X, num_rows_Y))
    
    # Calculate the Manhattan distance using broadcasting
    for i in range(num_rows_X):
        distance_matrix[i, :] = np.sum(np.abs(X[i, np.newaxis, :] - Y), axis=1)
    
    return distance_matrix

def pairwise_euclidean_distance(X, Y):
    return np.sqrt(((X[:, np.newaxis] - Y) ** 2).sum(axis=2))

pairwise_f = {'pearson': pairwise_pearson_correlation, 
              'spearman': pairwise_spearman_correlation, 
             'manhattan': pairwise_manhattan_distance, 
             'euclidean': pairwise_euclidean_distance}

def calculate_pairwise_distances(df1, 
                                 df2: Optional[pd.DataFrame] = None,
                                 distance_metric:Literal['euclidean', 'manhattan', 'pearson', 'spearman'] = 'euclidean', 
                                 axis: Literal[0,1]=0, 
                                 invert_corr: bool = True):
    """Calculate pairwise distances between two dataframes using various distance metrics.

    Parameters
    ----------
    df1 : pandas.DataFrame
        The first dataframe containing the first set of vectors.
    df2 : pandas.DataFrame
        The second dataframe containing the second set of vectors. If none, calculates the pairwise distances within df1.
    axis : Literal[0,1], optional
        The axis along which to compute the distances (0 for rows, 1 for columns), by default 0
    distance_metric : _type_, optional
        The distance metric to use. Options are 'euclidean', 'manhattan', 'pearson', 'spearman'. Default is 'euclidean'., by default Literal['euclidean', 'manhattan', 'pearson', 'spearman']
    invert_corr:bool=True
        Takes the negative of the value for correlation based metrics if True (larger number is a larger distance)
        
    Returns
    ----------
    pandas.DataFrame
        A DataFrame of pairwise distances between the vectors from df1 and df2. Rows are entries from 
        the first dataframe and columns are entries from the second dataframe. 
    """
    # Transpose if comparing along columns (axis=0)
    if df2 is None:
        df2 = df1.copy()
    if axis == 1:
        df1 = df1.T
        df2 = df2.T
    
    pairwise_distances = pd.DataFrame(pairwise_f[distance_metric](df1.values, df2.values), 
                                      index = df1.index, 
                                      columns = df2.index)
        
    if distance_metric in ['pearson', 'spearman'] and invert_corr:
        pairwise_distances *= -1

    return pairwise_distances

def get_upper_triangle(df):
    """Returns the upper triangle (excluding diagonal) of a pandas dataframe as a numpy vector."""
    mask = np.triu(np.ones(df.shape), k=1).astype(bool)
    return df.where(mask).stack().values


def _get_pairwise_distance(X: pd.DataFrame, 
                          md: pd.DataFrame, 
                          label: str, 
                         comb, distance_metric):
    samples_1 = md[md[label] == comb[0]].index.tolist()
    samples_2 = md[md[label] == comb[1]].index.tolist()
    
    pairwise_distances = calculate_pairwise_distances(df1 = X.loc[samples_1,:], 
                            df2 = X.loc[samples_2,:], 
                            axis = 0, 
                            distance_metric = distance_metric, 
                            invert_corr = True).values
    # this does same as the line above, maybe a bit faster, but without the invert_corr argument
#     pairwise_distances = pairwise_f[distance_metric](X.loc[samples_1,:].values, X.loc[samples_2,:].values)
    
    return pairwise_distances

def cohen_d(vector_1, vector_2):
    # Calculate the means of the two vectors
    mean1 = np.mean(vector_1)
    mean2 = np.mean(vector_2)

    # Calculate the standard deviations of the two vectors
    std1 = np.std(vector_1, ddof=1)
    std2 = np.std(vector_2, ddof=1)

    # Calculate the pooled standard deviation
    n1 = len(vector_1)
    n2 = len(vector_2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    return (mean1 - mean2) / pooled_std


def calculate_confidence(mean, std, n, confidence_level = 0.95):
    confidence_level = 0.95
    alpha = 1 - confidence_level
    t_score = stats.t.ppf(1 - alpha/2, df=n-1)
    margin_of_error = t_score * (std / np.sqrt(n))
    confidence_interval = (mean - margin_of_error, mean + margin_of_error)
    return confidence_interval

def calculate_null_pval(null_dist: List[float], actual_val: float, 
                        alternative: Literal['greater', 'two-sided', 'less']):
    """Calculates a p-value given a null distribution

    Parameters
    ----------
    null_dist : List[float]
        the null distribution
    actual_val : float
        the actual value
    alternative : Literal['greater', 'two-sided', 'less']
        defines the alternative hypothesis 

    Returns
    -------
    pval : float
        the p-value
    """
    # calculate the p-value
    if alternative == 'greater': 
        pval = (np.sum(null_dist >= actual_val) + 1) / (len(null_dist) + 1)
    elif alternative == 'two-sided':
        pval = 2 * min((np.sum(null_dist >= actual_val) + 1) / (len(null_dist) + 1),
                          ((np.sum(null_dist <= actual_val) + 1) / (len(null_dist) + 1)))
    elif alternative == 'less':
        pval = (np.sum(null_dist <= actual_val) + 1) / (len(null_dist) + 1)
    return pval

def _par_pairwisedistance(comb, X, obs, label, normal, seed, n_perm, null_samples, null_md, alternative, distance_metric, 
                         exclude_null_cells, label_counts):
    pairwise_distances = _get_pairwise_distance(X = X, md = obs, label = label, comb = comb, 
                                               distance_metric = distance_metric)
    if comb[0] == comb[1]:
        pdf = np.triu(pairwise_distances, k = 1).flatten()
    else:
        pdf = pairwise_distances.flatten()
    if normal: 
        central = np.mean(pdf)
        std = np.std(pdf, ddof=1) 
        cl, cu = calculate_confidence(central, std, n = pdf.shape[0], confidence_level = 0.95)
    else:
        central = np.median(pdf)
        # boostrapped confidence
        np.random.seed(seed)
        cl, cu = np.percentile([np.median(np.random.choice(pdf, size=len(pdf), replace=True)) for _ in range(1000)], 
                                    [2.5, 97.5])

    # generate the null distribution
    null_centrals = []
    cds = []
    null_md[label + '_null'] = null_md[label].tolist()
    lc = label_counts.loc[list(comb), :] 
    if exclude_null_cells is not None and lc[lc['diff'] != 0].shape[0] != 0:
        n_cells_null = int(lc.og.sum())
        if n_cells_null > null_md.shape[0]:
            n_cells_null = null_md.shape[0]
        if comb[0] != comb[1]:
            frac_vals = lc.og/n_cells_null
            np.random.seed(seed)
            null_cell_labels = list(np.random.choice(frac_vals.index, size=n_cells_null, p=frac_vals.values))
        else:
            null_cell_labels = [comb[0]]*n_cells_null
        null_md[label + '_null'] = null_cell_labels + ['']*(null_md.shape[0] - n_cells_null)
    for i in range(n_perm):
        null_md.index = null_samples[:, i]          
        null_distances = _get_pairwise_distance(X = X, md = null_md, label = label + '_null', comb = comb, 
                                                distance_metric = distance_metric)    
        if comb[0] == comb[1]:
            null_distances = np.triu(null_distances, k = 1).flatten()
        else:
            null_distances = null_distances.flatten()

        

        if normal:
    #         pval = ttest_ind(pdf, null_distances, equal_var = False, random_state = seed, alternative = 'greater').pvalue
            nc = np.mean(null_distances)
            cd = cohen_d(pdf, null_distances)  # positive if actual > null
        else:
    #         pval = mannwhitneyu(distances.actual, distances.null, alternative = 'greater').pvalue
            nc = np.median(null_distances)
            cd, _ = cliffs_delta(pdf, null_distances)  # positive if actual > null

        null_centrals.append(nc)
        cds.append(cd)

    # calculate the p-value
    pval = calculate_null_pval(null_dist = null_centrals, actual_val = central, alternative = alternative)

#     mean_cd = np.mean(cds)
#     cld, cud = calculate_confidence(mean_cd, np.std(cds, ddof=1), n = len(cds), confidence_level = 0.95)
    median_cd = np.median(cds)
    np.random.seed(seed)
    cld, cud = np.percentile([np.median(np.random.choice(cds, size=len(cds), replace=True)) for _ in range(n_perm)], 
                                [2.5, 97.5])
    
    
    return central, pval, cl, cu, median_cd, cld, cud


def _filter_combs(tf_adata, label, comparison_combination_subset, comparison_subset, label_subset, 
                  include_self, use_pcs, rank, n_perm, seed, feature_subset, exclude_null_cells):
    
    tf_adata = tf_adata.copy()
    
    # permute labels for null distribution, independent of specific combinations
    # if move this to after the label_subset filtering, null distribution will only be based on the subset
    null_md = tf_adata.obs.copy()
    
    if label_subset is not None and len(label_subset) > 0:
        tf_adata = tf_adata[tf_adata.obs[tf_adata.obs[label].isin(label_subset)].index, :]
    if feature_subset is not None and len(feature_subset) > 0:
        tf_adata = tf_adata[:, feature_subset]

    if use_pcs:
        if 'X_pca' not in tf_adata.obsm:
            raise ValueError('Please run "preprocess.embed_tf_activity" first.')
        if not rank:
            if 'pca' in tf_adata.uns and 'pca_rank' in tf_adata.uns['pca']:
                rank = tf_adata.uns['pca']['pca_rank']
            else:
                rank = tf_adata.obsm['X_pca'].shape[1] # use all PCs

        X = pd.DataFrame(tf_adata.obsm['X_pca'][:, :rank], 
                         index = tf_adata.obs.index, 
                        columns = ['PC_{}'.format(i + 1) for i in range(rank)])
    else:
        X = tf_adata.to_df()

    # create the label combinations
    if isinstance(tf_adata.obs[label].dtype, pd.CategoricalDtype):
        labels = tf_adata.obs[label].cat.categories.tolist()
    else:
        labels = sorted(set(tf_adata.obs[label]))

    if include_self:
        label_combinations = list(itertools.combinations_with_replacement(labels, 2))
    else:
        label_combinations = list(itertools.combinations(labels, 2))
    if comparison_subset is not None and len(comparison_subset) > 0:
        label_combinations = [comb for comb in label_combinations if comb[0] in comparison_subset or comb[1] in comparison_subset]
    if comparison_combination_subset is not None and len(comparison_combination_subset) > 0:
        label_combinations = [i for i in label_combinations if i in comparison_combination_subset]
        
    # generate the null distribution
    #all_combs = list(set([value for tup in label_combinations for value in tup]))

    if exclude_null_cells is not None:
        null_md.drop(index = exclude_null_cells, inplace = True)
    label_counts = pd.DataFrame({'og': tf_adata.obs[label].value_counts(), 
                                'null': null_md[label].value_counts()})
    label_counts.loc[label_counts['null'].isna(), 'null'] = 0
    label_counts['diff'] = label_counts.og - label_counts.null
    
    null_samples = []
    for i in range(n_perm):
        np.random.seed(seed + i)   
        null_samples.append(np.random.permutation(null_md.index))
    null_samples = np.column_stack(null_samples)
        
    return tf_adata, X, label_combinations, null_md, null_samples, label_counts

def quantify_cluster_distance(tf_adata, 
                              label: str, 
                              comparison_combination_subset: Optional[List[Tuple[str]]] = None,
                              comparison_subset: Optional[List[str]] = None,
                              label_subset: Optional[List[str]] = None,
                              include_self: bool = False, 
                              feature_subset: Optional[List[str]] = None,
                              exclude_null_cells: Optional[List[str]] = None, 
                              distance_metric: Literal['euclidean', 'manhattan', 'pearson', 'spearman'] = 'euclidean',
                              normal: bool = True, 
                              use_pcs: bool = False,
                              rank: int = None, 
                              n_perm: int = 1000, 
                              alternative: Literal['two-sided', 'less', 'greater'] = 'greater', 
                              seed: int = 888, 
                             n_cores: int = 1):
    """Quantify the single-cell pairwise distance between all pairs of cell labels (e.g. cluster). 

    Parameters
    ----------
    tf_adata : AnnData
        AnnData object with input TF activity estimates stored in `.X`, and dimensionality reduction and clustering outputs stored
        output of `preprocess.embed_tf_activity`
    label : str
        the cell grouping label (e.g., cluster)
    comparison_combination_subset : Optional[List[Tuple[str]]]
        this will only make a comparison if it is present as a tuple here
    comparison_subset : Optional[List[str]], optional
        this will only make comparisons if atleast one of the two labels is in `comparison_subset`, by default all comparisons
    label_subset : Optional[List[str]], optional
        this will subset dataset to those with the labels in `label_subset`, by default all labels
    include_self : bool, optional
        whether to include comparisons within the same group (pairwise distances between single-cells in the same group), by default False
    feature_subset : Optional[List[str]], optional
        this will subset dataset to this list of features, by default all features
    exclude_null_cells : Optional[List[str]], optional
        this will exclude specific cells (by their AnnData ID `adata.obs_names`) from the null distribution
        if this excludes more cells than in a comparison, we use a relative proportion
    distance_metric : Literal['euclidean', 'manhattan', 'pearson', 'spearman']
        the distance metric to calculate between cell labels, by default euclidean 
    normal : bool, optional
        whether to assume that pairwise distances are normally distributed or run non-parametric tests, by default True
    use_pcs : bool, optional
        whether to calculate distances on PC space or full feature space
    rank : int, optional
        # of PCs to use when calculating distance, by default the one automatically selected in 
        preprocess.embed_tf_activity
        only used when `use_pcs` = True
    n_perm : int, optional
        number of permutations from which to create the null distribution, by default 100
    alternative : Literal['two-sided', 'less', 'greater'], optional
        specifies the comparison of the actual distance statistic relative the null distribution, by default greater
        **correlation values are inverted such that they should be consistent with the interpretation of euclidean and manhattan distance
    seed : int, optional
        random seed for numpy operations, by default 888, by default 888
    n_cores : int, optional
        parallelize using `n_cores` cores if n_cores > 1 , by default 1

    Returns
    -------
    distances_df : pd.DataFrame
        a dataframe with rows representing each pairwise group comparison and with the following columns:
            - 'central_tendency': mean (normal = True) or median (normal = False) pairwise Euclidean distance between 
            all cells in the two labels
            - 'pval': the p-value of the central tendency with respect to the null distribution
            - 'CL': the lower bound of the 95% confidence interval w.r.t. the central tendency 
            - 'CU': the upper bound of the 95% confidence interval w.r.t. the central tendency
            - 'median_cd':  the median Cohen's D (normal) or Cliff's Delta (not normal) across all comparisons of the 
            actual pairwise distances with the null pairwise distances. Positive values indicate that the actual distribution > null distribution, negative indicate that the null distribution > actual distribution. 
            - 'CL_cd': the lower bound of the 95% confidence interval w.r.t. the mean_cd
            - 'CU_cd': the upper bound of the 95% confidence interval w.r.t. the mean_cd
            - 'BH_FDR': the BH FDR corrected pvalues
    """
#     if distance_metric in ['euclidean', 'manhattan'] and alternative != 'greater':
#         warnings.warn('Recommended to test that the actual distance is greater than (more dissimilar) that of the null')
#     if distance_metric in ['pearson', 'spearman'] and alternative != 'less':
#         warnings.warn('Recommended to test that the actual distance is less than (more dissimilar) that of the null')

    tf_adata, X, label_combinations, null_md, null_samples, label_counts = _filter_combs(tf_adata, label, comparison_combination_subset, 
                                                          comparison_subset, label_subset, include_self, use_pcs, 
                                                          rank, n_perm, seed, feature_subset, 
                                                                          exclude_null_cells = exclude_null_cells)
    
    
    if n_cores is None or n_cores <= 1:
        distances_df = pd.DataFrame(columns = ['central_tendency', 'pval', 'CL', 'CU', 'median_cd', 'CL_cd', 'CU_cd'])
        # iterate through the pairwise cell label combinations
        for comb in tqdm(label_combinations): #tqdm(label_combinations): 
            # get actual value stats
            pairwise_distances = _get_pairwise_distance(X = X, md = tf_adata.obs, label = label, comb = comb, 
                                                       distance_metric = distance_metric)
            if comb[0] == comb[1]:
                pdf = np.triu(pairwise_distances, k = 1).flatten()
            else:
                pdf = pairwise_distances.flatten()
            if normal: 
                central = np.mean(pdf)
                std = np.std(pdf, ddof=1) 
                cl, cu = calculate_confidence(central, std, n = pdf.shape[0], confidence_level = 0.95)
            else:
                central = np.median(pdf)
                # boostrapped confidence
                np.random.seed(seed)
                cl, cu = np.percentile([np.median(np.random.choice(pdf, size=len(pdf), replace=True)) for _ in range(n_perm)], 
                                            [2.5, 97.5])

            # generate the null distribution
            null_centrals = []
            cds = []
            
            # generate labels in proportion to what they were prior to dropping, if they do not exist in the numbers as before
            # otherwise, assign as many cells as before
            null_md[label + '_null'] = null_md[label].tolist()
            lc = label_counts.loc[list(comb), :] 
            if exclude_null_cells is not None and lc[lc['diff'] != 0].shape[0] != 0:
                n_cells_null = int(lc.og.sum())
                if n_cells_null > null_md.shape[0]:
                    n_cells_null = null_md.shape[0]
                if comb[0] != comb[1]:
                    frac_vals = lc.og/n_cells_null
                    np.random.seed(seed)
                    null_cell_labels = list(np.random.choice(frac_vals.index, size=n_cells_null, p=frac_vals.values))
                else:
                    null_cell_labels = [comb[0]]*n_cells_null
                null_md[label + '_null'] = null_cell_labels + ['']*(null_md.shape[0] - n_cells_null)
            
            for i in range(n_perm):
                null_md.index = null_samples[:, i]          
                null_distances = _get_pairwise_distance(X = X, md = null_md, label = label + '_null', comb = comb, 
                                                        distance_metric = distance_metric)
                if comb[0] == comb[1]:
                    null_distances = np.triu(null_distances, k = 1).flatten()
                else:
                    null_distances = null_distances.flatten()

                if normal:
            #         pval = ttest_ind(pdf, null_distances, equal_var = False, random_state = seed, alternative = 'greater').pvalue
                    nc = np.mean(null_distances)
                    cd = cohen_d(pdf, null_distances)  # positive if actual > null
                else:
            #         pval = mannwhitneyu(distances.actual, distances.null, alternative = 'greater').pvalue
                    nc = np.median(null_distances)
                    cd, _ = cliffs_delta(pdf, null_distances)  # positive if actual > null

                null_centrals.append(nc)
                cds.append(cd)

            # calculate the p-value
            pval = calculate_null_pval(null_dist = null_centrals, actual_val = central, alternative = alternative)

#             mean_cd = np.mean(cds)
#             cld, cud = calculate_confidence(mean_cd, np.std(cds, ddof=1), n = len(cds), confidence_level = 0.95)
            median_cd = np.median(cds)
            np.random.seed(seed)
            cld, cud = np.percentile([np.median(np.random.choice(cds, size=len(cds), replace=True)) for _ in range(n_perm)], 
                                        [2.5, 97.5])
            
            distances_df.loc['-'.join(comb), :] = [central, pval, cl, cu, median_cd, cld, cud]
    else:
        pool = multiprocessing.Pool(processes = n_cores)
        enc = exclude_null_cells if exclude_null_cells is None else 1
        res = pool.starmap(_par_pairwisedistance, zip(label_combinations, repeat(X), repeat(tf_adata.obs), repeat(label),
                                                     repeat(normal), repeat(seed), repeat(n_perm), repeat(null_samples), 
                                                      repeat(null_md),
                                                     repeat(alternative), repeat(distance_metric), 
                                                     repeat(enc), repeat(label_counts)))
        distances_df = pd.DataFrame(res,
                                    columns = ['central_tendency', 'pval', 'CL', 'CU', 'median_cd', 'CL_cd', 'CU_cd'],
                                    index = ['-'.join(lc) for lc in label_combinations])
        
    distances_df['BH_FDR'] = smm.multipletests(distances_df.pval, alpha=0.1, method='fdr_bh')[1]

    return distances_df

def calculate_emd(df_1, df_2, 
                  emd_loss_fn = default_emd_loss_fn, device = default_device):
    df_1_tensor = torch.tensor(df_1.values, device = device, dtype = torch.float32)
    df_2_tensor = torch.tensor(df_2.values, device = device, dtype = torch.float32)
    
    # # normalization for different sample sizes is handled internally by SamplesLoss
#     n_1, n_2 = df_1_tensor.shape[0], df_2_tensor.shape[0]
#     # normalize if sample sizes are different
#     if n_1 != n_2:
#         print('wee')
#         weights_1 = torch.ones(n_1) / n_1
#         weights_2 = torch.ones(n_2) / n_2
#         emd_loss = loss_fn(weights_1, df_1_tensor, weights_2, df_2_tensor)
#     else:
#         emd_loss = loss_fn(df_1_tensor, df_2_tensor)
    
    return emd_loss_fn(df_1_tensor, df_2_tensor).detach().item()

def _par_emd(comb, X, obs, label, n_perm, null_samples, null_md, alternative, device, emd_loss_fn, exclude_null_cells, label_counts):
    samples_1 = obs[obs[label] == comb[0]].index.tolist()
    samples_2 = obs[obs[label] == comb[1]].index.tolist()

    emd = calculate_emd(df_1 = X.loc[samples_1,:],
                  df_2 = X.loc[samples_2,:], 
                  device = device, emd_loss_fn = emd_loss_fn)

    null_emds = []
    null_md[label + '_null'] = null_md[label].tolist()
    lc = label_counts.loc[list(comb), :]
    if exclude_null_cells is not None and lc[lc['diff'] != 0].shape[0] != 0:
        frac_vals = lc.og/lc.og.sum()
        np.random.seed(seed)
        null_md[label + '_null'] = np.random.choice(frac_vals.index, size=null_md.shape[0], p=frac_vals.values)

    for i in range(n_perm):
        null_md.index = null_samples[:, i]
        samples_1 = null_md[null_md[label + '_null'] == comb[0]].index.tolist()
        samples_2 = null_md[null_md[label + '_null'] == comb[1]].index.tolist()
        null_emd = calculate_emd(df_1 = X.loc[samples_1,:],
                                 df_2 = X.loc[samples_2,:], 
                                 device = device, emd_loss_fn = emd_loss_fn)
        null_emds.append(null_emd)
    pval = calculate_null_pval(null_dist = np.array(null_emds), actual_val = emd, alternative = alternative)

    return emd, pval


def quantify_emd(tf_adata, 
                              label: str, 
                              comparison_combination_subset: Optional[List[Tuple[str]]] = None,
                              comparison_subset: Optional[List[str]] = None,
                              label_subset: Optional[List[str]] = None,
                              include_self: bool = False, 
                              feature_subset: Optional[List[str]] = None,
                 exclude_null_cells: Optional[List[str]] = None,
                              use_pcs: bool = False,
                              rank: int = None, 
                              n_perm: int = 1000, 
                              alternative: Literal['two-sided', 'less', 'greater'] = 'greater', 
                              seed: int = 888,
                               n_cores: int = 1,
                               emd_loss_fn = default_emd_loss_fn, 
                               device = default_device):
    """Quantify the earth mover's distance between all pairs of cell labels (e.g. cluster). 

    Parameters
    ----------
    tf_adata : AnnData
        AnnData object with input TF activity estimates stored in `.X`, and dimensionality reduction and clustering outputs stored
        output of `preprocess.embed_tf_activity`
    label : str
        the cell grouping label (e.g., cluster)
    comparison_combination_subset : Optional[List[Tuple[str]]]
        this will only make a comparison if it is present as a tuple here
    comparison_subset : Optional[List[str]], optional
        this will only make comparisons if atleast one of the two labels is in `comparison_subset`, by default all comparisons
    label_subset : Optional[List[str]], optional
        this will subset dataset to those with the labels in `label_subset`, by default all labels
    include_self : bool, optional
        whether to include comparisons within the same group (pairwise distances between single-cells in the same group), by default False
    feature_subset : Optional[List[str]], optional
        this will subset dataset to this list of features, by default all features
    exclude_null_cells : Optional[List[str]], optional
        this will exclude specific cells (by their AnnData ID `adata.obs_names`) from the null distribution
    use_pcs : bool, optional
        whether to calculate distances on PC space or full feature space
    rank : int, optional
        # of PCs to use when calculating distance, by default the one automatically selected in 
        preprocess.embed_tf_activity
        only used when `use_pcs` = True
    n_perm : int, optional
        number of permutations from which to create the null distribution, by default 100
    alternative : str, optional
        specifies the comparison of the actual distance statistic to the null distribution, by default greater
        even correlations should be greater, because we invert their values (switch the signs)
    seed : int, optional
        random seed for numpy operations, by default 888, by default 888
    n_cores : int, optional
        parallelize using `n_cores` cores if n_cores > 1 , by default 1
    emd_loss_fn :
    
    device :


    Returns
    -------
    distances_df : pd.DataFrame
        a dataframe with rows representing each pairwise group comparison and with the following columns:
            - 'emd': the earth mover's distance between cells in the two labels
            - 'pval': the p-value of the central tendency with respect to the null distribution
            - 'BH_FDR': the BH FDR corrected pvalues
    """

    tf_adata, X, label_combinations, null_md, null_samples, label_counts = _filter_combs(tf_adata, label, comparison_combination_subset, 
                                                          comparison_subset, label_subset, include_self, use_pcs, 
                                                          rank, n_perm, seed, feature_subset, 
                                                                          exclude_null_cells = exclude_null_cells)
    pvals = dict()
    if n_cores is not None and n_cores > 1 and device == 'cpu':
        pool = multiprocessing.Pool(processes = n_cores)
        enc = exclude_null_cells if exclude_null_cells is None else 1
        res = pool.starmap(_par_emd, zip(label_combinations, repeat(X), repeat(tf_adata.obs), repeat(label),
                                                     repeat(n_perm), repeat(null_samples), 
                                                      repeat(null_md),
                                                     repeat(alternative), repeat(device), repeat(emd_loss_fn), 
                                        repeat(enc), repeat(label_counts)))
        distances_df = pd.DataFrame(res,
                            columns = ['emd', 'pval'],
                            index = ['-'.join(lc) for lc in label_combinations])
        
    else:
        distances_df = pd.DataFrame(columns = ['emd', 'pval'])
        for comb in tqdm(label_combinations):
            samples_1 = tf_adata.obs[tf_adata.obs[label] == comb[0]].index.tolist()
            samples_2 = tf_adata.obs[tf_adata.obs[label] == comb[1]].index.tolist()

            emd = calculate_emd(df_1 = X.loc[samples_1,:],
                          df_2 = X.loc[samples_2,:], 
                          device = device, emd_loss_fn = emd_loss_fn)

            null_emds = []
            null_md[label + '_null'] = null_md[label].tolist()
            lc = label_counts.loc[list(comb), :]
            if exclude_null_cells is not None and lc[lc['diff'] != 0].shape[0] != 0:
                frac_vals = lc.og/lc.og.sum()
                np.random.seed(seed)
                null_md[label + '_null'] = np.random.choice(frac_vals.index, size=null_md.shape[0], p=frac_vals.values)

            for i in range(n_perm):
                null_md.index = null_samples[:, i]
                samples_1 = null_md[null_md[label + '_null'] == comb[0]].index.tolist()
                samples_2 = null_md[null_md[label + '_null'] == comb[1]].index.tolist()
                null_emd = calculate_emd(df_1 = X.loc[samples_1,:],
                                         df_2 = X.loc[samples_2,:], 
                                         device = device, emd_loss_fn = emd_loss_fn)
                null_emds.append(null_emd)

            pval = calculate_null_pval(null_dist = np.array(null_emds), actual_val = emd, alternative = alternative)
    
    
            distances_df.loc['-'.join(comb), :] = [emd, pval]
        
    distances_df['BH_FDR'] = smm.multipletests(distances_df.pval, alpha=0.1, method='fdr_bh')[1]
    return distances_df

# def compute_centroid_distances(adata, 
#                                column_name: str, 
#                                n_pcs: Optional[int] = None):
#     """
#     Computes the pairwise Euclidean distances between the centroids of groups defined 
#     by a metadata column in PCA space on the first n PCs. 
    
#     Parameters:
#     adata : anndata.AnnData
#         AnnData object containing PCA results and metadata.
#     column_name : str
#         The column name in .obs for grouping observations.
#     n_pcs : int, optional
#         The number of PCs to calculate the Euclidean distance on, by default all PCs available

#     Returns:
#     pd.DataFrame
#         A DataFrame containing the pairwise Euclidean distances between centroids.
#     """
#     if n_pcs is None:
#         if 'pca_rank' in adata.uns['pca'].keys():
#             n_pcs = adata.uns['pca']['pca_rank']
#         else:
#             n_pcs = adata.obsm['X_pca'].shape[1]
    
#     # Step 1: Extract the PCA coordinates for the first n PCs
#     pca_coords = adata.obsm['X_pca'][:, :n_pcs]  # Assuming PCA is stored in 'X_pca'

#     # Step 2: Get the metadata for the specified column
#     annotations = adata.obs[column_name]

#     # Step 3: Compute the centroids for each group
#     centroids = pd.DataFrame(pca_coords, index=annotations).groupby(level=0).mean()

#     # Step 4: Calculate pairwise Euclidean distances between centroids
#     distances = pdist(centroids, metric='euclidean')
#     distance_matrix = pd.DataFrame(squareform(distances), index=centroids.index, columns=centroids.index)

#     return distance_matrix

def cross_condition_distances(adata, column_1: str, column_2: str, n_pcs: Optional[int] = None):
    """
    Computes the Euclidean distances between centroids of groups defined by `column_1` 
    within different conditions specified by `column_2` in PCA space on the first n PCs.
    
    Parameters:
    adata : anndata.AnnData
        AnnData object containing PCA results and metadata.
    column_1 : str
        The column name in .obs for grouping observations (e.g., 'seurat_annotations').
    column_2 : str
        The column name in .obs for different conditions (e.g., 'stim').

    Returns:
    pd.DataFrame
        A DataFrame containing the Euclidean distances between centroids for each group in column_1 
        across different conditions in column_2.
    """
    if n_pcs is None:
        if 'pca_rank' in adata.uns['pca'].keys():
            n_pcs = adata.uns['pca']['pca_rank']
        else:
            n_pcs = adata.obsm['X_pca'].shape[1]
    # Step 1: Extract the PCA coordinates for the first n PCs
    pca_coords = adata.obsm['X_pca'][:, :n_pcs]  # Assuming PCA is stored in 'X_pca'

    # Step 2: Combine the metadata columns to create a unique identifier for each group
    combined_annotations = adata.obs[column_1].astype(str) + "_" + adata.obs[column_2].astype(str)

    # Step 3: Compute the centroids for each unique group (combined column_1 and column_2)
    centroids = pd.DataFrame(pca_coords, index=combined_annotations).groupby(level=0).mean()

    # Step 4: Separate centroids back into their component groups
    centroid_groups = pd.DataFrame(pd.DataFrame(centroids.index)[0].apply(lambda x: x.split('_')).tolist())
    centroids.index = pd.MultiIndex.from_frame(centroid_groups, names=[column_1, column_2])

    # Step 5: Initialize a DataFrame to store distances between conditions in column_2
    distance_results = []

    # Step 6: Calculate the Euclidean distances for the same groups in column_1 across different conditions in column_2
    for group in centroids.index.get_level_values(column_1).unique():
        # Extract the centroids for a specific group in column_1 across different conditions
        subset = centroids.loc[group]

        # Ensure there are at least two conditions to compare
        if len(subset) > 1:
            conditions = subset.index.get_level_values(column_2)
            for i, cond1 in enumerate(conditions):
                for cond2 in conditions[i + 1:]:
                    dist = euclidean(subset.loc[cond1], subset.loc[cond2])
                    distance_results.append([group, cond1, cond2, dist])

    # Convert results to a DataFrame for easy visualization
    distance_df = pd.DataFrame(distance_results, columns=[column_1, f'{column_2}_1', f'{column_2}_2', 'distance'])

    return distance_df


def get_k_neighbors(adata, cell_id, k):
    """Identifies the k nearest neighbors of a cell

    Parameters
    ----------
    adata : _type_
        AnnData object with neighbor graph already calculated
    cell_id : _type_
        ID of cell upon which to calculate the nearest neighbors
    k : _type_
        number of nearest neighbors to return

    Returns
    -------
    neighbor_ids
        the k nearest neighbors
    """
    if k > adata.uns['neighbors']['params']['n_neighbors']:
        raise ValueError('You have selected a k larger than the size of the local neighborhood')
    cell_index = np.where(adata.obs_names == cell_id)[0][0]
    neighbors_matrix = adata.obsp['connectivities']
    neighbor_indices = neighbors_matrix[cell_index].toarray().argsort()[0][-k-1:-1]
    neighbor_ids = adata.obs_names[neighbor_indices]
    
    return neighbor_ids

def get_alignment_score(adata, batch_key, k: int = 15, normalize: bool = True):
    """Modified alignment score as described in https://doi.org/10.1038/s41587-020-00748-9. This is modified from
    the alignment score described in https://doi.org/10.1038/nbt.4096.

    Parameters
    ----------
    adata : _type_
        anndata object
    batch_key : _type_
        batch column key in adata.obs
    k : int, optional
        number of nearest neighbors to use when calculating the score, by default 15
    normalize : bool, optional
        whether to normalize the alignment score to a maximum of 1, by default True

    Returns
    -------
    alignment_score : float
        the alignment score. Minimum value is 0. Highest value is 1 if normalized. Larger values mean
        fewer batch effects
    """
    w = adata.obs[batch_key].value_counts()
    w /= w.sum()
    knn_tracker = dict(zip(adata.obs[batch_key].unique(), adata.obs[batch_key].nunique() * [[]]))

    for cell_id in tqdm(adata.obs.index):
        batch = adata.obs.loc[cell_id, batch_key]
        knn = get_k_neighbors(adata, cell_id = cell_id, k = k)
        same_batch_count = (adata.obs.loc[knn, batch_key] == batch).sum()

        knn_tracker[batch] += [same_batch_count]

    for batch, counts in knn_tracker.items():
        knn_tracker[batch] = np.mean(counts)

    alignment_score = 0
    for batch, x_i in knn_tracker.items():
        w_i = w.loc[batch]
        alignment_score += (w_i * (1 - (x_i - (w_i*k))/(k - (w_i*k))))

    if normalize:
        knn_tracker = {k: 0 for k in knn_tracker}

        max_alignment_score = 0
        for batch, x_i in knn_tracker.items():
            w_i = w.loc[batch]
            max_alignment_score += (w_i * (1 - (x_i - (w_i*k))/(k - (w_i*k))))
        alignment_score /= max_alignment_score
    return alignment_score

def discriminator_weight_curve(n_epochs: int,
                               min_penalty_weight: float | int,
                               max_penalty_weight: float | int,
                               a: float | int = 1,
                               b: float | int = 0.3, 
                              curve_type: Literal['power', 'exponential'] = 'power'):
    """Generates an exponential curve across epochs which can be used as the penalty weight for 
    adverserial training. This allows the discriminator to learn in preliminary epochs, then 
    the VAE/generator to learn in a manner that tricks the discriminator in later epochs.

    Parameters
    ----------
    n_epochs : int
        the number of epochs
    min_penalty_weight : float | int
        the minimum starting penalty 
    max_penalty_weight : float | int
        the maximum starting penalty
    a : float | int, optional
        'a' parameter in the power curve y = a * x^b, by default 1
        'a' parameter in eponential curve y = a * b ^ x
    b : float | int, optional
        'b' parameter in the power curve y = a * x^b, by default 0.3
        'b' parameter in eponential curve y = a * b ^ x
    """
    
    
    """Generates an exponential curve from min to max penalty weight across epochs. Used to 
    incrementally increase weight during adverserial training, which allows discriminator to 
    learn at first, then allows generator to learn how to trick the discriminator."""
    
    epochs = np.linspace(0, n_epochs- 1, n_epochs)
    if curve_type == 'power':
        dpw = a * (epochs ** b)
        dpw = min_penalty_weight + (max_penalty_weight - min_penalty_weight) * (dpw / dpw.max()) # normalize        
    elif curve_type == 'exponential':
        dpw = a * np.exp(b * epochs)
        dpw = (dpw - dpw.min()) / (dpw.max() - dpw.min()) # normalize
        dpw = min_penalty_weight + (max_penalty_weight - min_penalty_weight) * dpw
    discriminator_penalty_weight = dpw.tolist()
    
    return discriminator_penalty_weight





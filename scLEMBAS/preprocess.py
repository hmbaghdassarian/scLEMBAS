"""
Preprocessing functions for single-cell AnnData objects.
"""
import itertools
from itertools import repeat
import multiprocessing
from tqdm import tqdm
import warnings
from typing import List, Literal, Optional


import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind, rankdata
from scipy import stats
import statsmodels.stats.multitest as smm
from cliffs_delta import cliffs_delta

from sklearn.decomposition import PCA
from kneed import KneeLocator
import decoupler as dc
from decoupler.pre import extract

from anndata import AnnData
import scanpy as sc

grn_link = ppi_link = 'https://zenodo.org/records/11477837/files/grn_organism_06_04_24.csv'

def get_tf_activity(adata, organism: str, grn = 'collectri', 
                    verbose: bool = True, min_n: int = 5, use_raw: bool = False,
                    filter_pvals: bool = False, pval_thresh: float = 0.05,
                    hvg: bool = False, static: bool = True,
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
    kwargs : 
        passed to  `decoupler.decouple`.

    Returns
    -------
    estimate : DataFrame
        Consensus TF activity scores. Stored in `adata.obsm['consensus_estimate']`.
    pvals : DataFrame
        Obtained TF activity p-values. Stored in `adata.obsm['consensus_pvals']`.
    """
    
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
        print('Running consensus.')
    
    # # unnecessary, this is the default behavior    
    # if not kwargs:
    #     kwargs = {'methods': ['lm', 'ulm', 'wsum'], 
    #               'cns_metds': ['lm', 'ulm', 'wsum_norm']}
    # else:
    #     if 'methods' not in kwargs:
    #         kwargs['methods'] = ['lm', 'ulm', 'wsum']
    #     if 'cns_methods' not in kwargs and kwargs['methods'] == ['lm', 'ulm', 'wsum']:
    #         kwargs['cns_metds'] = ['lm', 'ulm', 'wsum_norm']

    dc.decouple(mat=adata, net=net, source='source', target='target', weight='weight', consensus = True,
                      min_n=min_n, verbose=verbose, use_raw=use_raw, **kwargs)

    if filter_pvals:
        estimate_key = [k for k in adata.obsm if k.endswith('_estimate')]
        pvals_key = [k for k in adata.obsm if k.endswith('_pvals')]
        
        if len(estimate_key) > 0 or len(pvals_key) > 0:
            warnings.warn('Multiple TF estimates/pvals, choosing first')
            
        adata.obsm[estimate_key[0]][adata.obsm[pvals_key[0]] > pval_thresh] = 0

    return adata

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
        
        
def _pca_simple(adata: AnnData, n_components: int = 50, random_state: int = 888):
    """Minimal re-implementation of scanpy's PCA with default parameters that returns the pca object.

    Parameters
    ----------
    adata : AnnData
        data matrix in `.X` of shape n_obs × n_vars. Rows correspond to cells and columns to features.
    n_components : int, optional
        Number of principal components to compute, by default 50
    random_state : int, optional
        Change to use different initial states for the optimization, by default 888
    """
    pca_mod = PCA(n_components=n_components, random_state = random_state)
    pca_mod.fit(adata.X)
    X_pca = pca_mod.transform(adata.X) # need to separate fit_transform otherwise won't be able to reproducibly run .transform
    adata.obsm["X_pca"] = X_pca
    adata.varm["PCs"] = pca_mod.components_.T
    
    uns_entry = {
        "params": {
            "zero_center": True,
            "use_highly_variable": False,
            "mask": None,
        },
        "variance": pca_mod.explained_variance_,
        "variance_ratio": pca_mod.explained_variance_ratio_,
        "pca_mod": pca_mod
    }
    adata.uns["pca"] = uns_entry

def _compute_elbow(adata, curve='convex', direction='decreasing', **kwargs):
    '''Computes the elbow of a curve. Adapted from cell2cell (https://github.com/earmingol/cell2cell/).

    Parameters
    ----------
    adata : AnnData
        AnnData object with computed pca variance in `.uns['pca']['variance_ratio']`. 

    curve : str, default='convex'
        If curve='concave', kneed will detect knees. If curve='convex',
        it will detect elbows.

    direction : str, default='decreasing'
        The direction parameter describes the line from left to right on the
        x-axis. If the knee/elbow you are trying to identify is on a positive
        slope use direction='increasing', if the knee/elbow you are trying to
        identify is on a negative slope, use direction='decreasing'.

    kwargs : 
        passed to  `kneed.KneeLocator`.

    Returns
    -------
    rank : int
        principle component where the elbow is located in the curve.
    '''
    variance_ratio = adata.uns['pca']['variance_ratio']
    pcs = np.array(range(len(variance_ratio))) + 1
    kneedle = KneeLocator(x = pcs, y = variance_ratio, curve=curve, direction=direction, **kwargs)
    rank = kneedle.elbow
    return rank

def embed_tf_activity(tf_adata: AnnData, scanpy_pca: bool = False):
    """Runs dimensionality reduction and clustering of cells from their TF activity using default scanpy parameters.

    Parameters
    ----------
    adata : AnnData
        AnnData object with TF activity scores stored in `.X`
    scanpy_pca : bool, optional
        whether to use scanpy's PCA (True) or sklearn (False), by default False
        Using scanpy's, sometimes projecting single vectors into PCA space causes issues which can be avoided with sklearn

    Returns
    -------
    tf_adata : AnnData
        AnnData object with dimensionality reduction and clustering outputs stored in default scanpy locations. 
        Cluster labels on TF activity space are stores in `adata.obs['TF_clusters']`
    """
    
    if scanpy_pca:
        sc.tl.pca(data = tf_adata)
    else:
        _pca_simple(adata = tf_adata) 
    
    pc_rank = _compute_elbow(adata = tf_adata)
    tf_adata.uns["pca"]['pca_rank'] = pc_rank

    # if not np.allclose(tf_adata.obsm['X_pca'], tf_adata.uns['pca']['pca_mod'].transform(tf_adata.X)): 
    #     raise ValueError('Unexpected disagreement when running PCA.transform')
    
    sc.pp.neighbors(adata = tf_adata, n_pcs=pc_rank) # construct neighborhood graph
    sc.tl.umap(adata = tf_adata) # run UMAP
    sc.tl.leiden(adata = tf_adata) # cluster

    tf_adata.obs.rename(columns = {'leiden': 'TF_clusters'}, inplace = True)
    
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

def spearman_rank_correlation(X, Y):
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


def _get_pairwise_distance(X: pd.DataFrame, 
                          md: pd.DataFrame, 
                          label: str, 
                         comb, distance_metric):
    samples_1 = md[md[label] == comb[0]].index.tolist()
    samples_2 = md[md[label] == comb[1]].index.tolist()

    if distance_metric == 'euclidean':
        pairwise_distances = np.sqrt(((X.loc[samples_1,:].values[:, np.newaxis] - X.loc[samples_2,:].values) ** 2).sum(axis=2))
#     pairwise_distances = pd.DataFrame(pairwise_distances, index = samples_1, columns = samples_2)
    elif distance_metric == 'pearson':
        pairwise_distances = pairwise_pearson_correlation(X.loc[samples_1,:].values, X.loc[samples_2,:].values)
    elif distance_metric == 'spearman':
        pairwise_distances = spearman_rank_correlation(X.loc[samples_1,:].values, X.loc[samples_2,:].values)
    elif distance_metric == 'manhattan':
        pairwise_distances = pairwise_manhattan_distance(X.loc[samples_1,:].values, X.loc[samples_2,:].values)
    
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

def _par_pairwisedistance(comb, X, obs, label, normal, seed, n_perm, null_samples, null_md, alternative, distance_metric):
    pairwise_distances = _get_pairwise_distance(X = X, md = obs, label = label, comb = comb, 
                                               distance_metric = distance_metric)
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
    for i in range(n_perm):
        null_md.index = null_samples[:, i]
        null_distances = _get_pairwise_distance(X = X, md = null_md, label = label, comb = comb, 
                                                distance_metric = distance_metric)

    #     distances = pd.DataFrame(np.column_stack((pairwise_distances.flatten(), null_distances.flatten())), 
    #                              columns = ['actual', 'null'])   
        distances = np.column_stack((pdf, null_distances.flatten()))
        if normal:
    #         pval = ttest_ind(distances[:, 0], distances[:, 1], equal_var = False, random_state = seed, alternative = 'greater').pvalue
            nc = np.mean(distances[:, 1])
            cd = cohen_d(distances[:, 0], distances[:, 1])  # positive if actual > null
        else:
    #         pval = mannwhitneyu(distances.actual, distances.null, alternative = 'greater').pvalue
            nc = np.median(distances[:, 1])
            cd, _ = cliffs_delta(distances[:, 0], distances[:, 1])  # positive if actual > null

        null_centrals.append(nc)
        cds.append(cd)

    # calculate the p-value
    if alternative == 'greater': 
        pval = (np.sum(null_centrals >= central) + 1) / (len(null_centrals) + 1)
    elif alternative == 'two-sided':
        pval = 2 * min((np.sum(null_centrals >= central) + 1) / (len(null_centrals) + 1),
                          ((np.sum(null_centrals <= central) + 1) / (len(null_centrals) + 1)))
    elif alternative == 'less':
        pval = (np.sum(null_centrals <= central) + 1) / (len(null_centrals) + 1)

#     mean_cd = np.mean(cds)
#     cld, cud = calculate_confidence(mean_cd, np.std(cds, ddof=1), n = len(cds), confidence_level = 0.95)
    median_cd = np.median(cds)
    np.random.seed(seed)
    cld, cud = np.percentile([np.median(np.random.choice(cds, size=len(cds), replace=True)) for _ in range(n_perm)], 
                                [2.5, 97.5])
    
    
    return central, pval, cl, cu, median_cd, cld, cud


def quantify_cluster_distance(tf_adata, 
                              label: str, 
                              label_subset: Optional[List[str]] = None,
                              distance_metric: Literal['euclidean', 'manhattan', 'pearson', 'spearman'] = 'euclidean',
                              normal: bool = True, 
                              rank: int = None, 
                              n_perm: int = 100, 
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
    label_subset : Optional[List[str]], optional
        this will subset the label pairwise comparisons to those that are present in `label_subset`
    distance_metric : Literal['euclidean', 'pearson']
        the distance metric to calculate between cell labels, by default euclidean 
    normal : bool, optional
        whether to assume that pairwise distances are normally distributed or run non-parametric tests, by default True
    rank : int, optional
        # of PCs to use when calculating distance, by default the one automatically selected in 
        preprocess.embed_tf_activity
    n_prem : int, optional
        number of permutations from which to create the null distribution, by default 100
    alternative : str, optional
        specifies the comparison of the actual distance statistic to the null distribution, by default greater
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
    if distance_metric in ['euclidean', 'manhattan'] and alternative != 'greater':
        warnings.warn('Recommended to test that the actual distance is greater than (more dissimilar) that of the null')
    if distance_metric in ['pearson', 'spearman'] and alternative != 'less':
        warnings.warn('Recommended to test that the actual distance is less than (more dissimilar) that of the null')

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

    # permute labels for null distribution
    null_md = tf_adata.obs.copy()
    null_samples = []
    for i in range(n_perm):
        np.random.seed(seed + i)   
        null_samples.append(np.random.permutation(X.index))
    null_samples = np.column_stack(null_samples)
    
    # create the label combinations
    if isinstance(tf_adata.obs[label].dtype, pd.CategoricalDtype):
        labels = tf_adata.obs[label].cat.categories.tolist()
    else:
        labels = sorted(set(tf_adata.obs[label]))
    label_combinations = list(itertools.combinations(labels, 2))
    if label_subset is not None and len(label_subset) > 0:
        label_combinations = [comb for comb in label_combinations if comb[0] in label_subset or comb[1] in label_subset]
    
    if n_cores is None or n_cores <= 1:
        distances_df = pd.DataFrame(columns = ['central_tendency', 'pval', 'CL', 'CU', 'median_cd', 'CL_cd', 'CU_cd'])
        # iterate through the pairwise cell label combinations
        for comb in tqdm(label_combinations):    
            # get actual value stats
            pairwise_distances = _get_pairwise_distance(X = X, md = tf_adata.obs, label = label, comb = comb, 
                                                       distance_metric = distance_metric)
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
            for i in range(n_perm):
                null_md.index = null_samples[:, i]
                null_distances = _get_pairwise_distance(X = X, md = null_md, label = label, comb = comb, 
                                                        distance_metric = distance_metric)

            #     distances = pd.DataFrame(np.column_stack((pairwise_distances.flatten(), null_distances.flatten())), 
            #                              columns = ['actual', 'null'])   
                distances = np.column_stack((pdf, null_distances.flatten()))
                if normal:
            #         pval = ttest_ind(distances[:, 0], distances[:, 1], equal_var = False, random_state = seed, alternative = 'greater').pvalue
                    nc = np.mean(distances[:, 1])
                    cd = cohen_d(distances[:, 0], distances[:, 1])  # positive if actual > null
                else:
            #         pval = mannwhitneyu(distances.actual, distances.null, alternative = 'greater').pvalue
                    nc = np.median(distances[:, 1])
                    cd, _ = cliffs_delta(distances[:, 0], distances[:, 1])  # positive if actual > null

                null_centrals.append(nc)
                cds.append(cd)

            # calculate the p-value
            if alternative == 'greater': 
                pval = (np.sum(null_centrals >= central) + 1) / (len(null_centrals) + 1)
            elif alternative == 'two-sided':
                pval = 2 * min((np.sum(null_centrals >= central) + 1) / (len(null_centrals) + 1),
                                  ((np.sum(null_centrals <= central) + 1) / (len(null_centrals) + 1)))
            elif alternative == 'less':
                pval = (np.sum(null_centrals <= central) + 1) / (len(null_centrals) + 1)

#             mean_cd = np.mean(cds)
#             cld, cud = calculate_confidence(mean_cd, np.std(cds, ddof=1), n = len(cds), confidence_level = 0.95)
            median_cd = np.median(cds)
            np.random.seed(seed)
            cld, cud = np.percentile([np.median(np.random.choice(cds, size=len(cds), replace=True)) for _ in range(n_perm)], 
                                        [2.5, 97.5])
            
            distances_df.loc['-'.join(comb), :] = [central, pval, cl, cu, median_cd, cld, cud]
    else:
        pool = multiprocessing.Pool(processes = n_cores)
        res = pool.starmap(_par_pairwisedistance, zip(label_combinations, repeat(X), repeat(tf_adata.obs), repeat(label),
                                                     repeat(normal), repeat(seed), repeat(n_perm), repeat(null_samples), 
                                                      repeat(null_md),
                                                     repeat(alternative), repeat(distance_metric)))
        distances_df = pd.DataFrame(res,
                                    columns = ['central_tendency', 'pval', 'CL', 'CU', 'median_cd', 'CL_cd', 'CU_cd'],
                                    index = ['-'.join(lc) for lc in label_combinations])
        
    distances_df['BH_FDR'] = smm.multipletests(distances_df.pval, alpha=0.1, method='fdr_bh')[1]

    return distances_df   


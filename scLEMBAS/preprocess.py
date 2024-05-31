"""
Preprocessing functions for single-cell AnnData objects.
"""
import pandas as pd
import numpy as np
from anndata import AnnData
import scanpy as sc
from sklearn.decomposition import PCA
from kneed import KneeLocator
import decoupler as dc
from decoupler.pre import extract

def get_tf_activity(adata, organism: str, grn = 'collectri', 
                    verbose: bool = True, min_n: int = 5, use_raw: bool = False,
                    filter_pvals: bool = False, pval_thresh: float = 0.05,
                    hvg: bool = False, 
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
        adata = adata[:, adata.var['highly_variable']]
    
    grn_map = {'collectri': dc.get_collectri, 'dorothea': dc.get_dorothea} # get_dorothea returns "A" confidence by default
    net = grn_map[grn](organism=organism, split_complexes=False) # builds on dorothea, used by Saez-Rodriguez lab

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

def embed_tf_activity(adata: AnnData, estimate_key: str = 'consensus_estimate', scanpy_pca: bool = False):
    """Runs dimensionality reduction and clustering of cells from their TF activity using default scanpy parameters.

    Parameters
    ----------
    adata : AnnData
        AnnData object with TF activity scores stored in `.obsm[estimate_key]`.
    estimate_key : str, optional
        `.obsm` key under which TF activity is stored, by default 'consensus_estimate'
        if None, assumes that the anndata object contains the TF activity in `X` 
    scanpy_pca : bool, optional
        whether to youse scanpy's PCA (True) or sklearn (False), by default False
        Using scanpy's, sometimes projecting single vectors into PCA space causes issues which can be avoided with sklearn

    Returns
    -------
    tf_adata : AnnData
        AnnData object with input TF activity estimates stored in `.X`, and dimensionality reduction and clustering outputs stored
        in default scanpy locations. Cluster labels on TF activity space are stores in `adata.obs['TF_clusters']`
    """

    if not estimate_key:
        tf_adata = adata
    else:
        tf_adata = AnnData(adata.obsm[estimate_key])

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

    if estimate_key:
        tf_adata.obs = pd.concat([adata.obs, tf_adata.obs], axis = 1)

    return tf_adata
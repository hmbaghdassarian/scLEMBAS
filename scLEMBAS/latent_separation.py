"""Set of functions for latent space expression shifts of cells."""

from typing import Literal, List
import math
import warnings
from joblib import Parallel, delayed

from tqdm import trange, tqdm
from tqdm_joblib import tqdm_joblib

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import scipy.sparse as sparse

from kneed import KneeLocator

import umap

from cliffs_delta import cliffs_delta

from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold, cross_val_predict, cross_validate
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import clone
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    make_scorer,
    r2_score,
    normalized_mutual_info_score
)

import scipy
from scipy import sparse, stats

# from . import preprocess as pp
from ._scanpy_umap import scanpy_umap    

############################ PCA ############################

# def _pca_simple(adata: AnnData, n_components: int = 50, random_state: int = 888):
#     """Minimal re-implementation of scanpy's PCA with default parameters that returns the pca object.

#     Parameters
#     ----------
#     adata : AnnData
#         data matrix in `.X` of shape n_obs × n_vars. Rows correspond to cells and columns to features.
#     n_components : int, optional
#         Number of principal components to compute, by default 50
#     zero_center: 
#     random_state : int, optional
#         Change to use different initial states for the optimization, by default 888
#     """
#     pca_mod = PCA(n_components=n_components, random_state = random_state)
#     X = adata.X
#     pca_mod.fit(X)
#     X_pca = pca_mod.transform(X) # need to separate fit_transform otherwise won't be able to reproducibly run .transform
#     adata.obsm["X_pca"] = X_pca
#     adata.varm["PCs"] = pca_mod.components_.T
    
#     uns_entry = {
#         "params": {
#             "zero_center": True,
#             "use_highly_variable": False,
#             "mask": None,
#         },
#         "variance": pca_mod.explained_variance_,
#         "variance_ratio": pca_mod.explained_variance_ratio_,
#         "pca_mod": pca_mod
#     }
#     adata.uns["pca"] = uns_entry

def _compute_elbow(adata, curve='concave', direction='increasing', **kwargs):
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
    # scanpy algs are not always monotonic, so use cumulative var instead
    cumulative_variance_ratio = np.cumsum(adata.uns['pca']['variance_ratio']) #adata.uns['pca']['variance_ratio']
    pcs = np.array(range(len(cumulative_variance_ratio))) + 1
    kneedle = KneeLocator(x = pcs, y = cumulative_variance_ratio, curve=curve, direction=direction, **kwargs)
    rank = kneedle.elbow
    return rank

def project_to_pca(X_new, adata):
    """
    Project new data into the PCA space computed by scanpy.
    
    Robustly handles both full-dimensionality and HVG-subset input data.
    
    Parameters
    ----------
    X_new : array-like, shape (n_samples, n_features)
        New data to project. Can be either:
        - Full dimensionality (same as original adata)
        - HVG subset (if PCA was computed with use_highly_variable=True)
    adata : AnnData
        AnnData object containing the PCA results from sc.pp.pca()
        
    Returns
    -------
    X_pca : np.ndarray, shape (n_samples, n_components)
        New data projected into PCA space
    """
    
    # Check if PCA has been run
    if 'pca' not in adata.uns:
        raise ValueError("PCA has not been run on this AnnData object. Run sc.pp.pca() first.")
    
    # Get PCA parameters
    pca_params = adata.uns['pca']['params']
    zero_center = pca_params.get('zero_center', True)
    use_highly_variable = pca_params.get('use_highly_variable', False)
    
    # Convert sparse to dense if needed
    if sparse.issparse(X_new):
        X_new_dense = X_new.toarray()
    else:
        X_new_dense = X_new.copy()
    
    # Get the loadings (always full dimensionality with zero-padding)
    loadings = adata.varm['PCs']
    
    # Handle highly variable genes
    if use_highly_variable and 'highly_variable' in adata.var.columns:
        hv_mask = adata.var['highly_variable'].values
        n_hvg = hv_mask.sum()
        n_total = len(hv_mask)

        # Determine input data type based on shape
        if X_new_dense.shape[1] == n_total:
            X_working = X_new_dense[:, hv_mask]
            loadings_working = loadings[hv_mask, :]
        elif X_new_dense.shape[1] == n_hvg:
            X_working = X_new_dense
            loadings_working = loadings[hv_mask, :]
        else:
            raise ValueError(
                f"Input dimensionality mismatch. Got {X_new_dense.shape[1]} features, "
                f"expected either {n_total} (full) or {n_hvg} (HVG subset)"
            )
    else:
        X_working = X_new_dense
        loadings_working = loadings
    
    # Handle zero centering
    if zero_center:
        # Get original data for computing means
        X_original = adata.X
        if sparse.issparse(X_original):
            X_original = X_original.toarray()
        
        # If we're working with HVGs, subset original data too
        if use_highly_variable and 'highly_variable' in adata.var.columns:
            X_original_for_mean = X_original[:, hv_mask]
        else:
            X_original_for_mean = X_original
        
        # Calculate mean and center
        mean_original = np.mean(X_original_for_mean, axis=0)
        X_centered = X_working - mean_original
        
        # Project
        X_pca = X_centered @ loadings_working
        
    else:
        # Direct projection without centering
        X_pca = X_working @ loadings_working
    
    return X_pca.astype(np.float32)

def embed_adata(adata: AnnData, 
                      cluster_col_name: str = 'TF_clusters', 
                     n_components: int = 50, 
                      pc_rank: str | int = 'automate',
                pc_projection_tol: float | None = 5e-4,
                      scale: bool = False, 
                     resolution: float | List[float] = 1, 
                     nmi_label: str = None, 
                      n_neighbors: int = 15, 
                      run_pca: bool = True,
                     run_umap: bool  = True, 
                     cluster_data: bool = True, 
                seed: int = 888, 
                      pcakwrgs = {}, 
                      umapkwrgs = {}
                     ):
    """Runs PCA/UMAP dimensionality reduction and clustering of cells from their TF activity using scanpy.

    Parameters
    ----------
    adata : AnnData
        AnnData object with matrix to be embedded in `.X`
    cluster_col_name : str, optional
        the name of the leiden cluster column in `adata.obs`, by default 'TF_clusters'
    n_components : int, optional
        Number of principal components to compute, by default 50
    pc_rank : str | int, optional
        the number of PCs (<= n_components) to use for downstream analyses. If 'automate', will automatically estimate
        at the elbow
    pc_projection_tol : str | int, optional
        if not None, will check that the projection matches scanpy's `adata.obsm['X_pca']` output by atol = pc_projection_tol
    scale : bool, optional
        whether to scale the data (True) prior to PCA, by default False
        not currently implemented (will always not scale)
        **Note, default behavorior of scanpy pca is to center but not scale data, as discussed here: https://github.com/scverse/scanpy/issues/2164.
    resolution : float | List[float], optional
        The resolution parameter for leiden clustering. If a list, will iterate through all resolutions, maximizing
        NMI between the clusters and the nmi_label in `adata.obs`
    nmi_label : str, optional
        A column in `adata.obs`. If resolution is a list, will identify the resolution that maximizes the 
        NMI between leiden clusters and nmi_label
    n_neighbors : int
        The number of neighbors to use, passed to `sc.pp.neighbors`, by default 15
    run_pca : bool, optional
        Whether to run umap or not, by default True
    run_umap : bool, optional
        Whether to run umap or not, by default True
    cluster_data : bool, optional
        Wheter to run leiden clustering or not, by default True
    **pcakwrgs : 
        keyword arguments for sc.pp.pca
    **umapkwrgs
        keyword arguments for sc.pp.umap

    Returns
    -------
    adata : AnnData
        AnnData object with dimensionality reduction and clustering outputs stored in default scanpy locations. 
        Cluster labels on TF activity space are stores in `adata.obs['TF_clusters']`
    """
#     if min(adata.shape) > n_comps:
#         n_comps = min(adata.shape)
    if isinstance(resolution, list) and nmi_label not in adata.obs.columns:
        raise ValueError('The nmi_label should be in the AnnData obs')
    if not run_pca and 'pca' not in adata.uns:
        raise ValueError('Need to calculate or run PCA')
    
    if run_pca:
#         if scanpy_pca:
#             sc.tl.pca(data = adata, n_comps = n_components)
#         else:
#             _pca_simple(adata = adata, n_components = n_components) 
        if scale:
            raise ValueError('Internal: this functionality needs to be checked, particularly w.r.t project_to_pca')
            zero_center = pcakwrgs.get('zero_center', True)
            if zero_center:
                warnings.warn(
                    "Both scaling and zero_center=True specified. Scaling already zero-centers data. "
                    "Consider setting zero_center=False to avoid redundant centering."
                )
            sc.pp.scale(adata)

        
        sc.tl.pca(data = adata, n_comps = n_components, **pcakwrgs)

        projection_check = np.allclose(project_to_pca(adata.X, adata), adata.obsm['X_pca'], atol = pc_projection_tol)
#         proj = project_to_pca(adata.X, adata)
#         projection_check = np.all(np.abs((proj - adata.obsm['X_pca']) / adata.obsm['X_pca']) <= pc_projection_rtol)
        if not projection_check:
            warnings.warn('Cannot reproduce scanpy pca projection')
#             raise ValueError('Cannot reproduce scanpy pca projection')
#         del proj

        if pc_rank == 'automate':
            pc_rank = _compute_elbow(adata = adata)
        else:
            pc_rank = n_components
        adata.uns["pca"]['pca_rank'] = pc_rank

    # if not np.allclose(adata.obsm['X_pca'], adata.uns['pca']['pca_mod'].transform(adata.X)): 
    #     raise ValueError('Unexpected disagreement when running PCA.transform')
    
    sc.pp.neighbors(adata = adata, 
                    n_pcs=adata.uns["pca"]['pca_rank'], 
                    n_neighbors = n_neighbors,
                    use_rep = 'X_pca')
    if run_umap:
        scanpy_umap(adata = adata, **umapkwrgs)

    # cluster
    if cluster_data:
        if not isinstance(resolution, list):
            sc.tl.leiden(adata = adata, resolution = resolution) # cluster
        else: # identify the leiden resolution that maximizes NMI with a pre-existing metadata label
            print('Iterate through leiden resolutions')
            best_res = None
            best_nmi = -np.inf
            for res in tqdm(resolution):
                sc.tl.leiden(adata = adata, resolution = res)
                nmi_val = normalized_mutual_info_score(adata.obs.leiden, adata.obs[nmi_label])
                if nmi_val > best_nmi:
                    best_nmi = nmi_val
                    best_res = res
            sc.tl.leiden(adata = adata, resolution = best_res)

        adata.obs.rename(columns = {'leiden': cluster_col_name}, inplace = True)


############################ PLS ############################ 
def prepare_input_matrix_plsda(adata,
                         control_confounders: List = [], #['cell_line, 'plate'],
                        enc_X = None):

    """Prepares input for PLSR, and fits one-hot encodings of
    confounding categorical covariates (cell line and plate)

    *Note, noticed that adding in cell cycle scores doesn't make a difference, but including
    the plate does make a substantial difference in assessment metrics.
    """

    if scipy.sparse.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = adata.X

    if len(control_confounders) != 0:
        covariate_arrays = []
        for cov in control_confounders:
            if not isinstance(adata.obs[cov].dtype, pd.CategoricalDtype):
                raise ValueError('All covariates being controlled for need to be categorical in this version')
            covariate_arrays.append(adata.obs[cov].astype(str).values)

        covariate_matrix = np.stack(covariate_arrays, axis=1)

        if enc_X is None:
            enc_X = OneHotEncoder(sparse_output=False, drop='first') # drop to avoid collinearity
            enc_X.fit(covariate_matrix)

        covariates = enc_X.transform(covariate_matrix)
        X = np.concatenate([X, covariates], axis=1)

#         cell_cycle_scores = np.concatenate([
#             adata.obs['S_score'].values.reshape(-1, 1),
#             adata.obs['G2M_score'].values.reshape(-1, 1)
#         ], axis=1)

#         scaler = StandardScaler()
#         cell_cycle_scaled = scaler.fit_transform(cell_cycle_scores)

#         X = np.concatenate([X, covariates, cell_cycle_scaled], axis=1)
    return X, enc_X

def calculate_pls_explained_variance_ratio(
    pls_model, X, y
) -> tuple[np.ndarray, np.ndarray]:
    """
    Copied straight from: https://github.com/scikit-learn/scikit-learn/issues/32675

    Calculate explained variance ratios using sequential deflation.
    Useful for per-component assessment. 

    This implements the variance decomposition for PLS regression following
    the deflation methodology described in Wegelin (2000).

    This method calculates how much variance each component explains by
    sequentially deflating the X and Y matrices. This is the standard
    approach in PLS and provides accurate component-wise variance.

    Parameters
    ----------
    pls_model : sklearn.cross_decomposition.PLSRegression
        A fitted PLSRegression model.
    X : array-like of shape (n_samples, n_features)
        Training vectors. Accepts numpy arrays, pandas DataFrames.
    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target vectors. Accepts 1D (univariate) or 2D (multivariate) targets.

    Returns
    -------
    tuple[ndarray, ndarray]
        - X variance ratios of shape (n_components,)
        - Y variance ratios of shape (n_components,)
    """
    # Convert to arrays and ensure y is 2D (handles pandas DataFrame/Series)
    X = np.asarray(X, dtype=float)
    y_array = np.asarray(y, dtype=float)
    y = np.atleast_2d(y_array).T if y_array.ndim == 1 else y_array

    # Center X and Y (PLS already centers data, but we need the original
    # centered versions)
    X_centered = X - X.mean(axis=0)
    y_centered = y - y.mean(axis=0)

    # Check for scaling
    if pls_model.scale:
        X_std = X.std(axis=0, ddof=1)
        X_std[X_std == 0.0] = 1.0
        X_centered /= X_std
        y_std = y.std(axis=0, ddof=1)
        y_std[y_std == 0.0] = 1.0
        y_centered /= y_std

    # Total variance in centered data
    X_total_var = np.var(X_centered, axis=0, ddof=1).sum()
    y_total_var = np.var(y_centered, axis=0, ddof=1).sum()
    has_x_variance = not np.isclose(X_total_var, 0.0)
    has_y_variance = not np.isclose(y_total_var, 0.0)

    # Initialize matrices for deflation
    X_current = X_centered.copy()
    y_current = y_centered.copy()

    X_var_ratios = []
    y_var_ratios = []

    # For each component, calculate variance explained then deflate
    for a in range(pls_model.n_components):
        # Get scores and loadings for component a (using slicing to keep 2D)
        t_a = pls_model.x_scores_[:, a : a + 1]  # (n_samples, 1)
        p_a = pls_model.x_loadings_[:, a : a + 1]  # (n_features_X, 1)
        q_a = pls_model.y_loadings_[:, a : a + 1]  # (n_features_y, 1)

        # Reconstruct X and y using current component
        X_hat = t_a @ p_a.T
        y_hat = t_a @ q_a.T

        # Variance of current residual before deflation
        X_var_before = np.var(X_current, axis=0, ddof=1.0).sum()
        y_var_before = np.var(y_current, axis=0, ddof=1.0).sum()

        # Deflate X and y
        X_current -= X_hat
        y_current -= y_hat

        # Variance of residual after deflation
        X_var_after = np.var(X_current, axis=0, ddof=1.0).sum()
        y_var_after = np.var(y_current, axis=0, ddof=1.0).sum()

        # Store variance explained as ratio of total variance
        if has_x_variance:
            X_var_ratios.append((X_var_before - X_var_after) / X_total_var)
        else:
            X_var_ratios.append(0.0)
        if has_y_variance:
            y_var_ratios.append((y_var_before - y_var_after) / y_total_var)
        else:
            y_var_ratios.append(0.0)

    return np.array(X_var_ratios), np.array(y_var_ratios)

def pls_deflation_r2(pls_model, X, y):
    """
    Assess an already fitted ``PLSRegression`` model.

    This function computes the cumulative explained variance in the X and Y
    matrices using the deflation-based variance decomposition implemented in
    ``calculate_pls_explained_variance_ratio``.
    
    Results are numerically identical to `calculate_pls_reconstruction_r2`. Use that function instead, this is slower

    Parameters
    ----------
    pls_model : sklearn.cross_decomposition.PLSRegression
        A fitted PLSRegression model.

    X : array-like of shape (n_samples, n_features)
        The predictor matrix used to fit the model.

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        The response matrix used to fit the model.

    Returns
    -------
    R2X : float
        Cumulative explained variance in the X matrix (should match reconstruction-based R2X coefficient of determination as in `calculate_pls_reconstruction_r2`)

    R2Y : float
        Cumulative explained variance in the Y matrix (should match reconstruction-based R2Y coefficient of determination as in `calculate_pls_reconstruction_r2`).
    """

    explained_x, explained_y = calculate_pls_explained_variance_ratio(pls_model = pls_model, X = X, y = y)
    R2X = explained_x.sum()
    R2Y = explained_y.sum()
    # sanity check
#     assert np.allclose((R2X, R2Y), calculate_pls_reconstruction_r2(pls_model, X, y)), 'Deflation- and reconstruction-based R2 does not match'

    
    return R2X, R2Y


def calculate_pls_reconstruction_r2(pls_model: PLSRegression, X: np.ndarray, y: np.ndarray):
    """
    Reconstruction-based R2X and R2Y for a fitted PLSRegression model.

    Uses the same centering/scaling as the model:
        R2X = 1 - ||Xc - X_hat||² / ||Xc||²
        R2Y = 1 - ||yc - y_hat||² / ||yc||²

    Parameters
    ----------
    pls_model : sklearn.cross_decomposition.PLSRegression
        A fitted PLSRegression model.

    X : array-like of shape (n_samples, n_features)
        The predictor matrix used in model fitting.

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        The response matrix used in model fitting.

    Returns
    -------
    R2X : float
        Reconstruction-based coefficient of determination for X.

    R2Y : float
        Reconstruction-based coefficient of determination for y.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]

    # --- center/scale exactly as the model did ---
    Xc = X - pls_model._x_mean
    if getattr(pls_model, "scale", False):
        Xc = Xc / pls_model._x_std

    yc = y - pls_model._y_mean
    y_std = getattr(pls_model, "_y_std", None)
    if y_std is not None:
        yc = yc / y_std

    # --- reconstruct in centered/scaled space ---
    T = pls_model.x_scores_
    P = pls_model.x_loadings_
    C = pls_model.y_loadings_

    X_hat = T @ P.T
    y_hat = T @ C.T

    # --- reconstruction-based R² ---
    ssx_tot = np.sum(Xc ** 2)
    ssy_tot = np.sum(yc ** 2)

    ssx_res = np.sum((Xc - X_hat) ** 2)
    ssy_res = np.sum((yc - y_hat) ** 2)

    R2X = 1.0 - ssx_res / ssx_tot
    R2Y = 1.0 - ssy_res / ssy_tot

    return R2X, R2Y


def calculate_pls_q2y(pls_model, X, y, n_folds=5, seed=888):
    """
    Compute Q²Y for a fitted PLSRegression model using K-fold CV
    following the PRESS definition:

        Q²Y = 1 - PRESS / SSY

    Parameters
    ----------
    pls_model : sklearn.cross_decomposition.PLSRegression
        A fitted PLSRegression model. Only its hyperparameters are used;
        the model is refit inside each fold.

    X : array-like of shape (n_samples, n_features)
        Predictor matrix.

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Response matrix. (OK for one-hot encoded PLS-DA.)

    n_folds : int, default=5
        Number of CV folds.

    seed : int, default=888
        Random seed for reproducibility.

    Returns
    -------
    q2y : float
        The Q²Y metric.
    """
    rng = check_random_state(seed)

    # Ensure y is 2D
    if y.ndim == 1:
        y = y[:, None]

    # --- Total Sum of Squares in Y ---
    y_mean = y.mean(axis=0)
    SSY = np.sum((y - y_mean)**2)

    # --- PRESS accumulator ---
    PRESS = 0.0

    # --- K-Fold CV ---
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Clone and fit model
        pls_cv = clone(pls_model)
        pls_cv.fit(X_train, y_train)

        # Predict
        y_pred = pls_cv.predict(X_test)

        # Add the squared prediction error for this fold
        PRESS += np.sum((y_test - y_pred)**2)

    # --- Q²Y formula ---
    Q2Y = 1.0 - PRESS / SSY
    return Q2Y


def calculate_pls_accuracy(pls_model, X, y, n_folds=5, seed=888):
    """
    Compute model accuracy for a fitted PLSRegression model using K-fold CV, similar to Q2Y.

    Parameters
    ----------
    pls_model : sklearn.cross_decomposition.PLSRegression
        A fitted PLSRegression model. Only its hyperparameters are used;
        the model is refit inside each fold.

    X : array-like of shape (n_samples, n_features)
        Predictor matrix.

    y : array-like of shape (n_samples, n_classes)
        One-hot encoded response matrix for PLS-DA.

    n_folds : int, default=5
        Number of CV folds.

    seed : int, default=888
        Random seed for reproducibility.

    Returns
    -------
    accuracy : float
        Mean cross-validated classification accuracy.
    """
    rng = check_random_state(seed)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    accuracy = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Clone and fit model
        pls_cv = clone(pls_model)
        pls_cv.fit(X_train, y_train)

        # Predict
        y_pred = pls_cv.predict(X_test)
        
        accuracy_cv = accuracy_score(np.argmax(y_test, axis = 1), np.argmax(y_pred, axis = 1))
        accuracy.append(accuracy_cv)

    return np.mean(accuracy)

def _single_pls_assess_perm(ir, pls_model, X, y, seed, n_folds, 
                 get_q2_pval, get_r2_pval, get_accuracy_pval):

    rng = check_random_state(seed + ir + 1)
    perm_idx = rng.permutation(len(y))
    y_perm = y[perm_idx]

    q2 = r2 = acc = None

    if get_q2_pval:
        pls_q = clone(pls_model)
        q2 = calculate_pls_q2y(pls_q, X, y_perm, n_folds=n_folds, seed=seed + ir + 1)

    if get_r2_pval:
        pls_r = clone(pls_model)
        pls_r.fit(X, y_perm)
        r2 = calculate_pls_reconstruction_r2(pls_r, X, y_perm)[1]

    if get_accuracy_pval:
        pls_a = clone(pls_model)
        acc = calculate_pls_accuracy(pls_a, X, y_perm, n_folds=n_folds, seed=seed + ir + 1)

    return q2, r2, acc

def assess_pls_model(pls_model, X, y, 
                     n_perm: int = 100,
                     get_q2_pval: bool = True, 
                     get_r2_pval: bool = False, 
                     get_accuracy_pval: bool = False,
                     n_folds=5, 
                     n_cores: int = None,
                     seed=888):
    """
    Assess a fitted PLSRegression model and optionally compute permutation
    p-values for R²Y and Q²Y.

    This function computes:
        - R²X, R²Y (reconstruction-based coefficients of determination)
        - Q²Y (cross-validated predictive ability)
        - p-values for Q²Y and R²Y using a Y-permutation test

    Parameters
    ----------
    pls_model : sklearn.cross_decomposition.PLSRegression
        A fitted PLSRegression model. Hyperparameters are reused for CV and
        permutation fits.

    X : array-like of shape (n_samples, n_features)
        Predictor matrix used for fitting or consistent with the fitted model.

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Response matrix. For PLS-DA, this should be a one-hot encoded matrix
        of class labels.

    n_perm : int, optional
        Number of permutations to perform for estimating permutation-based
        p-values. If None, permutation testing is skipped.

    n_folds : int, default=5
        Number of cross-validation folds used to compute Q²Y.

    seed : int, default=888
        Random seed for reproducibility of CV splits and permutations.

    Returns
    -------
    R2X : float
        Reconstruction-based coefficient of determination for X.

    R2Y : float
        Reconstruction-based coefficient of determination for y.

    Q2Y : float
        Cross-validated predictive ability of the model based on PRESS.
    
    accuracy : float
        Mean cross-validated classification accuracy.

    p_Q2Y : float or None
        Permutation p-value for Q²Y. None if `get_q2_pval` is False.

    p_R2Y : float or None
        Permutation p-value for R²Y. None if `get_r2_pval` is False.
    
    p_accuracy : float or None
        Permutation p-value for accuracy. None if `get_accuracy_pval` is False.
    """

    R2X, R2Y = calculate_pls_reconstruction_r2(pls_model, X, y)
    Q2Y = calculate_pls_q2y(pls_model, X, y, n_folds=n_folds, seed=seed)
    accuracy = calculate_pls_accuracy(pls_model, X, y, n_folds=n_folds, seed=seed)
    
    
    p_Q2Y = None
    p_R2Y = None
    p_accuracy = None
    
    if n_perm is None:
        permute = False
    else:
        permute = get_q2_pval or get_r2_pval or get_accuracy_pval
        
    
    if permute:
        if n_cores is None or n_cores <= 1:
            rng = check_random_state(seed)
            q2_perm = np.zeros(n_perm, float)
            r2_perm = np.zeros(n_perm, float)
            accuracy_perm = np.zeros(n_perm, float)
            for ir in trange(n_perm):
                # Permute Y *rows*  (ropls permutes class labels)
                perm_idx = rng.permutation(len(y))
                y_perm = y[perm_idx]

                if get_q2_pval:
                    pls_perm_q2 = clone(pls_model)
                    q2_perm[ir] = calculate_pls_q2y(pls_perm_q2, X, y_perm, n_folds=n_folds, seed=seed + ir + 1)
                
                if get_r2_pval:
                    pls_perm_r2 = clone(pls_model)
                    pls_perm_r2.fit(X, y_perm)
                    r2_perm[ir] = calculate_pls_reconstruction_r2(pls_perm_r2, X, y_perm)[1]
                    
                if get_accuracy_pval:
                    pls_perm_accuracy = clone(pls_model)
                    accuracy_perm[ir] = calculate_pls_accuracy(pls_perm_accuracy, X, y_perm, n_folds=n_folds, seed=seed + ir + 1)
        else:
            with tqdm_joblib(tqdm(total=n_perm, desc="PLS Assessment Permutations")):
                results = Parallel(n_jobs=n_cores, backend="loky")(
                    delayed(_single_pls_assess_perm)(
                        ir, pls_model, X, y, seed, n_folds,
                        get_q2_pval, get_r2_pval, get_accuracy_pval
                    )
                    for ir in range(n_perm)
                )

            # Unpack
            q2_perm   = np.array([r[0] for r in results])
            r2_perm   = np.array([r[1] for r in results])
            accuracy_perm  = np.array([r[2] for r in results])
        
        if get_q2_pval:
            p_Q2Y = (np.sum(q2_perm >= Q2Y) + 1) / (n_perm + 1)
        if get_r2_pval:
            p_R2Y = (np.sum(r2_perm >= R2Y) + 1) / (n_perm + 1)
        if get_accuracy_pval: 
            p_accuracy = (np.sum(accuracy_perm >= accuracy) + 1) / (n_perm + 1)

    return R2X, R2Y, Q2Y, accuracy, p_R2Y, p_Q2Y, p_accuracy


def select_pls_components(
    pls_model, X, y,
    max_components: int = 25,
    metric: Literal['Q2Y', 'R2Y', 'accuracy'] = 'accuracy', 
    method: Literal['maximize', 'elbow'] = 'elbow', 
    n_folds=5,
    verbose: bool = False,
    seed=888,
):
    """Determine the number of components for the PLS model. 

    Parameters
    ----------
    pls_model : 
        `PLSRegression` instance with initialized hyperparameters
    X : _type_
        array-like of shape (n_samples, n_features)
        Predictor matrix used for fitting or consistent with the fitted model.
    y : 
        array-like of shape (n_samples,) or (n_samples, n_targets)
        Response matrix. For PLS-DA, this should be a one-hot encoded matrix
        of class labels.
    max_components : int, optional
        maximum number of components to test, by default 25
    metric : Literal['Q2Y', 'R2Y', 'accuracy'], optional
        what metric to use for selection, by default 'accuracy'
        see `assess_pls_model` for details
    method : Literal['maximize', 'elbow'], optional
        whether to find the elbow in components or component that gets the optimal metric value, by default 'elbow'
    n_folds : int, optional
        number of folds to use if using Q2Y or accuracy, by default 5
    seed : int, optional
        Random seed for reproducibility of CV splits and permutations, by default 888
        
    Returns
    -------
    rank: int
        the non-zero indexed optimal number of PLS components
    metric_per_rank: list
        the metric value at each component (e.g., at index 4, the metric value of the PLS model with 5 components)
    """
    range_ = trange if verbose else range
    if metric == 'R2Y':
        pls_model_ranking = clone(pls_model)
        pls_model_ranking.n_components = max_components
        pls_model_ranking.fit(X, y)
        explained_var_x, explained_var_y = calculate_pls_explained_variance_ratio(pls_model_ranking, X, y)
        metric_per_rank = list(np.cumsum(explained_var_y))
    else: 
        if verbose:
            print('Iterate through components for PLS selection of number of components')   
        metric_per_rank = [np.nan]*max_components
        for r in range_(1, max_components+1):
            pls_model_ranking = clone(pls_model)
            pls_model_ranking.n_components = r

            if metric == 'Q2Y':
                metric_per_rank[r-1] = calculate_pls_q2y(pls_model_ranking, X, y, n_folds=n_folds, seed=seed)
            elif metric == 'accuracy':
                metric_per_rank[r-1] = calculate_pls_accuracy(pls_model_ranking, X, y, n_folds=n_folds, seed=seed)

    # get the non-zero_indexed rank 
    if method == 'elbow':
        kneedle = KneeLocator(x = range(1, max_components + 1), y = metric_per_rank, curve='concave', direction='increasing')
        rank = kneedle.elbow
    elif method == 'maximize':
        rank = np.argmax(metric_per_rank) + 1  
    return rank, metric_per_rank
    

    
default_pls_component_selection_kwargs = {
    'max_components': 25, 
    'metric': 'accuracy', 
    'method': 'elbow', 
    'n_folds': 5, 
    'seed': 888
}

default_pls_assessment_kwargs = {
    'n_perm': 100, 
    'get_q2_pval': True, 
    'get_r2_pval': True, 
    'get_accuracy_pval': True,
    'n_folds': 5, 
    'seed': 888
}

def pls_da(
    adata,
    n_components: int = None,
    control_confounders: List | None = None, #'plate',
    assess: bool = True,
    enc_X = None,
    enc_Y = None,
    separate_by: Literal['perturbation', 'category'] = 'perturbation',
    pert_col: str = 'drug',
    cat_col: str = 'cell_line', 
    pls_kwargs = None, 
    component_selection_kwargs = None,
    assessment_kwargs = None, 
    n_cores: int = 1,
    verbose: bool = False
          ):
    """Creates a PLS-DA model (e.g., drug ~ TF activity) given an input AnnData object.

    Parameters
    ----------
    adata : _type_
        anndata object
    n_components : int, optional
        Number of PLS components to use, by default None
        If None, selected the number of components using the function `select_pls_components`
    control_confounders : List, optional
        controls for various confounders in the X-block of PLS-DA, by default doesn't control for confounders.
        Currently deprecated (prevents good projection for visualization). 
    assess : bool, optional
        gets assessment metrics for PLS fit, by default True
    enc_X:
        fit model for X covariate encoding. If not provided (when fitting on actual data) will fit an encoding.
        If provided (when projecting predicted data), will use the fit encoding to transform the input. 
        Currently deprecated (prevents good projection for visualization). 
    enc_Y:
        fit model for y encoding. If not provided (when fitting on actual data) will fit an encoding.
        If provided (when projecting predicted data), will use the fit encoding to transform the input. 
    separate_by : Literal[perturbation, category], optional
        whether to fit the PLS model to separate by the `pert_col` ('perturbation') or `cat_col` ('categorical), by default 'perturbation'
    pert_col: str, optional
        specifies the label of the perturbation column in the metadata
    cat_col: str, optional
        specifies the label of the categorical column in the metadata
    pls_kwargs: 
        additional key word arguments to pass to sklearn's `PLSRegression`
    component_selection_kwargs:
        additional key word arguments to pass to `select_pls_components`
    assessment_kwargs:
        additioanl key word arguments to pass to `assessment_kwargs`
        
    Returns
    -------
    models: 
        A dictionary of the various fit models
        - enc_X: depreacted
        - enc_Y: the encoder for the y response
        - pls_model: the fit pls_model with the following additional attributes
            - metric_per_component: the assessment per each additional component if `n_components` is None
            - explained_x_variance_ratio_: X explained variance per component, if `assess` is True
            - explained_y_variance_ratio, y explained variance per component, if `assess` is True
            - assessment_metrics: standard pls model assessment metrics and associated p-values, if `assess` is True
    X_pls:
        The X scores from the model fit on the input X values
    """

    if pls_kwargs is None:
        pls_kwargs = {}
    if component_selection_kwargs is None:
        component_selection_kwargs = {}
    if assessment_kwargs is None:
        assessment_kwargs = {}

    if separate_by == 'perturbation':
        y = adata.obs[pert_col].astype(str).values.reshape(-1,1)
    elif separate_by == 'category':
        y = adata.obs[cat_col].astype(str).values.reshape(-1,1)

    if enc_Y is None:
        enc_Y = OneHotEncoder(sparse_output=False, drop=None)
        enc_Y.fit(y)
    y_encoded = enc_Y.transform(y)

    X, enc_X = prepare_input_matrix_plsda(
        adata = adata,
        control_confounders = [] if control_confounders is None else control_confounders,
        enc_X = enc_X
    )

    metric_per_component = None
    if n_components is None:
        component_selection_kwargs = {**default_pls_component_selection_kwargs, **component_selection_kwargs}
        n_components, metric_per_component = select_pls_components(
            pls_model = PLSRegression(**pls_kwargs), 
            X = X, y = y_encoded,
            verbose = verbose,
            **component_selection_kwargs
        )
        
    pls_model = PLSRegression(n_components=n_components, **pls_kwargs)
    pls_model.fit(X, y_encoded)
    pls_model.metric_per_component = metric_per_component

    if assess:
        if verbose:
            print('Begin assessment of final model fit')
        # get explained variance from model
        explained_x, explained_y = calculate_pls_explained_variance_ratio(pls_model = pls_model, X = X, y = y_encoded)
        pls_model.explained_x_variance_ratio_ = explained_x
        pls_model.explained_y_variance_ratio_ = explained_y

        assessment_kwargs = {**default_pls_assessment_kwargs, **assessment_kwargs}
        R2X, R2Y, Q2Y, accuracy, p_R2Y, p_Q2Y, p_accuracy = assess_pls_model(
            pls_model, X, y_encoded, 
            n_cores = n_cores,
            **assessment_kwargs
        )
        pls_model.assessment_metrics = {
            'R2X': {'value': R2X, 'pval': None}, 
            'R2Y': {'value': R2Y, 'pval': p_R2Y}, 
            'Q2Y': {'value': Q2Y, 'pval': p_Q2Y},  
            'accuracy': {'value': accuracy, 'pval': p_accuracy}
        }
        
    X_pls = pls_model.transform(X)
    models = {'pls_model': pls_model,
             'encoder_x': enc_X,
             'encoder_y': enc_Y}
    

    return models, X_pls

def manual_pls_projections(pls_model, X):
    """Manually projects the X block into the latent space. Should be analogous to 
    `pls_model.transform(X)`."""
    
    # manual transform
    W = pls_model.x_weights_
    P = pls_model.x_loadings_

    
    Xc = (X - pls_model._x_mean) 
    if pls_model.scale:
        Xc /= pls_model._x_std

    T_manual = Xc @ W @ np.linalg.inv(P.T @ W)
    
    return T_manual


def compute_vip(pls_model):
    """
    Compute Variable Importance in Projection (VIP) scores for a fitted sklearn PLSRegression model.
    Works for both single- and multi-response Y.

    Parameters
    ----------
    pls_model : sklearn.cross_decomposition.PLSRegression
        A fitted PLSRegression object (after .fit()).

    Returns
    -------
    vip_scores : np.ndarray of shape (n_features,)
        VIP score for each feature in X.
    """
    T = pls_model.x_scores_      # (n_samples, n_components)
    W = pls_model.x_weights_     # (n_features, n_components)
    Q = pls_model.y_loadings_    # (n_targets, n_components)

    # Compute sum of squares of Y explained by each component (across all targets)
    ssy = np.sum(np.sum((T ** 2), axis=0) * np.sum((Q ** 2), axis=0))  # scalar total
    ssy_per_comp = np.sum((T ** 2), axis=0) * np.sum((Q ** 2), axis=0)  # per-component
    
    p = W.shape[0]  # number of predictors
    
    # Compute VIP scores
    vip = np.sqrt(p * np.sum(ssy_per_comp * (W ** 2) / np.sum(W ** 2, axis=0), axis=1) / ssy)
    
    return vip



def assess_pls_separation(tf_adata, pls_model, enc_Y, 
                          pert_col: str, 
                          ctrl_pert: str,
                         get_pert_separation_stats = True
                         ):
    """
    Assess how well a fitted PLS model captures separation of perturbation.
    Quantify which features contribute most to that separation.

    This function evaluates both component- and feature- level metrics that together
    describe the strength and distribution of separation encoded by the PLS embedding.

    Specifically, it:
      • Computes VIP scores and X-loadings per feature, quantifying global and component-wise importance.
      • Calculates per-component variance in Y explained (R²Y).
      • Optionally, computes full feature space Cliff’s delta and Mann–Whitney U p-values 
        between binary perturbation groups (perturbation vs control), allowing comparison of discriminant features 
        in full feature space vs PLS space.

    Parameters
    ----------
    tf_adata : anndata.AnnData
        Expression matrix (features × samples) used for the PLS fit. 
        Must contain a column in `obs` corresponding to the binary perturbation.
    pls_model : sklearn.cross_decomposition.PLSRegression
        A fitted PLSRegression model.
    enc_Y : sklearn.preprocessing.OneHotEncoder
        Encoder used to one-hot encode the perturbation variable prior to PLS fitting.
    pert_col : str
        Column name in `tf_adata.obs` indicating perturbation labels.
    ctrl_pert : str
        Label identifying the control condition (used for Cliff’s delta and MWU tests).
    get_pert_separation_stats : bool, default=True
        If True, compute per-feature Cliff’s delta and Mann–Whitney U statistics 
        for binary perturbation vs control separation.

    Returns
    -------
    pls_feature_importance : pandas.DataFrame
        Feature-level metrics including:
            - 'VIP' : Variable Importance in Projection (global feature importance)
            - 'PLS_i' : X-loadings per PLS component
            - (optional) 'Cliff’s Delta' and 'MWU p-val' : univariate separation metrics
        Indexed by `tf_adata.var_names`, sorted by decreasing VIP.
    pls_stats : pandas.DataFrame
        Component-level metrics including:
            - 'Y variance explained' (% of Y variance captured per component)
            - 'Spearman Correlation (VIP, X loadings)' : alignment between global and component-wise feature importance
            - 'Pearson Correlation (X scores, Y)' : strength of sample-level separation along each latent axis

    Notes
    -----
    • Component-level metrics: 
        • High R²Y and strong Pearson Correlation (X scores, Y) correlations in early components indicate that 
          observation separation by perturbation is captured by the first few components. 
        • High (VIP, X loadings) Spearman correlations tells us how much a component's high feature importance 
          is reflected across all components.
        • A strong association between these observation separation metrics (bullet point 1) and 
          (VIP, X loadings) Spearman correlations (bullet point 2) tells us that the features in those PLS 
          components are informative of separation. <-- this is unintuitive but should be somewhat correct.
          Expectation is that there should be strong associations, by definition of the PLS objective.
    • Feature-level metrics: 
        • Non-uniform VIP / loading distributions imply sparsity in discriminant features (only a few drive signal) 
          in PLS space, making projection into this PLS space a stringent test of model fidelity.
        • Non-uniform Cliff's delta distributions imply sparsity in discriminant features (only a few drive signal)
          in full feature space, making capturing separation a difficult task for the model.
        • High correlation between Cliff’s delta and PLS metrics confirms that PLS captures 
          the same discriminant structure as the full feature space.

    """    
    # get the feature contributions 
    vip_scores = compute_vip(pls_model)
    vip_scores = pd.DataFrame(data = {'VIP': vip_scores})
    x_loadings = pd.DataFrame(pls_model.x_loadings_, 
                columns = ['PLS_{}'.format(i +1) for i in range(pls_model.n_components)])
    pls_feature_importance = pd.concat([vip_scores, x_loadings], axis = 1)
    pls_feature_importance.set_index(tf_adata.var_names, inplace = True)

    pls_feature_importance.sort_values(by = ['VIP'], ascending = False, inplace = True)
    
    
    pls_stats = {
        'Y variance explained': [], 
        'Spearman Correlation (VIP, X loadings)': [], 
        'Pearson Correlation (X scores, Y)': [],
    }
    
    y = tf_adata.obs[pert_col].values.reshape(-1,1)
    y = enc_Y.transform(y)
    
    R2Y_per_comp = pls_model.explained_y_variance_ratio_ #pls_explained_variance_y(pls_model, y)
    

    
    for i in range(pls_model.n_components):
        var_explained = R2Y_per_comp[i] * 100
        sr = stats.spearmanr(pls_feature_importance["VIP"], pls_feature_importance["PLS_{}".format(i+1)]).statistic
        # only with the first dummy variable (totally fine if binary)
        pr = stats.pearsonr(pls_model.x_scores_[:, i], y[:, 0]).statistic
        
        
        pls_stats['Y variance explained'].append(var_explained)
        pls_stats['Spearman Correlation (VIP, X loadings)'].append(sr)
        pls_stats['Pearson Correlation (X scores, Y)'].append(pr)
        
    pls_stats = pd.DataFrame(pls_stats)
    pls_stats.index = ['PLS_{}'.format(i+1) for i in range(pls_model.n_components)]   
    
    
    
    if get_pert_separation_stats:
        assert tf_adata.obs[pert_col].nunique() == 2, 'Stats are only for binary separation'
        assert ctrl_pert in tf_adata.obs[pert_col].values, 'Stats are only for binary separation from control perturbation'


        mwus = []
        cds = []
        for feature in pls_feature_importance.index:
            stats_df = tf_adata[:, feature].to_df()
            stats_df[pert_col] = tf_adata.obs[pert_col]

            mask = (stats_df[pert_col] == ctrl_pert)

            ctrl_expr = stats_df[mask][feature].values
            pert_expr = stats_df[~mask][feature].values

            mwu_pval = stats.mannwhitneyu(pert_expr, ctrl_expr).pvalue
            cd = cliffs_delta(pert_expr, ctrl_expr)[0] # positive if pert > ctrl, negative othersie

            mwus.append(mwu_pval)
            cds.append(cd)
        
        pls_feature_importance['{} Separation MWU p-val'.format(pert_col)] = mwu_pval
        pls_feature_importance["{} Separation Cliff's Delta".format(pert_col)] = cds
    
    return pls_feature_importance, pls_stats


############################ PIPELINES ############################ 
def pls_da_pipeline(
    adata,
    pert_ids: list | str,
    cat_ids: list | str,
    n_components: int = None,
    control_confounders: list | None = None,
    assess_pls_fit: bool = True, 
    pert_col = 'drug',
    cat_col = 'cell_line',
    separate_by: Literal['perturbation', 'category'] = 'perturbation',
    pls_kwargs = None, 
    component_selection_kwargs = None,
    assessment_kwargs = None,
    covariate_associations: list | None = None,
    per_component_association: bool = False, 
    global_component_association: bool = True, 
    run_umap: bool = True,
    file_prefix: str | None = None,
    verbose: bool = False,
    n_cores: int = -1,
    seed: int = 888,
):
    """Full PLS DA pipeline to separate values in adata.X by a categorical covariate. Pipeline will:
        1) identify the # of components that optimizes PLS model fit and retain this model
        2) quantify the extent to which a given covariate is associated univariately with each PLS component
        3) run categorical UMAP on the PLS space

    Parameters
    ----------
    adata : _type_
        anndata object
    pert_ids : list | str
        values within `pert_col` to include in analysis
    cat_ids : list | str
        values within `cat_col` to include in analysis
    n_components : int, optional
        number of components to fit model on, by default None
        if None, will identify with an automated elbow selection up to 25 components (this can take a while to run)
    assess_pls_fit : bool, optional
        gets assessment metrics for PLS fit, by default True
    control_confounders : List, optional
        controls for various confounders in the X-block of PLS-DA, by default doesn't control for confounders.
        Currently deprecated (prevents good projection for visualization). 
    pert_col : str, optional
        perturbation column in `adata.obs`, by default 'drug'
    cat_col : str, optional
        categorical column in `adata.obs`, by default 'cell_line'
    separate_by : Literal[perturbation, category], optional
        whether to fit the PLS model to separate by the `pert_col` ('perturbation') or `cat_col` ('category'), by default 'perturbation'
    covariate_associations : list, optional
        calculates statistical associations between PLS space and a covariate in `adata.obs`
    per_component_association : bool, optional
        whether to calculate the `covariate_associations` univariately per PLS component, see `latent_association_per_component` for details, by default False
    global_component_association : bool, optional
        whether to calculate the `covariate_associations` globally across all PLS components, see `latent_association_global` for details, by default True 
    run_umap: bool, optional
        whether to run umap (True) or not (False), by default True
    file_prefix : str | None, optional
        saves assessment metrics to this file_prefix, by default None
    verbose : bool, optional
        whether to print information about progress, by default False
    n_cores : int, optional
        number of cores to parallelize latent space associations on
    seed : int, optional
        random state, by default 888
    """
    if type(cat_ids) == str:
        cat_ids = [cat_ids]
    if type(pert_ids) == str:
        pert_ids = [pert_ids]

    if separate_by == 'perturbation':
        separate_col = pert_col
    elif separate_by == 'category':
        separate_col = cat_col


    mask = (adata.obs[cat_col].isin(cat_ids)) & ((adata.obs[pert_col].isin(pert_ids)))
    adata_sub = adata[mask].copy()

    print("Fit PLS Model") if verbose else None
    models, X_pls = pls_da(
        adata = adata_sub,
        n_components = n_components,
        control_confounders = control_confounders,
        assess = assess_pls_fit,
        enc_X = None,
        enc_Y = None,
        separate_by = separate_by,
        pert_col = pert_col,
        cat_col = cat_col,
        pls_kwargs = pls_kwargs, 
        component_selection_kwargs = component_selection_kwargs,
        assessment_kwargs = assessment_kwargs, 
        n_cores = n_cores,
        verbose = verbose
    )
    adata_sub.obsm['X_pls'] = X_pls

    r2_df_per_component = None
    cv_df_global = None
    if covariate_associations is not None and len(covariate_associations) != 0:
        latent_association_functions = {}
        if per_component_association:
            latent_association_functions['per_component'] = latent_association_per_component
        if global_component_association:
            latent_association_functions['global'] = latent_association_global

        print("Calculate covariate - PLS associations") if verbose else None
        for calc_type, latent_association_function in latent_association_functions.items():
            
            la_df_linear = latent_association_function(
                adata = adata_sub,
                covariates = covariate_associations,
                model_type = 'linear',
                latent_label = 'pls',
                n_cores = n_cores,
                seed = seed
            )

            la_df_nl = latent_association_function(
                adata = adata_sub,
                covariates = covariate_associations,
                model_type = 'nonlinear',
                latent_label = 'pls',
                n_cores = n_cores,
                seed = seed
            )

            la_df = pd.concat([la_df_linear, la_df_nl])
            if file_prefix is not None:
                la_df.to_csv(file_prefix + '{}_pls_associations.csv'.format(calc_type))

            if calc_type == 'per_component':
                r2_df_per_component = la_df
            elif calc_type == 'global':
                cv_df_global = la_df

    X_umap = None
    if run_umap:
        print("Get UMAP") if verbose else None
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*n_jobs value 1 overridden to 1 by setting random_state.*", category=UserWarning)

            umap_model = umap.UMAP(
                n_neighbors=15,
                n_components=2,
                metric='euclidean',
                #target_metric='categorical',
                n_jobs = n_cores,
                random_state = seed)
        
            umap_model.fit(X_pls) #, adata_sub.obs[separate_col].cat.codes.values)
            X_umap = umap_model.transform(X_pls)
        models['umap_model'] = umap_model

        
    # store values in AnnData object
    adata_sub.uns['pls'] = {'pls_mod': models['pls_model'],
                        'encoder_x': models['encoder_x'], 
                        'encoder_y': models['encoder_y'],
                        }

    

    if run_umap:
        adata_sub.uns['umap_pls'] = {'umap_pls_mod': umap_model}
        adata_sub.obsm['X_umap_pls'] = X_umap

    return adata_sub, r2_df_per_component, cv_df_global


def pc_pipeline(
    adata,
    pert_ids: list | str,
    cat_ids: list | str,
    n_components: int = None,
    pert_col = 'drug',
    cat_col = 'cell_line',
    get_hvgs: bool = False, 
    run_umap: bool = True,
    covariate_associations: list = ['cell_line', 'drug', 'plate', 'phase', 'S_score', 'G2M_score', 'pcnt_mito'],
    per_component_association: bool = False, 
    global_component_association: bool = True, 
    file_prefix: str | None = None,
    verbose: bool = False,
    n_cores: int = -1,
    seed: int = 888,
    hvgkwrgs = {},
    embkwrgs = {}
):
    """Full PCA pipeline to quantify variance in adata.X. Pipeline will:
        1) identify the # of components that optimizes PC fit
        2) quantify the extent to which a given covariate is associated univariately with each PC component
        3) run  UMAP on the PC space

    Assumes log-normalized data in the`adata.X` and hvgs already calculated. PCA is run on HVGs

    Parameters
    ----------
    adata : _type_
        anndata object
    pert_ids : list | str
        values within `pert_col` to include in analysis
    cat_ids : list | str
        values within `cat_col` to include in analysis
    n_components : int, optional
        number of components to fit model on, by default None
        if None, will identify with an automated elbow selection up to 25 components (this can take a while to run)
    pert_col : str, optional
        perturbation column in `adata.obs`, by default 'drug'
    cat_col : str, optional
        categorical column in `adata.obs`, by default 'cell_line'
    get_hvgs: bool, optional
        whether to calculate HVGs on the anndata subset (True) or use those in the passed object (False)
        only relevant if `use_highly_variable` = True in embkwrgs (which is the default in PCA)
    run_umap: bool, optional
        whether to run umap (True) or not (False), by default True
    covariate_associations : list, optional
        calculates statistical associations between PC space and a covariate in `adata.obs`
    per_component_association : bool, optionalde
        whether to calculate the `covariate_associations` univariately per PCA component, see `latent_association_per_component` for details, by default Falsede
    global_component_association : bool, optional
        whether to calculate the `covariate_associations` globally across all PCA components, see `latent_association_global` for details, by default True 
    file_prefix : str | None, optional
        saves assessment metrics to this file_prefix, by default None
    verbose : bool, optional
        whether to print information about progress, by default False
    n_cores : int, optional
        number of cores to parallelize latent space associations on
    seed : int, optional
        random state, by default 888
    hvgkwrgs: 
        key word arguments to pass to sc.pp.highly_variable_genes
    embkwrgs: 
        key word arguments to pass to `embed_adata` 
    """
    if type(cat_ids) == str:
        cat_ids = [cat_ids]
    if type(pert_ids) == str:
        pert_ids = [pert_ids]

    mask = (adata.obs[cat_col].isin(cat_ids)) & ((adata.obs[pert_col].isin(pert_ids)))
    adata_sub = adata[mask].copy()
    
    # reaclculate HVGs on the adata_sub?
    use_highly_variable = True  # default scanpy behavior
    if 'pcakwrgs' in embkwrgs:
        use_highly_variable = embkwrgs['pcakwrgs'].get('use_highly_variable', True)
    if get_hvgs and use_highly_variable:
        sc.pp.highly_variable_genes(adata_sub, **hvgkwrgs)
    
    print("Run dimensionality reductions") if verbose else None
    embed_adata(
        adata = adata_sub,
        n_components = 50 if n_components is None else n_components,
        pc_rank = 'automate' if n_components is None else n_components,
        run_pca = True,
        run_umap = run_umap,
        **embkwrgs
    )

    r2_df_per_component = None
    cv_df_global = None
    if covariate_associations is not None and len(covariate_associations) != 0:
        latent_association_functions = {}
        if per_component_association:
            latent_association_functions['per_component'] = latent_association_per_component
        if global_component_association:
            latent_association_functions['global'] = latent_association_global

        print("Calculate covariate - PC associations") if verbose else None
        for calc_type, latent_association_function in latent_association_functions.items():
            
            la_df_linear = latent_association_function(
                adata = adata_sub,
                covariates = covariate_associations,
                model_type = 'linear',
                latent_label = 'pca',
                n_cores = n_cores,
                seed = seed
            )

            la_df_nl = latent_association_function(
                adata = adata_sub,
                covariates = covariate_associations,
                model_type = 'nonlinear',
                latent_label = 'pca',
                n_cores = n_cores,
                seed = seed
            )

            la_df = pd.concat([la_df_linear, la_df_nl])
            if file_prefix is not None:
                la_df.to_csv(file_prefix + '{}_pc_associations.csv'.format(calc_type))

            if calc_type == 'per_component':
                r2_df_per_component = la_df
            elif calc_type == 'global':
                cv_df_global = la_df

    return adata_sub, r2_df_per_component, cv_df_global


############################ VISUALIZATION AND QUANTIFICATION ############################
def ss_explained_var(Y, Y_pred):
    ss_res = np.sum((Y - Y_pred) ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    return 1 - ss_res / ss_tot

def calc_adj_r2(r2, n, p):
    """Calculated the adjusted R^2"""
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def calc_nmi(y,cov):
    y_disc = pd.qcut(y, q=min(20, len(np.unique(y))), duplicates='drop', labels=False)

    # Discretize covariate only if continuous
    if pd.api.types.is_numeric_dtype(cov):
        x_disc = pd.qcut(cov, q=min(20, len(np.unique(cov))), duplicates='drop', labels=False)
    else:
        x_disc = cov.astype('category').cat.codes

    nmi = normalized_mutual_info_score(x_disc, y_disc)
    
    return nmi

def latent_association_per_component(
    adata,
    covariates: List[str],
    model_type: Literal['linear', 'nonlinear'],
    n_cores: int,
    latent_label: str = 'pca',
    seed: int = 888):
    """Gets the linear association between a set of latent variables stored in the AnnData object
    and covariates in the `adata.obs`, returning the R^2. 
    
    LV_i ~ covariate_j. This answers how much each latent variable can be explained by a given covariate. 
    Gives an indication of, visually, the extent of separation we expect to see.

    Parameters
    ----------
    adata : _type_
        anndata object
    covariates : List[str]
        categorical covariates asssociated to test for
    model_type : Literal['linear', 'nonlinear']
        uses LinearRegression if linear and RandomForest if nonlinear
    latent_label : str, optional
        name of latent space stored in ``adata.obsm`` as 'X_<latent_label>' 
        
    """
    n_lvs = adata.obsm['X_' + latent_label].shape[1]
    if latent_label == 'pca' and 'pca_rank' in adata.uns['pca']:
        n_lvs = adata.uns['pca']['pca_rank']

    _label_name = latent_label.upper() if latent_label != 'pca' else 'PC'
    
    lvs = adata.obsm['X_' + latent_label][:, :n_lvs]
    
    if model_type == 'linear':
        model_general = LinearRegression()
    elif model_type == 'nonlinear':
        model_general = RandomForestRegressor(random_state=seed,
                n_jobs=n_cores,                 
                verbose=False )
    
    res = []
    for cov_ in covariates:
        print(cov_)
        cov = adata.obs[cov_]
        
        if pd.api.types.is_numeric_dtype(cov):
            X_cov = cov.values.reshape(-1, 1)
        else:
            enc = OneHotEncoder(drop="first", sparse_output=False)
            X_cov = enc.fit_transform(cov.astype(str).values.reshape(-1, 1))
        
        r2_scores = []
        for lv_idx in trange(lvs.shape[1]):
            y = lvs[:, lv_idx]
            model = model_general.fit(X_cov, y)
            y_pred = model.predict(X_cov)
            r2 = ss_explained_var(y, y_pred)
            adj_r2 = calc_adj_r2(r2, n=len(y), p=X_cov.shape[1])
            nmi = calc_nmi(y,cov)


#             model = model_general.fit(X_cov, y)            
#             r2 = model.score(X_cov, y)
   
            r2_scores.append({
                _label_name: lv_idx + 1,
                "R2": r2,
                "Adj_R2": adj_r2,
                'NMI': nmi
#                 "NMI": nmi
            })

        r2_df = pd.DataFrame(r2_scores)
        r2_df['covariate'] = cov_
        res.append(r2_df)
        
    r2_df = pd.concat(res, axis=0, ignore_index=True)
    r2_df = r2_df.pivot(index=_label_name, columns='covariate')[["R2", "Adj_R2", "NMI"]]
    r2_df.columns = ['_'.join(col).strip() for col in r2_df.columns.values]
    r2_df = r2_df.reset_index()

    r2_df['model_type'] = model_type
    
    return r2_df


## cv scorers for latent_association_global
def adjusted_r2_score_cv(y_true, y_pred, p):
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def make_adjusted_r2_scorer(p):
    return make_scorer(lambda yt, yp: adjusted_r2_score_cv(yt, yp, p))

def chance_adjusted_acc(y_true, y_pred):
    K = len(np.unique(y_true))
    chance = 1.0 / K
    acc = accuracy_score(y_true, y_pred)
    return (acc - chance) / (1 - chance)
chance_adj_scorer = make_scorer(chance_adjusted_acc)

def nmi_scorer_func(y_true, y_pred):
    return normalized_mutual_info_score(y_true, y_pred)
nmi_scorer = make_scorer(nmi_scorer_func)


def latent_association_global(
    adata,
    covariates: List[str],
    model_type: Literal['linear', 'nonlinear'],
    n_cores: int,
    latent_label: str = 'pca',
    seed: int = 888):
    """Determines how well the latent space informs covariates of interest. Separately regresses each covariate.
    
    Multivariately runs Cov_j ~ LVs. Contrast to `latent_association_per_component`, which runs per component
    individually, with the covariate as the independent variable. 
    
    Parameters
    ----------
    adata : _type_
        anndata object
    covariates : List[str]
        categorical covariates asssociated to test for
    model_type : Literal['linear', 'nonlinear']
        uses LinearRegression if linear and RandomForest if nonlinear
    latent_label : str, optional
        name of latent space stored in ``adata.obsm`` as 'X_<latent_label>' 
    """


    n_lvs = adata.obsm['X_' + latent_label].shape[1]
    if latent_label == 'pca' and 'pca_rank' in adata.uns['pca']:
        n_lvs = adata.uns['pca']['pca_rank']

    _label_name = latent_label.upper() if latent_label != 'pca' else 'PC'

    lvs = adata.obsm['X_' + latent_label][:, :n_lvs]
    p = lvs.shape[1]


    cv_res = None #{}
    for cov_ in covariates:
        print(cov_)
        cov = adata.obs[cov_]

        if pd.api.types.is_numeric_dtype(cov):
            y = cov.values
            cv = KFold(n_splits=5, shuffle=True, random_state=seed)
            scoring = {
                'r2': 'r2',
                'adj_r2': make_adjusted_r2_scorer(p),
            }
            
            if model_type == 'linear':
                model_general = LinearRegression(n_jobs = 1)
            elif model_type == 'nonlinear':
                model_general = RandomForestRegressor(random_state=seed,
                        n_jobs=n_cores,                 
                        verbose=False )
        else:
            y = cov.astype(str).values
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            scoring = {
                'accuracy': 'accuracy',
                'balanced_accuracy': make_scorer(balanced_accuracy_score),
                'chance_adjusted_accuracy': chance_adj_scorer,
                'NMI': nmi_scorer,
            }


            if model_type == 'linear':
                model_general = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', LogisticRegression(
                        solver='saga',
                        penalty='l2',
                        max_iter=1000,
                        n_jobs=1,
                        random_state=seed
                    ))
                ])
            elif model_type == 'nonlinear':
                model_general = RandomForestClassifier(random_state=seed,
                        n_jobs=n_cores,                 
                        verbose=False )
                

        cv_out = cross_validate(
            estimator=model_general,
            X=lvs,
            y=y,
            cv=cv,
            scoring=scoring,
            n_jobs=n_cores if model_type == 'linear' else 1,
            return_train_score=False
        )

        tmp = pd.DataFrame({
            "{}_{}".format(cov_, metric): cv_out["test_{}".format(metric)]
            for metric in scoring.keys()
        })

        tmp.insert(0, "fold", range(1, tmp.shape[0] + 1))
        tmp.insert(1, "model_type", model_type)

        # Merge into final DF
        if cv_res is None:
            cv_res = tmp
        else:
            cv_res = cv_res.merge(tmp, on=["fold", "model_type"], how="left")
    
    return cv_res

def visualize_latent_space(
    adata, 
    latent_label: Literal['pca', 'pls', 'umap', 'umap_pls'], 
    covariates: list, 
    plot_type: Literal['scatter', 'contour'] = 'scatter',
    panel_titles: list = None,
    components: list = [1,2], 
    n_frac: float = 1, #0.2, 
    frac_col: str | None = None, 
    fig_title: str | None = None,
    legend = False, 
    seed: int = 888, 
    file_name: str | None = None, 
    show_fig: bool = True,
    **kwargs

):
    """Visualize the latent space with specific covariates as the hue.

    Parameters
    ----------
    adata : _type_
        _description_
    latent_label : Literal['pca', 'pls', 'umap', 'umap_pls']
        which latent space to visualize
    covariates : list
        which covariates to visualize
    plot_type : Literal['scatter', 'contour']
        whether to visualize with a scatter or contour plot
    panel_titles : list, optional
        mapping of covariates to titles in the figure panels (corresponding index)
    components : list | dict, optional
        which components to display on the 2 axes, by default [1,2]
        if a dictionary, will map each covariate to that component
    n_frac : float, optional
        subset to this fraction of the data for quicker visualization, by default 1
    frac_col : str | None, optional
        if set, will evenly subset each value in this column, by default None
    fig_title : str, optional
        title of the figure, by default None
    legend : bool, optional
        whether each panel should receive a legend, by default False
    seed : int, optional
        random state, by default 888
    file_name : str, optional
        save to this file name, by default None
    show_fig : bool, optional
        whether to show the figure when calling the function (i.e., when running in Notebook cell)
    **kwargs : 
        additional key word arguments to pass to main seaborn plot
    """
        
    if panel_titles is None or len(panel_titles) == 0:
        panel_titles = covariates
    
    _label_name = latent_label.upper() if latent_label != 'pca' else 'PC'
    if type(components) == list:
        components = [_label_name + '{}'.format(component) for component in components]
        components = {covariate: components for covariate in covariates}
    else:
        for component, plot_components in components.items():
            components[component] = [_label_name + '{}'.format(plot_component) for plot_component in plot_components]

    
    viz_df = pd.DataFrame(adata.obsm['X_' + latent_label])
    viz_df.columns = [_label_name + '{}'.format(i+1) for i in range(viz_df.shape[1])]
    
    _covariates = covariates.copy()
    if frac_col is not None and frac_col not in _covariates:
        _covariates.append(frac_col)
        
    for covariate in _covariates:
        viz_df[covariate] = adata.obs[covariate].reset_index(drop=True)
    
    if frac_col is None:
        viz_df = viz_df.sample(frac=n_frac, replace=False, random_state=seed)
    else:
        sampled_indices = (
            viz_df.groupby(frac_col, observed=False)
                  .sample(frac=n_frac, replace=False, random_state=seed)
                  .index
        )

        # shuffle
        np.random.seed(seed)
        sampled_indices=np.random.permutation(sampled_indices)

        viz_df = viz_df.loc[sampled_indices, :].copy()
        
    max_cols = 3
    n_cols = min(max_cols, len(covariates))
    n_rows = math.ceil(len(covariates) / max_cols)

    fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(5.1*n_cols, 5.1*n_rows))
    ax = axes if isinstance(axes, np.ndarray) else np.array([axes])
    ax = ax.flatten()
    for (i, covariate) in enumerate(covariates):
        plot_component = components[covariate]

        if plot_type == 'scatter':
            scatter_defaults = {"s": 10}
            final_kwargs = {**scatter_defaults, **kwargs}
            sns.scatterplot(data = viz_df, x = plot_component[0], y = plot_component[1], hue = covariate,
                            ax = ax[i], **final_kwargs)
        elif plot_type == 'contour':
            contour_defaults = {"fill": False, "levels": 10}
            final_kwargs = {**contour_defaults, **kwargs}
            sns.kdeplot(data = viz_df, x = plot_component[0], y = plot_component[1], hue = covariate,
                        ax = ax[i], **final_kwargs)
        
        if not legend:
            ax[i].legend_.remove()
        ax[i].set_title(panel_titles[i])
        
    for j in range(len(covariates), len(ax)): ax[j].set_visible(False)

    if fig_title is not None:
        fig.suptitle(fig_title)
    fig.tight_layout()
    if file_name is not None:
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        
    if not show_fig:
        plt.close(fig)
        
        
        
def visualize_latent_association_per_component(
    metric_df,
    fig_title: str | None = None, 
    metric_label = 'Fraction of Variance Explained',
    file_name: str | None = None,
):
    """Visualize the variance explained in the latent space, as calculated by`latent_association`."""
    latent_label = metric_df.columns[0]
    fig, ax = plt.subplots(ncols = 2, figsize = (10, 5))
    for i, model_type in enumerate(['linear', 'nonlinear']):
        viz_df = metric_df[metric_df.model_type == model_type]
        viz_df = viz_df.drop(columns = [latent_label, 'model_type']).copy()

        ranked_covars = viz_df.median(axis = 0).sort_values(ascending = False).index.tolist()

        viz_df = pd.melt(viz_df, var_name='covariate', value_name = 'var_explained')
        viz_df.covariate = pd.Categorical(viz_df.covariate, 
                                     categories = ranked_covars, 
                                     ordered=True)

        sns.boxplot(data = viz_df, x = 'covariate', y = 'var_explained', ax = ax[i])

        ax[i].set_title(model_type.capitalize())
        ax[i].set_ylabel(metric_label)
        ax[i].set_xlabel('Covariates')

        for label in ax[i].get_xticklabels():
            label.set_rotation(45)

    if fig_title is not None:
        fig.suptitle(fig_title)
    fig.tight_layout()
    if file_name is not None:
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        
# def visualize_latent_association_per_component(
#     r2_df,
#     fig_title: str | None = None, 
#     file_name: str | None = None,
#     use_adj_r2: bool = False,
# ):
#     """Visualize the variance explained in the latent space, as calculated by `latent_association`."""
#     latent_label = r2_df.columns[0]
#     metric = "Adj_R2" if use_adj_r2 else "R2"

#     fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
#     for i, model_type in enumerate(['linear', 'nonlinear']):
#         viz_df = r2_df[r2_df.model_type == model_type]
#         viz_df = viz_df.filter(like=metric).copy()
#         viz_df.insert(0, latent_label, r2_df[latent_label])
#         viz_df['model_type'] = model_type

#         viz_df = viz_df.drop(columns=[latent_label, 'model_type']).copy()
#         ranked_covars = viz_df.median(axis=0).sort_values(ascending=False).index.tolist()

#         viz_df = pd.melt(viz_df, var_name='covariate', value_name='var_explained')
#         viz_df.covariate = pd.Categorical(viz_df.covariate, categories=ranked_covars, ordered=True)

#         sns.boxplot(data=viz_df, x='covariate', y='var_explained', ax=ax[i])
#         ax[i].set_title(model_type.capitalize())
#         ax[i].set_ylabel('Fraction of Variance Explained')
#         ax[i].set_xlabel('Covariates')

#         for label in ax[i].get_xticklabels():
#             label.set_rotation(45)

#     if fig_title is not None:
#         fig.suptitle(fig_title)
#     fig.tight_layout()
#     if file_name is not None:
#         plt.savefig(file_name, dpi=300, bbox_inches="tight")
        
        

def get_top_components(r2_df, top_components_cov, verbose = True):
    """Identify top 2 components explaining a covariate."""
    latent_label = r2_df.columns[0]
    top_components = r2_df[[latent_label, top_components_cov, 'model_type']].copy()
    top_components = top_components[top_components.model_type == 'linear']
    top_components = top_components.sort_values(by = top_components_cov, ascending=False).iloc[:2, :].reset_index(drop = True)

    if verbose:
        print('The two {} components that best univariately separate by {} are components {} and {} explaining {:.2f}% and {:.2f}% of variance, respectively'.format(
            latent_label,
            top_components_cov,
            top_components[latent_label][0],
            top_components[latent_label][1], 
            top_components[top_components_cov][0]*100, 
            top_components[top_components_cov][1]*100,
        ))

    top_components = ['{}'.format(i) for i in top_components[latent_label]]
    return top_components



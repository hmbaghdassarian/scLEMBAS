"""Set of functions quantifying how latent space captures expression shifts of cells."""

from typing import Literal, List
import math

from tqdm import trange

import numpy as np
import pandas as pd
import scanpy as sc

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline

import scipy

# from . import preprocess as pp
from scLEMBAS import preprocess as pp

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


def pls_da(adata,
           n_components: int,
          control_confounders: List = [], #'plate',
          assess: bool = True,
           return_components: bool = True,
          seed: int = 888,
          enc_X = None,
          enc_Y = None,
           separate_by: Literal['perturbation', 'category'] = 'perturbation',
           pert_col: str = 'drug',
           cat_col: str = 'cell_line'
          ):
    """Creates a PLS-DA model (drug ~ TF activity) with confounding covariates

    Parameters
    ----------
    adata : _type_
        anndata object
    n_components : int
        Number of PLS components to use
    control_confounders : List, optional
        controls for various confounders in the X-block of PLS-DA, by default doesn't control for confounders
    assess : bool, optional
        gets assessment metrics for PLS fit, by default True
    return_components : bool, optional
        returns the X PLS components, by default True
    seed : int, optional
        random state, by default 888
    enc_X:
        fit model for encoding. If not provided (when fitting on actual data) will fit an encoding.
        If provided (when projecting predicted data), will use the fit encoding to transform the input
    """


    if separate_by == 'perturbation':
        y = adata.obs[pert_col].astype(str).values.reshape(-1,1)
    elif separate_by == 'category':
        y = adata.obs[cat_col].astype(str).values.reshape(-1,1)

    if enc_Y is None:
        enc_Y = OneHotEncoder(sparse_output=False, drop=None)
        enc_Y.fit(y)
    Y = enc_Y.transform(y)

    X, enc_X = prepare_input_matrix_plsda(
        adata = adata,
        control_confounders = control_confounders,
        enc_X = enc_X
    )

    pls_model = PLSRegression(n_components=n_components)
    pls_model.fit(X, Y)

    X_pls = None
    if assess or return_components:
        X_pls = pls_model.transform(X)

    assessment = None
    if assess:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        accuracy_score = cross_val_score(clf, X_pls, y.ravel(),
                                         cv=StratifiedKFold(5, random_state=seed, shuffle=True),
                                         scoring='accuracy').mean()

#         explained_var_y = np.var(pls_model.y_scores_, axis=0, ddof=0) / np.var(Y, axis=0, ddof=0).sum()
#

        Y_pred = pls_model.predict(X)
        explained_var_y = pp.ss_explained_var(Y, Y_pred) #1 - np.sum((Y - Y_pred) ** 2) / np.sum((Y - Y.mean(axis=0)) ** 2)


#         X_pred = pls_model.x_scores_ @ pls_model.x_loadings_.T
#         explained_var_x = pp.ss_explained_var(X, X_pred)
#         explained_var_x = np.var(pls_model.x_scores_, axis=0) / np.var(X, axis=0).sum()
#         cum_explained_x = np.cumsum(explained_var_x)[-1]

        assessment = {
            'n_components': n_components,
            'accuracy': accuracy_score,
            'explained_y': explained_var_y}

    models = {'pls_model': pls_model,
             'encoder_x': enc_X,
             'encoder_y': enc_Y}

    return models, assessment, X_pls

def pls_elbow(
    adata,
    pls_components_max: int = 25,
    elbow_metric: Literal['accuracy', 'explained_y'] = 'explained_y',
    separate_by: Literal['perturbation', 'categorical'] = 'perturbation',
    control_confounders: list = [],
    pert_col: str = 'drug',
    cat_col: str = 'cell_line',
    seed: int = 888,
):
    """Identifies  the number of PLS components to use with an automated elbow analysis.

    Parameters
    ----------
    adata : _type_
        anndata object
    pls_components_max : int, optional
        max number of PLS components to iterate through, by default 25
    elbow_metric : Literal['accuracy', 'explained_y'], optional
        what assessment metric to use for identifying the elbow, by default 'explained_y'
        - 'accuracy': Mean accuracy score across 5-fold CV of a logistic regression classifier trained
        on the PLS components (cannot directly use PLS model CV because it is technically a
        regression model on the one-hot encodings, and the high R^2 does not necessarily mean
        high classification accuracy)
        - 'explained_y': the fraction of variance in the test y-block (drug) by the PLS model prediction
    separate_by : Literal['perturbation', 'categorical'], by default 'perturbation'
        whether to separate PLS-DA by perturbation or categorical covariate
    control_confounders : List, optional
        controls for various confounders in the X-block of PLS-DA, by default doesn't control for confounders

    Returns
    -------
    assessment_df: pd.DataFrame
        results from each PLS component
    n_components: identified number of PLS components to use for downstream analysis
    """

    assessment_df = []
    iter_res = {}
    for n_components in trange(pls_components_max, 0, -1): #(1, pls_components_max + 1):
        models, assessment, X_pls = pls_da(
            adata = adata,
            n_components = n_components,
            control_confounders = control_confounders,
            assess = True,
            return_components = True,
            seed = seed,
            enc_X = None,
            enc_Y = None,
            separate_by = separate_by,
          pert_col = pert_col,
           cat_col = cat_col
        )
        assessment_df.append(assessment)
        iter_res[n_components] = (models, assessment, X_pls)

    assessment_df = pd.DataFrame(assessment_df)
    assessment_df = assessment_df.sort_values(by = 'n_components').reset_index(drop = True)

    y_ax = assessment_df[elbow_metric]
    x_ax = np.array(range(len(y_ax))) + 1
    kneedle = KneeLocator(x = x_ax, y = y_ax, curve='concave', direction='increasing')
    n_components = kneedle.elbow

    selected_res = iter_res[n_components]
    del iter_res

    return assessment_df, n_components, *selected_res

def pls_da_pipeline(
    adata,
    pert_ids: list | str,
    cat_ids: list | str,
    n_components: int = None,
    pert_col = 'drug',
    cat_col = 'cell_line',
    control_confounders: list = [],
    separate_by: Literal['perturbation', 'category'] = 'perturbation',
    covariate_associations: list = ['cell_line', 'drug', 'plate', 'phase', 'S_score', 'G2M_score', 'pcnt_mito'],
    scale: bool = False, # for TF adata it is already z-scored
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
    pert_col : str, optional
        perturbation column in `adata.obs`, by default 'drug'
    cat_col : str, optional
        categorical column in `adata.obs`, by default 'cell_line'
    control_confounders : list, optional
        columns in `adata.obs` to control for in PLS model fitting, by default does not control for any confounders
    separate_by : Literal[perturbation, category], optional
        whether to fit the PLS model to separate by the `pert_col` ('perturbation') or `cat_col` ('categorical), by default 'perturbation'
    covariate_associations : list, optional
        calculates univariate statistical associations between each PLS component and a covariate in `adata.obs`, by default['cell_line', 'drug', 'plate', 'phase', 'S_score', 'G2M_score', 'pcnt_mito']
        see `latent_association` for details
    scale : bool, optional
        whether to scale the adata object, by default False
        assumiing this is inferred TF activities, these are already Z-scored, so will note scale
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
    elif separate_by == 'categorical':
        separate_col = cat_col


    mask = (adata.obs[cat_col].isin(cat_ids)) & ((adata.obs[pert_col].isin(pert_ids)))
    adata_sub = adata[mask].copy()

    if scale:
        sc.pp.scale(adata_sub)

    # elbow selection
    if n_components is None:
        print("Run elbow selection") if verbose else None
        assessment_df, n_components, models, pls_assessment, X_pls = pls_elbow(
            adata = adata_sub,
            pls_components_max = 25,
            elbow_metric = 'explained_y',
            pert_col = pert_col,
            cat_col = cat_col,
            seed = seed
        )
        if file_prefix is not None:
            assessment_df.to_csv(file_prefix + '_PLS_elbow.csv')

    else:
        print("Fit PLS Model") if verbose else None
        assessment_df = None
        models, pls_assessment, X_pls = pls_da(
            adata = adata_sub,
            n_components = n_components,
            control_confounders = control_confounders,
            assess = True,
            return_components = True,
            seed = seed,
            enc_X = None,
            enc_Y = None,
            separate_by = separate_by,
            pert_col = pert_col,
            cat_col = cat_col,
        )

    print("Calculate covariate - PLS associations") if verbose else None
    if len(covariate_associations) == 0:
        covariate_associations = [separate_col]
    elif separate_col not in covariate_associations:
        covariate_associations.append(separate_col)

    adata_sub.obsm['X_pls'] = X_pls

    r2_df_linear = latent_association(
        adata = adata_sub,
        covariates = covariate_associations,
        model_type = 'linear',
        latent_label = 'pls',
        n_cores = n_cores,
        seed = seed
    )

    r2_df_nl = latent_association(
        adata = adata_sub,
        covariates = covariate_associations,
        model_type = 'nonlinear',
        latent_label = 'pls',
        n_cores = n_cores,
        seed = seed
    )

    r2_df = pd.concat([r2_df_linear, r2_df_nl])
    if file_prefix is not None:
        r2_df.to_csv(file_prefix + '_pls_associations.csv')

    X_umap = None
    if run_umap:
        print("Get UMAP") if verbose else None
        umap_model = umap.UMAP(
            n_neighbors=15,
            n_components=2,
            metric='euclidean',
            target_metric='categorical',
            random_state = seed)
        umap_model.fit(
            X_pls,
            adata_sub.obs[separate_col].cat.codes.values)
        X_umap = umap_model.transform(X_pls)
        models['umap_model'] = umap_model

    adata_sub.uns['pls'] = {'pls_mod': models['pls_model'],
                        'encoder_x': models['encoder_x'], 
                        'encoder_y': models['encoder_y'],
                        'pls_rank': models['pls_model'].n_components,
                        'elbow_analysis': assessment_df, 
                        'model_fit': pls_assessment
                        }

    adata_sub.obsm['X_pls'] = X_pls

    if run_umap:
        adata_sub.uns['umap_pls'] = {'umap_pls_mod': umap_model}
        adata_sub.obsm['X_umap_pls'] = X_umap

    return adata_sub, r2_df


def pc_pipeline(
    adata,
    pert_ids: list | str,
    cat_ids: list | str,
    n_components: int = None,
    pert_col = 'drug',
    cat_col = 'cell_line',
    use_hvgs: bool = False, 
    get_hvgs: bool = False, 
    scanpy_pca: bool = True, 
    run_umap: bool = True,
    covariate_associations: list = ['cell_line', 'drug', 'plate', 'phase', 'S_score', 'G2M_score', 'pcnt_mito'],
    file_prefix: str | None = None,
    verbose: bool = False,
    n_cores: int = -1,
    seed: int = 888,
    **hvgkwargs
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
    use_hvgs: bool, optional
        whether to run PCA on HVGs (True) or all genes (False), by default False
        False for TF activity, which already has limited features
    get_hvgs: bool, optional
        whether to calculate HVGs on the anndata subset (True) or use those in the passed object (False)
        only relevant if `use_hvgs` is True 
    scanpy_pca: bool, optional
        whether to use the scanpy PCA (whcih won't store the PCA model) (True) or a similar re-implementation (False), be default True
        scanpy version is probably faster
    run_umap: bool, optional
        whether to run umap (True) or not (False), by default True
    covariate_associations : list, optional
        calculates univariate statistical associations between each PC component and a covariate in `adata.obs`, by default['cell_line', 'drug', 'plate', 'phase', 'S_score', 'G2M_score', 'pcnt_mito']
        see `latent_association` for details
    file_prefix : str | None, optional
        saves assessment metrics to this file_prefix, by default None
    verbose : bool, optional
        whether to print information about progress, by default False
    n_cores : int, optional
        number of cores to parallelize latent space associations on
    seed : int, optional
        random state, by default 888
    **hvgkwargs: 
        key word arguments to pass to `sc.pp.highly_variable_genes` 
    """
    if type(cat_ids) == str:
        cat_ids = [cat_ids]
    if type(pert_ids) == str:
        pert_ids = [pert_ids]

    mask = (adata.obs[cat_col].isin(cat_ids)) & ((adata.obs[pert_col].isin(pert_ids)))
    adata_sub = adata[mask].copy()

    # TODO: there is a way to generalize this to the pp.embed_tf_activity that doesn't involve subsetting
    if use_hvgs:
        if not get_hvgs and 'highly_variable' not in adata_sub.var: 
            msg = 'There are no HVGs in the input adata, please either '
            msg += 'calculate them before passing input, set `use_hvgs` to False, or set `get_hvgs` to True'
            raise ValueError(msg)
        else:
            # as in other parts of scLEMBAS code
            sc.pp.highly_variable_genes(adata_sub, **hvgkwargs)
        hvgs = adata_sub.var_names[adata_sub.var['highly_variable']]
        adata_sub = adata_sub[:, hvgs]
    elif 'highly_variable' in adata_sub.var: # avoids default scanpy behavior
        del adata_sub.var['highly_variable']
    
    print("Run dimensionality reductions") if verbose else None
    pp.embed_tf_activity(
        tf_adata = adata_sub,
        scanpy_pca = scanpy_pca, # faster, will not be using model fit 
        n_components = 50 if n_components is None else n_components,
        pc_rank = 'automate' if n_components is None else n_components,
        run_pca = True,
        run_umap = run_umap,
        cluster_data = False
    )

    print("Calculate covariate - PC associations") if verbose else None
    r2_df = None
    if len(covariate_associations) != 0:

        r2_df_linear = latent_association(
            adata = adata_sub,
            covariates = covariate_associations,
            model_type = 'linear',
            latent_label = 'pca',
            n_cores = n_cores,
            seed = seed
        )

        r2_df_nl = latent_association(
            adata = adata_sub,
            covariates = covariate_associations,
            model_type = 'nonlinear',
            latent_label = 'pca',
            n_cores = n_cores,
            seed = seed
        )

        r2_df = pd.concat([r2_df_linear, r2_df_nl])
        if file_prefix is not None:
            r2_df.to_csv(file_prefix + '_pc_associations.csv')

    return adata_sub, r2_df


def latent_association(
    adata,
    covariates: List[str],
    model_type: Literal['linear', 'nonlinear'],
    n_cores: int,
    latent_label: str = 'pca',
    seed: int = 888):
    """Gets the linear association between a set of latent variables stored in the AnnData object
    and covariates in the `adata.obs`, returning the R^2. 

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
            r2 = pp.ss_explained_var(y, y_pred)
#             model = model_general.fit(X_cov, y)            
#             r2 = model.score(X_cov, y)
            r2_scores.append({_label_name: lv_idx + 1, "R2": r2})

        r2_df = pd.DataFrame(r2_scores)
        r2_df['covariate'] = cov_
        res.append(r2_df)
        
    r2_df = pd.concat(res, axis=0, ignore_index=True)
    r2_df = r2_df.pivot(index=_label_name, columns='covariate', values='R2').reset_index()
    r2_df.columns.name = None
    r2_df['model_type'] = model_type
    
    return r2_df

def visualize_latent_space(
    adata, 
    latent_label: Literal['pca', 'pls', 'umap', 'umap_pls'], 
    covariates: list, 
    components: list = [1,2], 
    n_frac: float = 0.2, 
    frac_col: str | None = None, 
    fig_title: str | None = None,
    legend = False, 
    seed: int = 888, 
    file_name: str | None = None, 

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
    components : list, optional
        which components to display on the 2 axes, by default [1,2]
    n_frac : float, optional
        subset to this fraction of the data for quicker visualization, by default 0.2
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
    """
        
    _label_name = latent_label.upper() if latent_label != 'pca' else 'PC'
    components = [_label_name + '{}'.format(component) for component in components]
    
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
        n_per_condition = int(np.round(adata.obs[frac_col].value_counts().min() * 0.2))

        # Subsample indices evenly per condition
        sampled_indices = (
            viz_df.groupby(frac_col)
            .sample(n=n_per_condition, random_state=seed)
            .index
        )

        # shuffle
        np.random.seed(seed)
        sampled_indices=np.random.permutation(sampled_indices)

        viz_df = proj_df.loc[sampled_indices, :].copy()
        
    max_cols = 3
    n_cols = min(max_cols, len(covariates))
    n_rows = math.ceil(len(covariates) / max_cols)

    fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(5.1*n_cols, 5.1*n_rows))
    ax = axes.flatten()
    for (i, covariate) in enumerate(covariates):
        sns.scatterplot(data = viz_df, x = components[0], y = components[1], hue = covariate, 
               s = 10, ax = ax[i])
        
        if not legend:
            ax[i].legend_.remove()
        ax[i].set_title(covariate.capitalize())
        
    if fig_title is not None:
        fig.suptitle(fig_title)
    fig.tight_layout()
    if file_name is not None:
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        
        
def visualize_latent_association(
    r2_df,
    top_components_cov: str | None = None,
    fig_title: str | None = None, 
    file_name: str | None = None,
):
    """Visualize the variance explained in the latent space, as calculated by`latent_association`."""
    latent_label = r2_df.columns[0]
    fig, ax = plt.subplots(ncols = 2, figsize = (10, 5))
    for i, model_type in enumerate(['linear', 'nonlinear']):
        viz_df = r2_df[r2_df.model_type == model_type]
        viz_df = viz_df.drop(columns = [latent_label, 'model_type']).copy()

        ranked_covars = viz_df.median(axis = 0).sort_values(ascending = False).index.tolist()

        viz_df = pd.melt(viz_df, var_name='covariate', value_name = 'var_explained')
        viz_df.covariate = pd.Categorical(viz_df.covariate, 
                                     categories = ranked_covars, 
                                     ordered=True)

        sns.boxplot(data = viz_df, x = 'covariate', y = 'var_explained', ax = ax[i])

        ax[i].set_title(model_type.capitalize())
        ax[i].set_ylabel('Fraction of Variance Explained')
        ax[i].set_xlabel('Covariates')

        for label in ax[i].get_xticklabels():
            label.set_rotation(45)

    if fig_title is not None:
        fig.suptitle(fig_title)
    fig.tight_layout()
    if file_name is not None:
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
    
    top_components = None
    if top_components_cov is not None:
        # r2_df[['PC', 'drug', 'model_type']].sort_values(by = ['model_type', 'drug'], ascending = False)
        top_components = r2_df[[latent_label, top_components_cov, 'model_type']].copy()
        top_components = top_components[top_components.model_type == 'linear']
        top_components = top_components.sort_values(by = top_components_cov, ascending=False).iloc[:2, :].reset_index(drop = True)

        print('The two {} components that best univariately separate by {} are components {} and {} explaining {:.2f}% and {:.2f}% of variance, respectively'.format(
            latent_label,
            top_components_cov,
            top_components[latent_label][0],
            top_components[latent_label][1], 
            top_components.drug[0]*100, 
            top_components.drug[1]*100,
        ))
        
        top_components = ['{}'.format(i) for i in top_components[latent_label]]
    return top_components



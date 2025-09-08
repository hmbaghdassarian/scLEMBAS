"""Lightweight version of pertpy's Mixscape implementation (v1.0.2) to be used without needing to install the full package."""

import copy
import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING, Literal, Sequence

from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
# from fast_array_utils.stats import mean, mean_var
from pandas.errors import PerformanceWarning
from anndata import AnnData
from scanpy import get
from scanpy._utils import _check_use_raw, sanitize_anndata
from scanpy.plotting import _utils
from scanpy.tools._utils import _choose_representation
from scipy.sparse import csr_matrix, issparse, spmatrix
from sklearn.mixture import GaussianMixture


# def center_adata(adata):
#     """Centers AnnData object without scaling"""

#     if issparse(adata.X):
#         # Calculate mean and broadcast properly for sparse subtraction
#         mean_vals = np.array(adata.X.mean(axis=0)).flatten()
#         # Create a sparse matrix with the same shape where each row is the mean
#         mean_matrix = csr_matrix(np.tile(mean_vals, (adata.X.shape[0], 1)))
#         adata.X = adata.X - mean_matrix
#     else:
#         adata.X = adata.X - adata.X.mean(axis=0)

class Mixscape:
    """identify perturbation effects in CRISPR screens by separating cells into perturbation groups.
    
    The pertpy documentation and best practices tutorial is slightly unclear, however, the pert_key
    argument in the perturbation_signature method is equivalent to the labels argument in the mixscape method. 
    pert_key in the tutorial is a binarized version, where cells are either "control" or "not control". 
    However, by providing the control argument, the pert_key is internally appropriately binarized even if 
    all perturbation labels are provided. 
    
    """

    def __init__(self):
        pass

    def perturbation_signature(
        self,
        adata: AnnData,
        pert_key: str,
        control: str,
        *,
        ref_selection_mode: Literal["nn", "split_by"] = "nn",
        split_by: str | None = None,
        n_neighbors: int = 20,
        use_rep: str | None = None,
        n_dims: int | None = 15,
        n_pcs: int | None = None,
        batch_size: int | None = None,
        copy: bool = False,
        **kwargs,
    ):
        """
        Compute a per-cell perturbation signature relative to control cells.

        The signature is the difference between each cell’s mRNA expression profile and
        the average profile of matched control cells (selected via `ref_selection_mode`).
        This follows the idea in Seurat Mixscape. Note that the original implementation
        computes signatures on **unscaled** data; we recommend doing the same.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix.
        pert_key : str
            Column in :attr:`adata.obs` with perturbation categories. Must include `control`.
        control : str
            Name of the control category present in :paramref:`pert_key`.
        ref_selection_mode : {'nn', 'split_by'}, default 'nn'
            Strategy to select control reference cells.
            - ``'nn'``: For each perturbed cell, use the ``n_neighbors`` most similar
            control cells (in the chosen representation). *Note, selecting 'nn' does not negate the 
            ``'split_by'`` argument; will still choose the neargest-neighbors, but within the biological replicate
            - ``'split_by'``: Use all control cells from the same group defined by
            :paramref:`split_by` (e.g., biological replicate).
        split_by : str or None, default None
            Column in :attr:`adata.obs` indicating splits (e.g., replicate) used when
            ``ref_selection_mode='split_by'``.
        n_neighbors : int, default 20
            Number of nearest control neighbors to use when ``ref_selection_mode='nn'``.
        use_rep : str or None, default None
            Representation to use. Accepts ``'X'`` or a key in :attr:`adata.obsm`.
            If ``None``: use :attr:`adata.X` when ``adata.n_vars < 50``, else try
            ``'X_pca'`` (computed with default parameters if absent).
        n_dims : int or None, default 15
            Number of leading dimensions from the chosen representation to use. If
            ``None``, use all available dimensions.
        n_pcs : int or None, default None
            Number of principal components to compute when a PCA representation is used.
            If ``n_pcs == 0`` and ``use_rep is None``, fall back to :attr:`adata.X`.
        batch_size : int or None, default None
            Batch size used to compute signatures. If ``None``, computes in a single
            pass (higher memory). Batched mode can be inefficient for sparse data.
        copy : bool, default False
            If ``True``, return a copy of ``adata`` with signatures written to
            ``.layers['X_pert']``. Otherwise, modify ``adata`` in place and return ``None``.
        **kwargs
            Additional keyword arguments forwarded to :class:`pynndescent.NNDescent`
            when ``ref_selection_mode='nn'``.

        Returns
        -------
        AnnData or None
            If ``copy=True``, a copy of ``adata`` with the perturbation signatures stored in
            ``adata.layers['X_pert']``. If ``copy=False``, modifies ``adata`` in place and
            returns ``None``.

        Notes
        -----
        Inspired by Seurat Mixscape:
        https://satijalab.org/seurat/reference/runmixscape

        Examples
        --------
        Calculate perturbation signatures per cell:

        >>> import pertpy as pt
        >>> mdata = pt.dt.papalexi_2021()
        >>> ms = pt.tl.Mixscape()
        >>> ms.perturbation_signature(mdata["rna"], "perturbation", "NT", split_by="replicate")
        """
        ...

        if ref_selection_mode not in ["nn", "split_by"]:
            raise ValueError("ref_selection_mode must be either 'nn' or 'split_by'.")
        if ref_selection_mode == "split_by" and split_by is None:
            raise ValueError("split_by must be provided if ref_selection_mode is 'split_by'.")

        if copy:
            adata = adata.copy()

        adata.layers["X_pert"] = adata.X.copy()

        # Work with LIL for efficient indexing but don't store it in AnnData as LIL is not supported anymore
        X_pert_lil = adata.layers["X_pert"].tolil() if issparse(adata.layers["X_pert"]) else adata.layers["X_pert"]

        control_mask = adata.obs[pert_key] == control

        if ref_selection_mode == "split_by":
            for split in adata.obs[split_by].unique():
                split_mask = adata.obs[split_by] == split
                control_mask_group = control_mask & split_mask
                control_mean_expr = mean(adata.X[control_mask_group], axis=0)
                X_pert_lil[split_mask] = (
                    np.repeat(control_mean_expr.reshape(1, -1), split_mask.sum(), axis=0) - X_pert_lil[split_mask]
                )
        else:
            if split_by is None:
                split_masks = [np.full(adata.n_obs, True, dtype=bool)]
            else:
                split_obs = adata.obs[split_by]
                split_masks = [split_obs == cat for cat in split_obs.unique()]

            representation = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs)
            if n_dims is not None and n_dims < representation.shape[1]:
                representation = representation[:, :n_dims]

            from pynndescent import NNDescent

            for split_mask in tqdm(split_masks):
                control_mask_split = control_mask & split_mask
                R_split = representation[split_mask]
                R_control = representation[np.asarray(control_mask_split)]
                eps = kwargs.pop("epsilon", 0.1)
                nn_index = NNDescent(R_control, **kwargs)
                indices, _ = nn_index.query(R_split, k=n_neighbors, epsilon=eps)
                X_control = np.expm1(adata.X[np.asarray(control_mask_split)])
                n_split = split_mask.sum()
                n_control = X_control.shape[0]

                if batch_size is None:
                    col_indices = np.ravel(indices)
                    row_indices = np.repeat(np.arange(n_split), n_neighbors)
                    neigh_matrix = csr_matrix(
                        (np.ones_like(col_indices, dtype=np.float64), (row_indices, col_indices)),
                        shape=(n_split, n_control),
                    )
                    neigh_matrix /= n_neighbors
                    X_pert_lil[np.asarray(split_mask)] = (
                        sc.pp.log1p(neigh_matrix @ X_control) - X_pert_lil[np.asarray(split_mask)]
                    )
                else:
                    split_indices = np.where(split_mask)[0]
                    for i in range(0, n_split, batch_size):
                        size = min(i + batch_size, n_split)
                        select = slice(i, size)
                        batch = np.ravel(indices[select])
                        split_batch = split_indices[select]
                        size = size - i
                        means_batch = X_control[batch]
                        batch_reshaped = means_batch.reshape(size, n_neighbors, -1)
                        means_batch, _ = mean_var(batch_reshaped, axis=1)
                        X_pert_lil[split_batch] = np.log1p(means_batch) - X_pert_lil[split_batch]    

        if issparse(X_pert_lil):
            adata.layers["X_pert"] = X_pert_lil.tocsr()
        else:
            adata.layers["X_pert"] = X_pert_lil

        if copy:
            return adata
        
    def mixscape(
        self,
        adata: AnnData,
        labels: str,
        control: str,
        *,
        new_class_name: str | None = "mixscape_class",
        layer: str | None = None,
        min_de_genes: int | None = 5,
        logfc_threshold: float | None = 0.25,
        de_layer: str | None = None,
        test_method: str | None = "wilcoxon",
        iter_num: int | None = 10,
        scale: bool | None = True,
        split_by: str | None = None,
        pval_cutoff: float | None = 5e-2,
        perturbation_type: str | None = "KO",
        random_state: int | None = 0,
        copy: bool | None = False,
        **gmmkwargs,
    ):
        """
        Identify perturbed and non-perturbed gRNA expressing cells across multiple treatments,
        conditions, or chemical perturbations.

        This implementation resembles Seurat's Mixscape
        (https://satijalab.org/seurat/reference/runmixscape).

        Parameters
        ----------
        adata : AnnData
            The annotated data object.
        labels : str
            The column of `.obs` with target gene labels.
        control : str
            Control category from the `labels` column.
        new_class_name : str
            Name of mixscape classification to be stored in `.obs`.
        layer : str, optional
            Key from `adata.layers` whose value will be used to perform tests on.
            Default is `.layers["X_pert"]`.
        min_de_genes : int
            Required number of differentially expressed genes for the method to separate
            perturbed and non-perturbed cells.
        logfc_threshold : float, default=0.25
            Limit testing to genes which show, on average, at least this fold-change
            (log-scale) between the two groups of cells.
        de_layer : str or None, optional
            Layer to use for identifying differentially expressed genes.
            If `None`, `adata.X` is used.
        test_method : str
            Method to use for differential expression testing.
        iter_num : int
            Number of `normalmixEM` iterations to run if convergence does not occur.
        scale : bool
            Whether to scale the data specified in `layer` before running the GaussianMixture model.
        split_by : str, optional
            Column in `.obs` with experimental condition/cell type annotation, if
            perturbations are condition/cell type-specific.
        pval_cutoff : float
            P-value cut-off for selection of significantly DE genes.
        perturbation_type : str
            Type of CRISPR perturbation expected for labeling mixscape classifications.
        random_state : int
            Random seed for the GaussianMixture model.
        copy : bool
            If True, return a copy of `adata` with results written to `.obs`.
        **gmmkwargs : dict
            Additional keyword arguments passed to scikit-learn's Gaussian Mixture Model.

        Returns
        -------
        adata : AnnData or None
            If `copy=True`, returns a copy of `adata` with classification results stored in `.obs`.
            Otherwise, modifies `adata` in place.

        Notes
        -----
        The following columns will be added to `adata.obs`:

        - `mixscape_class` : pandas.Series
            Classification result with cells labeled as perturbed (e.g. KO) or non-perturbed (NP).
        - `mixscape_class_global` : pandas.Series
            Global classification result (perturbed, NP, or NT).
        - `mixscape_class_p_ko` : pandas.Series
            Posterior probabilities used to determine if a cell is KO (default; >0.5) or NP.
            The name of this column will change to match `perturbation_type`.

        Examples
        --------
        Calculate perturbation signature for each cell:

        >>> import pertpy as pt
        >>> mdata = pt.dt.papalexi_2021()
        >>> ms_pt = pt.tl.Mixscape()
        >>> ms_pt.perturbation_signature(mdata["rna"], "perturbation", "NT", split_by="replicate")
        >>> ms_pt.mixscape(mdata["rna"], "gene_target", "NT", layer="X_pert")
        """

        if copy:
            adata = adata.copy()

        if split_by is None:
            split_masks = [np.full(adata.n_obs, True, dtype=bool)]
            categories = ["all"]
        else:
            split_obs = adata.obs[split_by]
            categories = split_obs.unique()
            split_masks = [split_obs == category for category in categories]

        perturbation_markers = self._get_perturbation_markers(
            adata=adata,
            split_masks=split_masks,
            categories=categories,
            labels=labels,
            control=control,
            layer=de_layer,
            pval_cutoff=pval_cutoff,
            min_de_genes=min_de_genes,
            logfc_threshold=logfc_threshold,
            test_method=test_method,
        )

        adata_comp = adata
        if layer is not None:
            X = adata_comp.layers[layer]
        else:
            try:
                X = adata_comp.layers["X_pert"]
            except KeyError:
                raise KeyError(
                    "No 'X_pert' found in .layers! Please run perturbation_signature first to calculate perturbation signature!"
                ) from None

        # initialize return variables
        adata.obs[f"{new_class_name}_p_{perturbation_type.lower()}"] = 0
        adata.obs[new_class_name] = adata.obs[labels].astype(str)
        adata.obs[f"{new_class_name}_global"] = np.empty(
            [
                adata.n_obs,
            ],
            dtype=np.object_,
        )
        gv_list: dict[str, dict] = {}

        adata.obs[f"{new_class_name}_p_{perturbation_type.lower()}"] = 0.0
        for split, split_mask in enumerate(split_masks):
            category = categories[split]
            gene_targets = list(set(adata[split_mask].obs[labels]).difference([control]))
            for gene in gene_targets:
                post_prob = 0
                orig_guide_cells = (adata.obs[labels] == gene) & split_mask
                orig_guide_cells_index = list(orig_guide_cells.index[orig_guide_cells])
                nt_cells = (adata.obs[labels] == control) & split_mask
                all_cells = orig_guide_cells | nt_cells

                if len(perturbation_markers[(category, gene)]) == 0:
                    adata.obs.loc[orig_guide_cells, new_class_name] = f"{gene} NP"

                else:
                    de_genes = perturbation_markers[(category, gene)]
                    de_genes_indices = np.where(np.isin(adata.var_names, list(de_genes)))[0]

                    dat = X[np.asarray(all_cells)][:, de_genes_indices]
                    if scale:
                        dat = sc.pp.scale(dat)

                    converged = False
                    n_iter = 0
                    old_classes = adata.obs[new_class_name][all_cells]

                    nt_cells_dat_idx = all_cells[all_cells].index.get_indexer(nt_cells[nt_cells].index)
                    nt_cells_mean = np.mean(dat[nt_cells_dat_idx], axis=0)

                    while not converged and n_iter < iter_num:
                        # Get all cells in current split&Gene
                        guide_cells = (adata.obs[new_class_name] == gene) & split_mask

                        # get average value for each gene over all selected cells
                        # all cells in current split&Gene minus all NT cells in current split
                        # Each row is for each cell, each column is for each gene, get mean for each column
                        guide_cells_dat_idx = all_cells[all_cells].index.get_indexer(guide_cells[guide_cells].index)
                        guide_cells_mean = np.mean(dat[guide_cells_dat_idx], axis=0)
                        vec = guide_cells_mean - nt_cells_mean

                        # project cells onto the perturbation vector
                        if isinstance(dat, spmatrix):
                            pvec = dat.dot(vec) / np.dot(vec, vec)
                        else:
                            pvec = np.dot(dat, vec) / np.dot(vec, vec)
                        pvec = pd.Series(np.asarray(pvec).flatten(), index=list(all_cells.index[all_cells]))

                        if n_iter == 0:
                            gv = pd.DataFrame(columns=["pvec", labels])
                            gv["pvec"] = pvec
                            gv[labels] = control
                            gv.loc[guide_cells, labels] = gene
                            if gene not in gv_list:
                                gv_list[gene] = {}
                            gv_list[gene][category] = gv

                        means_init = np.array([[pvec[nt_cells].mean()], [pvec[guide_cells].mean()]])
                        std_init = np.array([pvec[nt_cells].std(), pvec[guide_cells].std()])
                        mm = MixscapeGaussianMixture(
                            n_components=2,
                            covariance_type="spherical",
                            means_init=means_init,
                            precisions_init=1 / (std_init**2),
                            random_state=random_state,
                            max_iter=100,
                            fixed_means=[pvec[nt_cells].mean(), None],
                            fixed_covariances=[pvec[nt_cells].std() ** 2, None],
                            **gmmkwargs,
                        ).fit(np.asarray(pvec).reshape(-1, 1))
                        probabilities = mm.predict_proba(np.array(pvec[orig_guide_cells_index]).reshape(-1, 1))
                        lik_ratio = probabilities[:, 0] / probabilities[:, 1]
                        post_prob = 1 / (1 + lik_ratio)

                        # based on the posterior probability, assign cells to the two classes
                        ko_mask = post_prob > 0.5
                        adata.obs.loc[np.array(orig_guide_cells_index)[ko_mask], new_class_name] = gene
                        adata.obs.loc[np.array(orig_guide_cells_index)[~ko_mask], new_class_name] = f"{gene} NP"

                        if sum(adata.obs[new_class_name][split_mask] == gene) < min_de_genes:
                            adata.obs.loc[guide_cells, new_class_name] = "NP"
                            converged = True
                        current_classes = adata.obs[new_class_name][all_cells]
                        if (current_classes == old_classes).all():
                            converged = True
                        old_classes = current_classes

                        n_iter += 1

                    adata.obs.loc[(adata.obs[new_class_name] == gene) & split_mask, new_class_name] = (
                        f"{gene} {perturbation_type}"
                    )

                adata.obs[f"{new_class_name}_global"] = [a.split(" ")[-1] for a in adata.obs[new_class_name]]
                adata.obs.loc[orig_guide_cells_index, f"{new_class_name}_p_{perturbation_type.lower()}"] = post_prob
        adata.uns["mixscape"] = gv_list

        if copy:
            return adata
        
    def _get_perturbation_markers(
            self,
            adata: AnnData,
            split_masks: list[np.ndarray],
            categories: list[str],
            labels: str,
            control: str,
            layer: str,
            pval_cutoff: float,
            min_de_genes: float,
            logfc_threshold: float,
            test_method: str,
        ) -> dict[tuple, np.ndarray]:
            """Determine gene sets across all splits/groups through differential gene expression.

            Args:
                adata: :class:`~anndata.AnnData` object
                split_masks: List of boolean masks for each split/group.
                categories: List of split/group names.
                labels: The column of `.obs` with target gene labels.
                control: Control category from the `labels` column.
                layer: Key from adata.layers whose value will be used to compare gene expression.
                pval_cutoff: P-value cut-off for selection of significantly DE genes.
                min_de_genes: Required number of genes that are differentially expressed for method to separate perturbed and non-perturbed cells.
                logfc_threshold: Limit testing to genes which show, on average, at least X-fold difference (log-scale) between the two groups of cells.
                test_method: Method to use for differential expression testing.

            Returns:
                Set of column indices.
            """
            perturbation_markers: dict[tuple, np.ndarray] = {}  # type: ignore
            for split, split_mask in enumerate(split_masks):
                category = categories[split]
                # get gene sets for each split
                gene_targets = list(set(adata[split_mask].obs[labels]).difference([control]))
                adata_split = adata[split_mask].copy()
                # find top DE genes between cells with targeting and non-targeting gRNAs
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    warnings.simplefilter("ignore", PerformanceWarning)
                    sc.tl.rank_genes_groups(
                        adata_split,
                        layer=layer,
                        groupby=labels,
                        groups=gene_targets,
                        reference=control,
                        method=test_method,
                        use_raw=False,
                    )
                    # get DE genes for each target gene
                    for gene in gene_targets:
                        logfc_threshold_mask = (
                            np.abs(adata_split.uns["rank_genes_groups"]["logfoldchanges"][gene]) >= logfc_threshold
                        )
                        de_genes = adata_split.uns["rank_genes_groups"]["names"][gene][logfc_threshold_mask]
                        pvals_adj = adata_split.uns["rank_genes_groups"]["pvals_adj"][gene][logfc_threshold_mask]
                        de_genes = de_genes[pvals_adj < pval_cutoff]
                        if len(de_genes) < min_de_genes:
                            de_genes = np.array([])
                        perturbation_markers[(category, gene)] = de_genes

            return perturbation_markers
        
class MixscapeGaussianMixture(GaussianMixture):
    def __init__(
        self,
        n_components: int,
        fixed_means: Sequence[float] | None = None,
        fixed_covariances: Sequence[float] | None = None,
        **kwargs,
    ):
        """Custom Gaussian Mixture Model where means and covariances can be fixed for specific components.

        Args:
            n_components: Number of Gaussian components
            fixed_means: Means to fix (use None for those that should be estimated)
            fixed_covariances: Covariances to fix (use None for those that should be estimated)
            **kwargs: Additional arguments passed to scikit-learn's GaussianMixture
        """
        super().__init__(n_components=n_components, **kwargs)
        self.fixed_means = fixed_means
        self.fixed_covariances = fixed_covariances

        self.fixed_mean_indices = []
        self.fixed_mean_values = []
        if fixed_means is not None:
            self.fixed_mean_indices = [i for i, m in enumerate(fixed_means) if m is not None]
            if self.fixed_mean_indices:
                self.fixed_mean_values = np.array([fixed_means[i] for i in self.fixed_mean_indices])

        self.fixed_cov_indices = []
        self.fixed_cov_values = []
        if fixed_covariances is not None:
            self.fixed_cov_indices = [i for i, c in enumerate(fixed_covariances) if c is not None]
            if self.fixed_cov_indices:
                self.fixed_cov_values = np.array([fixed_covariances[i] for i in self.fixed_cov_indices])

    def _m_step(self, X: np.ndarray, log_resp: np.ndarray):
        """Modified M-step to respect fixed means and covariances."""
        super()._m_step(X, log_resp)

        if self.fixed_mean_indices:
            self.means_[self.fixed_mean_indices] = self.fixed_mean_values

        if self.fixed_cov_indices:
            self.covariances_[self.fixed_cov_indices] = self.fixed_cov_values

        return self

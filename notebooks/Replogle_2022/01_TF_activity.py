#!/usr/bin/env python
# coding: utf-8

# In[25]:


import os

import numpy as np
import scanpy as sc
from kneed import KneeLocator

import seaborn as sns
import matplotlib.pyplot as plt

import sys
sclembas_path = '/home/hmbaghda/Projects/scLEMBAS/'
sys.path.insert(1, os.path.join(sclembas_path))
from scLEMBAS.preprocess import get_tf_activity, tf_to_adata
from scLEMBAS import io


# In[26]:


data_path = '/nobackup/users/hmbaghda/scLEMBAS/analysis'
seed = 888


# In[ ]:


author = 'Replogle'


# # Preprocessing of gene expression:

# In[27]:


rpe1 = sc.read_h5ad(os.path.join(data_path, 'raw', 'rpe1_normalized_singlecell_01.h5ad'))
rpe1_scperturb = sc.read_h5ad(os.path.join(data_path, 'raw', 'scperturb_ReplogleWeissman2022_rpe1.h5ad'))

k562 = sc.read_h5ad(os.path.join(data_path, 'raw', 'K562_essential_normalized_singlecell_01.h5ad'))
k562_scperturb = sc.read_h5ad(os.path.join(data_path, 'raw', 'scperturb_ReplogleWeissman2022_K562_essential.h5ad'))


# Some basic QC:

# In[28]:


# # this doesn't filter anything out -- good sign, indicates already QC'd
# sc.pp.filter_cells(rpe1, min_genes=100)
# sc.pp.filter_genes(rpe1, min_cells=100)

# sc.pp.filter_cells(k562, min_genes=100)
# sc.pp.filter_genes(k562, min_cells=100)


# In[29]:


np.random.seed(seed)
subset = np.random.choice(k562.obs_names, int(5e3), replace = False)
sc.pl.violin(
    k562[subset, :],
    ["mitopercent", "core_adjusted_UMI_count"],
    jitter=0.4,
    multi_panel=True,
)

# filters out 969 out of 310385 cells
k562 = k562[k562.obs["core_adjusted_UMI_count"] < int(35e3), :]


# In[30]:


np.random.seed(seed)
subset = np.random.choice(rpe1.obs_names, int(5e3), replace = False)
sc.pl.violin(
    rpe1[subset, :],
    ["mitopercent", "core_adjusted_UMI_count"],
    jitter=0.4,
    multi_panel=True,
)

# filters out 771 out of 247914 cells
rpe1 = rpe1[rpe1.obs["core_adjusted_UMI_count"] < int(35e3), :]


# Use scPerturb data to map the ENSG to the gene names:

# In[31]:


if rpe1_scperturb.var.ensembl_id.nunique() != rpe1_scperturb.var.shape[0]:
    raise ValueError('There is a not a 1-to-1 mapping of gene name to ensembl ID')
if sorted(rpe1_scperturb.var.ensembl_id) != sorted(rpe1.var_names):
    raise ValueError('There is not a 1-to-1 correspondence between the original data and scPerturb')
    
gene_map = dict(zip(rpe1_scperturb.var.ensembl_id, rpe1_scperturb.var.index))
rpe1.var_names = rpe1.var_names.map(gene_map)
del rpe1_scperturb


# In[32]:


if k562_scperturb.var.ensembl_id.nunique() != k562_scperturb.var.shape[0]:
    raise ValueError('There is a not a 1-to-1 mapping of gene name to ensembl ID')
if sorted(k562_scperturb.var.ensembl_id) != sorted(k562.var_names):
    raise ValueError('There is not a 1-to-1 correspondence between the original data and scPerturb')
    
gene_map = dict(zip(k562_scperturb.var.ensembl_id, k562_scperturb.var.index))
k562.var_names = k562.var_names.map(gene_map)
del k562_scperturb


# Exclude perturbations with fewer than 50 cells:

# In[33]:


thresh = 50


# In[34]:


fig, ax = plt.subplots(ncols = 2, figsize = (10,4))
sns.kdeplot(rpe1.obs.gene.value_counts(), ax = ax[0])
ax[0].set_xscale('log')
ax[0].set_xlabel('No. of Cells per Perturbation')
ax[0].set_title('RPE1')
ax[0].axvline(x=thresh, color='r', linestyle='--', linewidth=1)


n_perturbations_og = rpe1.obs.gene.nunique()
retain = rpe1.obs.gene.value_counts()[rpe1.obs.gene.value_counts() >= thresh].index.tolist()
rpe1 = rpe1[rpe1.obs.gene.isin(retain),:]
n_perturbations = rpe1.obs.gene.nunique()
# print('RPE1: Filtering for perturbations present in atleast {} single-cells decreases the no. of perturbations from {} to {}'.format(thresh, n_perturbations_og, ))


ax[0].annotate('n = {}'.format(n_perturbations_og - n_perturbations), xy=(0.15, 0.85), xycoords='axes fraction',
            fontsize=9, color='black', ha='center')
ax[0].annotate('n = {}'.format(n_perturbations), xy=(0.6, 0.85), xycoords='axes fraction',
            fontsize=9, color='black', ha='center')



sns.kdeplot(k562.obs.gene.value_counts(), ax = ax[1])
ax[1].set_xscale('log')
ax[1].set_xlabel('No. of Cells per Perturbation')
ax[1].set_title('k562')
ax[1].axvline(x=thresh, color='r', linestyle='--', linewidth=1)


n_perturbations_og = k562.obs.gene.nunique()
retain = k562.obs.gene.value_counts()[k562.obs.gene.value_counts() >= thresh].index.tolist()
k562 = k562[k562.obs.gene.isin(retain),:]
n_perturbations = k562.obs.gene.nunique()
# print('k562: Filtering for perturbations present in atleast {} single-cells decreases the no. of perturbations from {} to {}'.format(thresh, n_perturbations_og, ))


ax[1].annotate('n = {}'.format(n_perturbations_og - n_perturbations), xy=(0.15, 0.85), xycoords='axes fraction',
            fontsize=9, color='black', ha='center')
ax[1].annotate('n = {}'.format(n_perturbations), xy=(0.6, 0.85), xycoords='axes fraction',
            fontsize=9, color='black', ha='center')

fig.tight_layout()
("")




batch_key = 'batch'
rpe1.obs[batch_key] = 'rpe1'
k562.obs[batch_key] = 'k562'

adata = sc.concat([rpe1, k562], label=None)

adata


# Since 1) taking the intersection of genes already reduced the space to 7k genes and 2) HVG selection expects logarithmized data, we skip this part. This differs from processing of the Kang dataset.

# In[65]:


# sc.pp.highly_variable_genes(adata, n_top_genes=3000, batch_key=None, flavor = 'seurat')
sc.tl.pca(adata, 
          zero_center = False, # rather than used scaled data, calculates covariance matrix internally
          n_comps = 50,
          random_state = seed, 
          use_highly_variable = False,
         )

variance_ratio = adata.uns['pca']['variance_ratio']
pcs = np.array(range(len(variance_ratio))) + 1
kneedle = KneeLocator(x = pcs, y = variance_ratio, curve='convex', direction='decreasing')
adata.uns['pca']['pca_rank'] = kneedle.elbow
sc.pl.pca_variance_ratio(adata)
print('The elbow was automatically identified at PC {}'.format(adata.uns['pca']['pca_rank']))


# In[ ]:


sc.pp.neighbors(adata, n_pcs = adata.uns['pca']['pca_rank'])
sc.tl.umap(adata)

sc.pl.umap(adata, color=[batch_key], wspace=1)
adata.write_h5ad(os.path.join(data_path, 'interim', author + '_embedded.h5ad'))


# We can see that the data is well-mixesd between K562 and RPE1 in UMAP space, so there is no need to apply a batch correction. Next, let's get the TFs:

# In[ ]:


kwargs = {'args' : {'wsum' : {'times': int(1e3), 'batch_size': int(1e4)},
                       'ulm' : {'batch_size': int(1e4)}, 
                        'mlm': {'batch_size': int(1e4)}
                       }, 
#          'methods': ['wsum', 'ulm', 'mlm'], 
         'cns_metds': ['ulm_estimate', 'mlm_estimate', 'wsum_estimate']}
# default is wsum_norm, which introduces inf values that result in nan when z-scoring..
# also intuitively doesnt make sense to z-score an already normalized value
# particularly when the other z-scores or on the non-normalized values



adata = get_tf_activity(adata, organism = 'human', grn = 'collectri', verbose = True,
                consensus = True, hvg = True,
                min_n = 5, use_raw = False, filter_pvals = False, pval_thresh = 0.05, **kwargs)

adata.write_h5ad(os.path.join(data_path, 'processed', author + '_expr_scored.h5ad'))


#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os

import numpy as np
import scanpy as sc
from kneed import KneeLocator
import h5py

import seaborn as sns
import matplotlib.pyplot as plt

import sys
sclembas_path = '/home/hmbaghda/Projects/scLEMBAS/'
sys.path.insert(1, os.path.join(sclembas_path))
from scLEMBAS.preprocess import get_tf_activity, tf_to_adata
from scLEMBAS import io


# In[9]:


data_path = '/nobackup/users/hmbaghda/scLEMBAS/analysis'
seed = 888


# In[10]:


author = 'Replogle'


# # Preprocessing of gene expression:

# In[11]:


rpe1 = sc.read_h5ad(os.path.join(data_path, 'raw', 'rpe1_normalized_singlecell_01.h5ad'))
rpe1_scperturb = sc.read_h5ad(os.path.join(data_path, 'raw', 'scperturb_ReplogleWeissman2022_rpe1.h5ad'))

k562 = sc.read_h5ad(os.path.join(data_path, 'raw', 'K562_essential_normalized_singlecell_01.h5ad'))
k562_scperturb = sc.read_h5ad(os.path.join(data_path, 'raw', 'scperturb_ReplogleWeissman2022_K562_essential.h5ad'))


# Some basic QC:

# In[12]:


# # this doesn't filter anything out -- good sign, indicates already QC'd
# sc.pp.filter_cells(rpe1, min_genes=100)
# sc.pp.filter_genes(rpe1, min_cells=100)

# sc.pp.filter_cells(k562, min_genes=100)
# sc.pp.filter_genes(k562, min_cells=100)


# In[13]:


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


# In[14]:


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

# In[15]:


if rpe1_scperturb.var.ensembl_id.nunique() != rpe1_scperturb.var.shape[0]:
    raise ValueError('There is a not a 1-to-1 mapping of gene name to ensembl ID')
if sorted(rpe1_scperturb.var.ensembl_id) != sorted(rpe1.var_names):
    raise ValueError('There is not a 1-to-1 correspondence between the original data and scPerturb')
    
gene_map = dict(zip(rpe1_scperturb.var.ensembl_id, rpe1_scperturb.var.index))
rpe1.var_names = rpe1.var_names.map(gene_map)
del rpe1_scperturb


# In[16]:


if k562_scperturb.var.ensembl_id.nunique() != k562_scperturb.var.shape[0]:
    raise ValueError('There is a not a 1-to-1 mapping of gene name to ensembl ID')
if sorted(k562_scperturb.var.ensembl_id) != sorted(k562.var_names):
    raise ValueError('There is not a 1-to-1 correspondence between the original data and scPerturb')
    
gene_map = dict(zip(k562_scperturb.var.ensembl_id, k562_scperturb.var.index))
k562.var_names = k562.var_names.map(gene_map)
del k562_scperturb


# Exclude perturbations with fewer than 50 cells:

# In[17]:


thresh = 50


# In[18]:


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


# # Start

# In[19]:


# k562backup = k562.copy()
# rpe1backup = rpe1.copy()

# # k562 = k562backup.copy()
# # rpe1 = rpe1backup.copy()


# In[28]:


# # TDO DELETE
# np.random.seed(seed)
# subset_cells = np.random.choice(k562.obs_names, int(5e2), replace = False)
# subset_genes = np.random.choice(k562.var_names, int(1e3), replace = False)
# k562 = k562[subset_cells, subset_genes]


# np.random.seed(seed)
# subset_cells = np.random.choice(rpe1.obs_names, int(5e2), replace = False)
# subset_genes = np.random.choice(rpe1.var_names, int(1e3), replace = False)
# rpe1 = rpe1[subset_cells, subset_genes]


# # End

# In[21]:


batch_key = 'batch'
rpe1.obs[batch_key] = 'rpe1'
k562.obs[batch_key] = 'k562'

adata = sc.concat([rpe1, k562], label=None)
adata.obs_names_make_unique()

adata


# In[22]:


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


# In[23]:


sc.pp.neighbors(adata, n_pcs = adata.uns['pca']['pca_rank'])
sc.tl.umap(adata)
adata.write_h5ad(os.path.join(data_path, 'interim', author + '_expr_embedded.h5ad'))


# In[24]:


np.random.seed(seed)
cell_order = np.random.permutation(adata.obs_names)
sc.pl.umap(adata[cell_order, :], color=[batch_key], wspace=1)


# The two datasets are reasonably well-mixed. We proceed with TF activity inference:

# In[26]:


kwargs = {'args' : {'wsum' : {'times': int(1e3), 'batch_size': int(1e4)},
                       'ulm' : {'batch_size': int(1e4)}, 
                        'mlm': {'batch_size': int(1e4)}
                       }, 
         'cns_metds': ['ulm_estimate', 'mlm_estimate', 'wsum_estimate']}


kwargs = {'args' : {'wsum' : {'times': int(1e1), 'batch_size': int(1e4)},
                       'ulm' : {'batch_size': int(1e4)}, 
                        'mlm': {'batch_size': int(1e4)}
                       }, 
         'cns_metds': ['ulm_estimate', 'mlm_estimate', 'wsum_estimate']}

adata = get_tf_activity(adata, organism = 'human', grn = 'collectri', verbose = True,
                consensus = True, 
                        hvg = False,# different from Kang
                min_n = 5, use_raw = False, filter_pvals = False, pval_thresh = 0.05, **kwargs)

adata.write_h5ad(os.path.join(data_path, 'processed', author + '_expr_scored.h5ad'))


# In[27]:


fn_csv = os.path.join(data_path, 'interim', author + '_TF_activity.csv') 

for key in adata.obsm:
    if key.endswith('estimate') or key.endswith('pvals'):
        fn_csv_ = fn_csv.replace('TF_activity', key + '_TF_activity')
        adata.obsm[key].to_csv(fn_csv_)

tf_adata = tf_to_adata(adata, estimate_key = 'consensus_estimate')
io.write_tfad(tf_adata, file_name = os.path.join(data_path, 'interim', author + '_tf_activity.h5ad'))


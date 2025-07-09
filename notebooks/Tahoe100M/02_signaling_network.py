#!/usr/bin/env python
# coding: utf-8

# In[150]:


import os
import ast
import json
from tqdm import trange

import numpy as np
import pandas as pd

from scipy.stats import f_oneway, kruskal
from scipy import stats
from sklearn.metrics import normalized_mutual_info_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

import scanpy as sc

import matplotlib.pyplot as plt
import seaborn as sns

import sys
sclembas = '/home/hmbaghda/Projects/scLEMBAS'
sys.path.insert(1, os.path.join(sclembas))
from scLEMBAS import io
from scLEMBAS import parse_network as pn
from scLEMBAS import preprocess as pp


# In[2]:


n_cores = 30
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)

seed = 888
data_path = '/nobackup/users/hmbaghda/scLEMBAS/analysis'
author = 'Tahoe100M'


# In[3]:


tf_adata = io.read_tfad(file_name = os.path.join(data_path, 'interim', author + 'consensus_tf_activity.h5ad'))
# need to run without the control
obs = tf_adata.obs.copy()
obs = obs[obs.drug != 'DMSO_TF']

# parse the drug target
obs.drug_target = obs.drug_target.astype(str).apply(lambda x: ast.literal_eval(x))


# # Signaling Network Parsing

# Load and parse input signaling network:

# In[4]:


source_label = 'source_genesymbol'
target_label = 'target_genesymbol'
weight_label = 'mode_of_action'
stimulation_label = 'consensus_stimulation'
inhibition_label = 'consensus_inhibition'


# In[5]:


sn_ppis = pn.load_network('omnipath', organism = 'human', static = True)
sn_ppis = pn.correct_network(sn_ppis = sn_ppis,
                                        source_label = source_label, target_label = target_label,
                                        stimulation_label = stimulation_label, inhibition_label = inhibition_label)


# Add the drug-target interactions:

# In[6]:


url = "https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/main/metadata/drug_metadata.parquet"
drug_ds = pd.read_parquet(url)

retained_drugs = obs.drug.unique()

moa_map = {'activator/agonist': 1, 
           'inhibitor/antagonist': -1}
drug_moa = dict(zip(drug_ds.drug, drug_ds['moa-broad']))
drug_moa = {k: moa_map[v] for k,v in drug_moa.items() if k in retained_drugs}
if len(drug_moa) != len(retained_drugs):
    raise ValueError('Missing drug moa')
    


# In[7]:


delim = '&'
moa = []
interactions_to_add = []
for drug, drug_targets in dict(zip(obs.drug, obs.drug_target)).items():
    for drug_target in drug_targets:
        interactions_to_add.append(delim.join([drug, drug_target]))
        moa.append(drug_moa[drug])
        
if len(interactions_to_add) != len(set(interactions_to_add)):
    raise ValueError('Non-unique interactions are present')


# In[8]:


sn_ppis = pn.add_omnipath_interaction(sn_ppis = sn_ppis,
                                      interactions_to_add = interactions_to_add,  
                                      moa = moa,
                                      delim = delim,
                                      source_label = source_label,
                                      target_label = target_label, 
                                      stimulation_label = stimulation_label, 
                                      inhibition_label = inhibition_label
                           )


# Filtering the network for interactions with atleast one reference and a curation effort of 2: 

# In[9]:


sn_ppis = pn.extract_network(sn_ppis = sn_ppis.copy(), 
                             curation_effort_thresh = 2, 
                             n_references_thresh = 1,
                             resources = 'all', 
                             drop_self = True, 
                             source_label = source_label, 
                             target_label = target_label,
                             verbose = False)


# Retain a signaling network of nodes with full paths from input drugs to the inferred TFs:

# In[10]:


ligand_labels = obs.drug.unique().tolist()
tf_labels = tf_adata.var.index.unique().tolist()


# In[11]:


ppi_nodes = set(sn_ppis[source_label].tolist() + sn_ppis[target_label].tolist())
n_tf_intersect = len(ppi_nodes.intersection(tf_labels))
print('{} of {} TFs are present in Omnipath upon applying filters'.format(n_tf_intersect, 
                                                                          tf_adata.shape[1], 
                                                                         ))


# In[12]:


sn_ppis, ligand_connections = pn.create_connected_network(sn_ppis = sn_ppis, 
                                                       ligand_labels = ligand_labels, 
                                                       tf_labels = tf_labels, 
                                                       source_label = source_label, 
                                                       target_label = target_label,
                                                       path_finder = 'shortest')

with open(os.path.join(data_path, 'processed', author + '_ligand_connections.json'), 'w') as f:
    json.dump(ligand_connections, f)


# In[13]:


ppi_nodes = set(sn_ppis[source_label].tolist() + sn_ppis[target_label].tolist())
retained_drugs = [drug for drug, TFs in ligand_connections.items() if len(TFs) != 0]
retained_TFs = ppi_nodes.intersection(tf_labels)
msg = 'The pruned network has {} interactions between {} nodes, '.format(sn_ppis.shape[0], len(ppi_nodes))
msg += 'retaining {} of {} TFs '.format(len(retained_TFs), n_tf_intersect)
msg += 'and {} of {} drugs (excluding control)'.format(len(retained_drugs), obs.drug.nunique())
print(msg)


# Format network as needed for input to model building:

# In[14]:


sn_ppis = pn.format_network(sn_ppis, weight_label, stimulation_label, inhibition_label) 


# In[15]:


sn_ppis[[source_label, target_label, weight_label, stimulation_label, inhibition_label]].head()


# In[16]:


sn_ppis.to_csv(os.path.join(data_path, 'processed', author + '_sn_ppis.csv'))


# # TF AnnData Parsing

# Filter the TF object to only include those samples / features retained in the signaling network:

# In[35]:


n_cells = tf_adata.shape[0]
retained_drugs.add('DMSO_TF')
tf_adata = tf_adata[tf_adata.obs.drug.isin(retained_drugs), sorted(retained_TFs)]
print('{} of {} cells are retained after excluding drugs not present in the signaling network'.format(tf_adata.shape[0], 
                                                                                                     n_cells))


# In[42]:


# url = 'https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/main/metadata/cell_line_metadata.parquet'
# cell_line_md = pd.read_parquet(url)
# cell_line_map = dict(zip(cell_line_md.Cell_ID_Cellosaur, cell_line_md.cell_name))


# In[49]:


obs = tf_adata.obs.copy()
n_cl, n_drug, n_cond = obs.cell_line.nunique(), obs.drug.nunique(), obs.condition.nunique()

print('The final {} conditions span {} cell lines and {} drugs'.format(n_cond, n_cl, n_drug))


# In[82]:


url = "https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/main/metadata/drug_metadata.parquet"
drug_ds = pd.read_parquet(url)
n_moa = drug_ds[drug_ds.drug.isin(retained_drugs)]['moa-fine'].nunique()

url = 'https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/main/metadata/cell_line_metadata.parquet'
cell_line_md = pd.read_parquet(url)
cell_line_md = cell_line_md[cell_line_md.Cell_ID_Cellosaur.isin(tf_adata.obs.cell_line.unique().tolist())]

print('This represents {} moas, {} organs, and {} driver genes'.format(n_moa, 
                                                         cell_line_md.Organ.nunique(), 
                                                         cell_line_md.Driver_Gene_Symbol.nunique()))
print('See notebook 00A for the starting numbers prior to signaling network filtering')


# Only one MOA is lost from the drug filtering.

# In[50]:


obs.cell_line.value_counts().unique()


# In[51]:


obs.drug.value_counts().unique()


# In[58]:


obs.condition.value_counts().unique()


# In[52]:


obs.shape[0]


# We choose a resolution for Leiden clustering that maximizes the NMI with the condition, emulating the methods from [here](https://doi.org/10.1038/s41467-024-49806-9). We proceed to use this resolution for all downstream analyses.

# In[54]:


pp.embed_tf_activity(tf_adata = tf_adata, 
                     scanpy_pca = False, 
                     n_components = 50, 
                     cluster_col_name = 'TF_clusters',
                     resolution = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0], 
                     nmi_label = 'condition', 
                     pc_rank = 'automate',
                     n_neighbors = 15,
                     run_pca = True,
                     run_umap  = True,
                     cluster_data = True)

# write with embeddings and signaling network filterings
tf_adata.obs.drug_target = tf_adata.obs.drug_target.astype(str) # needed for writing
io.write_tfad(tf_adata,
              os.path.join(data_path, 'processed', author + '_consensus_tf_activity.h5ad'))


# In[95]:


sc.pl.pca_variance_ratio(tf_adata)
n_pcs = tf_adata.uns['pca']['pca_rank']
print('The elbow was automatically identified at PC {}'.format(n_pcs))


# In[92]:


nmi = normalized_mutual_info_score(tf_adata.obs.condition, tf_adata.obs.TF_clusters)
print('Condition NMI was maximized at a resolution of {:.2f} at a value of {:.2f}'.format(tf_adata.uns['leiden']['params']['resolution'], 
                                                                                  nmi))


# In[115]:


# subset to 20% of the dataset
n_per_condition = int(np.round(tf_adata.obs['condition'].value_counts().min() * 0.2))

# Subsample indices evenly per condition
sampled_indices = (
    tf_adata.obs.groupby('condition')
    .sample(n=n_per_condition, random_state=seed)
    .index
)
# shufflt
np.random.seed(seed)
sampled_indices=np.random.permutation(sampled_indices)

viz_adata = tf_adata[sampled_indices, :].copy()


# In[122]:


fig, ax = plt.subplots(ncols = 3, figsize = (18, 5))

sc.pl.umap(viz_adata, color='TF_clusters', ax = ax[0], show = False)
ax[0].set_title('Clusters')
ax[0].legend().set_visible(False)
# ax[0].legend(bbox_to_anchor=(-0.1, 1), ncol = 5, bbox_transform=ax[0].transAxes)

sc.pl.umap(viz_adata, color='cell_line', ax = ax[1], show = False)
ax[1].set_title('Cell Lines')
ax[1].legend().set_visible(False)

sc.pl.umap(viz_adata, color='drug', ax = ax[2], show = False)
ax[2].set_title('Perturbation')
ax[2].legend().set_visible(False)
# ax[2].annotate('NMI (TF Clusters, Cell Type):\n {:.2f}'.format(nmi(tf_tf_adata.obs.TF_clusters, tf_tf_adata.obs.celltype)),
#                 xy = (0.015, 0.9), xycoords='axes fraction', fontsize = 9)

fig.tight_layout()
plt.subplots_adjust(hspace=0.1)
plt.show()


# In[144]:


pcs = tf_adata.obsm["X_pca"][:, :n_pcs]

res = []
for cov_ in ['drug', 'cell_line']:
    cov = tf_adata.obs[cov_].astype(str)
    enc = OneHotEncoder(drop="first", sparse_output=False)
    cov_encoded = enc.fit_transform(cov.values.reshape(-1, 1))

    r2_scores = []
    for pc_idx in trange(pcs.shape[1]):
        y = pcs[:, pc_idx]
        model = LinearRegression().fit(cov_encoded, y)
        r2 = model.score(cov_encoded, y)
        r2_scores.append({"PC": pc_idx + 1, "R2": r2})

    r2_df = pd.DataFrame(r2_scores)
    r2_df['covariate'] = cov_
    res.append(r2_df)
    
r2_df = pd.concat(res, axis=0, ignore_index=True)


# In[165]:


fig, ax = plt.subplots()

pval = stats.mannwhitneyu(r2_df[r2_df.covariate == 'drug'].R2, 
                  r2_df[r2_df.covariate == 'cell_line'].R2).pvalue
sns.boxplot(data = r2_df, x = 'covariate', y = 'R2', ax = ax)

ax.annotate(
    "MWU p-val: {:.2E}".format(pval),
    xy=(0.02, 0.98),
    xycoords="axes fraction",
    ha="left",
    va="top"
)

ax.set_ylabel('R2 (PC ~ Covariate)')
ax.set_xlabel('Covariate')
ax.set_title('Top {} PCs'.format(n_pcs))


# In[206]:


top_drug = r2_df[r2_df.covariate == 'drug'].sort_values(by = 'R2', ascending = False)
top_drug_pcs = top_drug.PC[:2].values.tolist()
top_drug


# In[207]:


top_cell_line = r2_df[r2_df.covariate == 'cell_line'].sort_values(by = 'R2', ascending = False)
top_cell_line_pcs = top_cell_line.PC[:2].values.tolist()
top_cell_line


# Visualize top 2 PCs that separate drugs:

# In[208]:


fig, ax = plt.subplots(ncols = 2, figsize = (10, 5))


# DRUG
X_pca = pd.DataFrame(viz_adata.obsm['X_pca'][:, :n_pcs], 
                     columns = range(1, n_pcs + 1))
X_pca = X_pca[top_drug_pcs]
X_pca['drug'] = viz_adata.obs.drug.values
X_pca.columns = X_pca.columns.astype(str)
top_drug_pcs = [str(i) for i in top_drug_pcs]

sns.scatterplot(data = X_pca, x = top_drug_pcs[0], y = top_drug_pcs[1], hue = 'drug', 
               s = 10, ax = ax[0])
ax[0].legend_.remove()
ax[0].set_title('Drug')

# Cell Line
X_pca = pd.DataFrame(viz_adata.obsm['X_pca'][:, :n_pcs], 
                     columns = range(1, n_pcs + 1))
X_pca = X_pca[top_cell_line_pcs]
X_pca['cell_line'] = viz_adata.obs.cell_line.values
X_pca.columns = X_pca.columns.astype(str)
top_cell_line_pcs = [str(i) for i in top_cell_line_pcs]

sns.scatterplot(data = X_pca, x = top_cell_line_pcs[0], y = top_cell_line_pcs[1], hue = 'cell_line', 
               s = 10, ax = ax[1])
ax[1].legend_.remove()
ax[1].set_title('Cell Line')

("")


# These are clearly separating much more strongly by cell line than perturbation. We will likely need a better way to separate by perturbation when visualizing.

#!/usr/bin/env python
# coding: utf-8

# # Download Data

# Next, let's get the actual relevant gene counts. 
# 
# - h5ads can be found [here](https://lamin.ai/laminlabs/arc-virtual-cell-atlas/artifacts?filter[and][0][or][0][branch.name][eq]=Main&filter[and][1][or][0][is_latest][eq]=true&filter[and][2][or][0][projects.name][eq]=Tahoe-100M&filter[and][3][or][0][otype][eq]=AnnData)
# - processing adapted from [here](https://theislab.github.io/vevo_Tahoe_100m_analysis/vevo_100m_pca.html):

# In[1]:


from multiprocessing import Pool

import os
import time
import re
from collections import defaultdict, Counter

import numpy as np
import scanpy as sc
import anndata as ad
import h5py

from scipy.sparse import csr_matrix
from scipy import sparse
from scipy.sparse import hstack

import pandas as pd
from tqdm import tqdm
import time

import sys
sclembas_path = '/home/hmbaghda/Projects/scLEMBAS'
sys.path.insert(1, os.path.join(sclembas_path))
from scLEMBAS.utilities import flatten_list
from scLEMBAS import io


# In[2]:


n_cores = 120
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)
# os.environ["DASK_DISTRIBUTED__WORKER__PRELOAD"] = ""

# SPARSE_CHUNK_SIZE = 100_000
# cluster = LocalCluster(n_workers=n_cores)
# client = Client(cluster)
# mod = sc


# In[3]:


data_path = '/nobackup/users/hmbaghda/scLEMBAS/analysis/'
author = 'Tahoe100M'


# In[4]:


url_name = 'https://storage.googleapis.com/arc-ctc-tahoe100/2025-02-25/h5ad/plate{}_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad'


obs_final = pd.read_csv(os.path.join(data_path, 'processed', author + '_filtered_cell_metadata.csv'), 
                            index_col = 0)
plate_iterable = sorted([int(plate_no.split('plate')[1]) for plate_no in sorted(obs_final.plate.unique())])


# Download raw data:

# In[5]:


# # download all at once (will be needed to write final adata to memory)
# # ensure 380GB of memory is available
# # cmd = 'seq {} {}'.format(list(plate_iterable)[0], list(plate_iterable)[-1])
# cmd = 'echo ' + ' '.join(plate_iterable)
# cmd += ' | xargs -n 1 -P {} '.format(min(7, len(plate_iterable)))
# cmd += '-I {} wget -O '
# cmd += os.path.join(data_path, 'raw', os.path.basename(url_name))
# cmd += ' ' + url_name
# os.system(cmd)


# Find the barcodes that we have retained:

# In[6]:


plate_iterable


# In[ ]:


adata_list = []
for plate_no in plate_iterable:
#     adata = get_tahoe_adata(plate_no)
    print('Start processing plate {}'.format(plate_no))

    url_name_iter = url_name.format(plate_no)
    fn = os.path.basename(url_name_iter)
    file_path = os.path.join(data_path, 'raw', fn)
    
    # download the data
    if not os.path.isfile(file_path):
        cmd = 'wget -O ' + file_path +  ' ' + url_name_iter
        os.system(cmd)
    
    
    # get the relevant barcode subset for this plate
    obs_final = pd.read_csv(os.path.join(data_path, 'processed', author + '_filtered_cell_metadata.csv'), 
                            index_col = 0)
    keep_barcodes = obs_final[obs_final.plate == 'plate{}'.format(plate_no)].index.tolist()

#     with h5py.File(file_path, "r") as f:
#         adata = ad.AnnData(
#             obs=ad.io.read_elem(f["obs"]),
#             var=ad.io.read_elem(f["var"]),
#         )
#         adata.X = ad.experimental.read_elem_as_dask(
#             f["X"], chunks=(SPARSE_CHUNK_SIZE, adata.shape[1])
#         )

    adata = ad.read_h5ad(file_path)
        
    barcodes = adata.obs.index.tolist()

    def check_membership(b):
        return b in keep_barcodes

    with Pool(processes=n_cores) as pool:
        mask = list(tqdm(pool.imap(check_membership, barcodes, chunksize=1000), total=len(barcodes)))

    adata = adata[mask, :].copy()

#     sc.pp.normalize_total(adata, target_sum = 1e6)
#     sc.pp.log1p(adata)
    # sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    
#     adata.write_h5ad(os.path.join(data_path, 'interim', 
#                                   author + '_plate{}'.format(plate_no) + '_filtered_normalized_counts.h5ad'))
    
    adata_list.append(adata)
    
#     # remove downloaded data
#     cmd = 'rm ' + file_path
#     os.system(cmd)


# Convert to memory and write:

# In[ ]:


io.write_pickled_object(adata_list, os.path.join(data_path, 'interim', 'trash.pickle'))

print('Merge adatas')
adata = ad.concat(adata_list, join = 'outer')


# Filter to coding genes:

# In[ ]:


protein_coding_genes = [g for g in adata.var_names if not g.startswith('ENSG')]

# filter out non-coding genes like various RNA types
noncoding_prefixes = (
    'SNORD', 'SNORA', 'Y_RNA', 'MIR', 'MIRLET', 'MIRHG', 'LINC',
    'RNU', 'SCARNA', 'TRNA', '7SK', '7SL', 'U[1-7]', 'SRP', 'VT',
    'XIST', 'TSIX', 'RN7', 'RN5', 'RNA5', 'RPP'
)
pattern = re.compile(rf"^({'|'.join(noncoding_prefixes)})")
protein_coding_genes = [g for g in protein_coding_genes if not pattern.match(g)]


def is_non_coding_or_pseudogene(gene):
    return any([
        gene.startswith(prefix) for prefix in (
            'Metazoa_SRP', 'Vault-', 'RNVU', 'U8-', '5S_rRNA', '5_8S_rRNA', 'hsa-mir-', 'mir-', 'RNA', 'RNU'
        )
    ]) or any([
        re.search(p, gene) for p in (
            r'-AS\d*$',     # antisense
            r'-DT$',        # divergent transcript
            r'-OT\d*$',     # overlapping transcript
            r'^ERV',        # endogenous retrovirus
            r'^IG.*OR',     # immunoglobulin open reading frame
            r'^TR.*OR',     # TCR open reading frame
            r'^IGLV.*OR',   # light chain ORs
            r'^IGHV.*OR',   # heavy chain ORs
            r'^IGKV.*OR',   # kappa chain ORs
            r'ORF',         # any gene with ORF in the name
            r'P\d+$',       # pseudogenes (e.g., *P1)
            r'P$',          # ends in 'P'
            r'P\d+-\d+$'    # pseudogenes with multiple digits
        )
    ])
protein_coding_genes = [g for g in protein_coding_genes if not is_non_coding_or_pseudogene(g)]


def is_known_pseudogene_or_low_confidence(gene):
    return any([
        'PSEUDO' in gene.upper(),                         # labeled explicitly (rare)
        gene in {'FAM86DP-1', 'GOLGA2P2Y-1', 'FAM95B1-1'}  # known noncoding examples
    ])

protein_coding_genes = [g for g in protein_coding_genes if not is_known_pseudogene_or_low_confidence(g)]

adata = adata[:, protein_coding_genes]


# Some genes appear to be mapping duplicates, we will aggregate by the sum:

# In[ ]:


base_name_map = defaultdict(list)
for gene in protein_coding_genes:
    base = re.sub(r'-\d+$', '', gene)
    base_name_map[base].append(gene)

duplicate_artifact_candidates = {base: names for base, names in base_name_map.items() if len(names) > 1}

mapping_artifacts = {
    base: variants for base, variants in duplicate_artifact_candidates.items()
    if any(v == base for v in variants) and len(variants) == 2
}

duplicated = set(flatten_list(mapping_artifacts.values()))

expr_df = adata.to_df()

aggregated_expr = pd.DataFrame(index=adata.obs_names)
for base, variants in mapping_artifacts.items():
    aggregated_expr[base] = expr_df[variants].sum(axis = 1)
    
expr_df = expr_df.drop(columns=duplicated)
for col in aggregated_expr.columns:
    expr_df[col] = aggregated_expr[col].values


# In[ ]:


print('Create filtered AnnData Object')
X_sparse = csr_matrix(expr_df.values)
adata_new = ad.AnnData(X=X_sparse)
adata_new.obs = adata.obs.copy()
adata_new.var = pd.DataFrame(index=expr_df.columns)


# Filter out ultra-rare genes:

# In[ ]:


sc.pp.filter_genes(adata_new, min_cells = 100, inplace = True) # filter out ultra-rare genes


# Add any metadata from 00A:

# In[ ]:


obs_final = pd.read_csv(os.path.join(data_path, 'processed', author + '_filtered_cell_metadata.csv'), 
                            index_col = 0)
obs_final = obs_final.loc[adata_new.obs_names, :].copy()

for col in obs_final.columns:
    if col in adata_new.obs.columns:
        if np.any(obs_final[col].values != adata_new.obs[col].values):
            if not np.allclose(obs_final[col].values, adata_new.obs[col].values):
                print(col)
                raise ValueError('Unexpected difference in metadata')
    else:
        adata_new.obs[col] = obs_final[col].values
    


# In[ ]:


print('Write file')
adata_new.write_h5ad(os.path.join(data_path, 'interim', author + '_filtered_counts.h5ad'))


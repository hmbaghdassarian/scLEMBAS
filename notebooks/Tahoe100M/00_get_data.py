#!/usr/bin/env python
# coding: utf-8


# # Download Data

# Next, let's get the actual relevant gene counts (adapted from [here](https://theislab.github.io/vevo_Tahoe_100m_analysis/vevo_100m_pca.html)):

# In[ ]:


import time

from scipy.sparse import csr_matrix
import anndata

import scanpy as sc
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

import os

data_path = '/nobackup/users/hmbaghda/scLEMBAS/analysis/'
author = 'Tahoe100M'


# In[ ]:


# if obs.BARCODE_SUB_LIB_ID.nunique() != obs.shape[0]:
#     raise ValueError('Expected unique barcode per cell')
# else:
#     obs.set_index('BARCODE_SUB_LIB_ID', inplace = True)
    
# obs_final = obs.copy()
# obs_final.to_csv(os.path.join(data_path, 'processed', '_filtered_cell_metadata.csv'))
obs_final = pd.read_csv(os.path.join(data_path, 'processed', '_filtered_cell_metadata.csv'), 
                       index_col = 0)

MY_BARCODES = obs_final.index.tolist()
n_total = 100648790
n_filtered = len(MY_BARCODES)


# In[164]:


tahoe_100m_ds = load_dataset('vevotx/Tahoe-100M', streaming=True, split='train')
# filtered_ds = tahoe_100m_ds.filter(lambda row: row["BARCODE_SUB_LIB_ID"] in MY_BARCODES)


# In[165]:


gene_metadata = load_dataset("vevotx/Tahoe-100M", name="gene_metadata", split="train")
gene_vocab = {entry["token_id"]: entry["gene_symbol"] for entry in gene_metadata}
if len(gene_vocab) != len(set(gene_vocab.values())):
    raise ValueError('Expected a 1-to-1 gene ID mapping')


# In[150]:


def create_anndata_from_generator(generator, gene_vocab, sample_size=None):
    sorted_vocab_items = sorted(gene_vocab.items())
    token_ids, gene_names = zip(*sorted_vocab_items)
    token_id_to_col_idx = {token_id: idx for idx, token_id in enumerate(token_ids)}

    data, indices, indptr = [], [], [0]
    obs_data = []

    match_count = 0
    for i, cell in tqdm(enumerate(generator), total=n_total, desc="Scanning cells"):
#         if sample_size is not None and i >= sample_size:
#             break

        if cell["BARCODE_SUB_LIB_ID"] in MY_BARCODES:
            match_count += 1

            genes = cell['genes']
            expressions = cell['expressions']
            if expressions[0] < 0: 
                genes = genes[1:]
                expressions = expressions[1:]

            col_indices = [token_id_to_col_idx[gene] for gene in genes if gene in token_id_to_col_idx]
            valid_expressions = [expr for gene, expr in zip(genes, expressions) if gene in token_id_to_col_idx]

            data.extend(valid_expressions)
            indices.extend(col_indices)
            indptr.append(len(data))

            obs_entry = {k: v for k, v in cell.items() if k not in ['genes', 'expressions']}
            obs_data.append(obs_entry)
            
            if match_count == len(MY_BARCODES):
                break

        if i % 1e5 == 0:
            print('The match count at cell {} is {} of {}'.format(i, match_count, n_filtered))

    expr_matrix = csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, len(gene_names)))
    obs_df = pd.DataFrame(obs_data)

    adata = anndata.AnnData(X=expr_matrix, obs=obs_df)
    adata.var.index = pd.Index(gene_names, name='gene_symbol')

    return adata

adata = create_anndata_from_generator(tahoe_100m_ds, gene_vocab, sample_size=None)
adata.write_h5ad(os.path.join(data_path, 'raw', author + '_filtered_raw_counts.h5ad'))


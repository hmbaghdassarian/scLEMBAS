#!/usr/bin/env python
# coding: utf-8

# # Download Data

# Next, let's get the actual relevant gene counts. 
# 
# - h5ads can be founder [here](https://lamin.ai/laminlabs/arc-virtual-cell-atlas/artifacts?filter[and][0][or][0][branch.name][eq]=Main&filter[and][1][or][0][is_latest][eq]=true&filter[and][2][or][0][projects.name][eq]=Tahoe-100M&filter[and][3][or][0][otype][eq]=AnnData)
# - processing adapted from [here](https://theislab.github.io/vevo_Tahoe_100m_analysis/vevo_100m_pca.html)):

# In[1]:


from pathlib import Path
from multiprocessing import Pool

import numpy as np
import dask.distributed as dd
import scanpy as sc
import anndata as ad
import h5py
import dask

from collections import Counter
import pandas as pd
from tqdm import tqdm
import dask
import time
from dask.distributed import Client, LocalCluster

import dask


import os


# In[2]:


n_cores = 80
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)
os.environ["DASK_DISTRIBUTED__WORKER__PRELOAD"] = ""


SPARSE_CHUNK_SIZE = 100_000
cluster = LocalCluster(n_workers=n_cores)
client = Client(cluster)
mod = sc


# In[3]:


data_path = '/nobackup/users/hmbaghda/scLEMBAS/analysis/'
author = 'Tahoe100M'


# In[4]:


# obs_final = pd.read_csv(os.path.join(data_path, 'processed', '_filtered_cell_metadata.csv'), 
#                        index_col = 0)


# In[6]:


url_name = 'https://storage.googleapis.com/arc-ctc-tahoe100/2025-02-25/h5ad/plate{}_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad'


def get_tahoe_adata(plate_no):
    """Pulls the anndata object with the subset of interest for a given plate number."""
    
    print('Start processing plate {}'.format(plate_no))
    # download the data
    url_name_iter = url_name.format(plate_no)
    fn = os.path.basename(url_name_iter)
    file_path = os.path.join(data_path, 'raw', fn)

    cmd = 'wget -O ' + file_path +  ' ' + url_name_iter
    os.system(cmd)

    # get the relevant barcode subset for this plate
    obs_final = pd.read_csv(os.path.join(data_path, 'processed', '_filtered_cell_metadata.csv'), 
                            index_col = 0)
    keep_barcodes = obs_final[obs_final.plate == 'plate{}'.format(plate_no)].index.tolist()

    with h5py.File(file_path, "r") as f:
        adata = ad.AnnData(
            obs=ad.io.read_elem(f["obs"]),
            var=ad.io.read_elem(f["var"]),
        )
        adata.X = ad.experimental.read_elem_as_dask(
            f["X"], chunks=(SPARSE_CHUNK_SIZE, adata.shape[1])
        )
        
    barcodes = adata.obs.index.tolist()

    def check_membership(b):
        return b in keep_barcodes

    with Pool(processes=n_cores) as pool:
        mask = list(tqdm(pool.imap(check_membership, barcodes, chunksize=1000), total=len(barcodes)))

    adata = adata[mask, :].copy()

    sc.pp.normalize_total(adata, target_sum = 1e6)
    sc.pp.log1p(adata)
    # sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    
    # remove downloaded data
    cmd = 'rm ' + file_path
    os.system(cmd)
    
    return adata

adata_list = []
plate_iterable = range(1,15)
for plate_no in plate_iterable:
    adata = get_tahoe_adata(plate_no)
    adata_list.append(adata)

print('Merge objects')
adata = sc.concat(adata_list, 
                           join='outer')
#                            label='filtered_plate', 
#                            keys=list(plate_iterable))
print('Write filtered object')
adata.write_h5ad(os.path.join(data_path, 'interim', author + '_filtered_normalized_counts.h5ad'))


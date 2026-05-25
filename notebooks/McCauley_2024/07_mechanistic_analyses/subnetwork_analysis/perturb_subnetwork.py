#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import sys

import pandas as pd
import numpy as np

import scanpy as sc
import scipy.sparse as sp
import anndata as ad
from scipy import stats

import decoupler as dc
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.default_inference import DefaultInference

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from adjustText import adjust_text   

sys.path.insert(1, '../../.')
from McCauley_utils import initialize_mod_and_trainer, all_data


# In[9]:


seed = 888
data_path = '/home/hmbaghda/orcd/pool/scLEMBAS/analysis'
author = 'McCauley'

n_cores = 30
os.environ["OMP_NUM_THREADS"] = str(1)
os.environ["MKL_NUM_THREADS"] = str(1)
os.environ["OPENBLAS_NUM_THREADS"] = str(1)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(1)
os.environ["NUMEXPR_NUM_THREADS"] = str(1)


# In[10]:


(sn_ppis, tf_adata, adata, expr, source_label, target_label, weight_label, 
 stimulation_label, inhibition_label, cat_col, pert_col, ctrl_pert) = all_data


# Based on (1) perturbation of interest (TGFB1) (2) cell types of interest (Basal, Club) and (3) the scLEMBAS model fit, we define a subnetwork consisting of the most relevant nodes.

# In[11]:


subnetwork = pd.read_csv(
    os.path.join(data_path, 'processed', 'consensus_subnetwork_edges_TGFB1^BasalClub.csv')
)



# In[2]:


n_nodes = len(set(sn_ppis[source_label]).union(set(sn_ppis[target_label])))
subnetwork_nodes = sorted(set(subnetwork['source']).union(set(subnetwork['target'])))
n_subnetwork_nodes = len(subnetwork_nodes)


# Let's look at whether the identified subnetwork nodes show expression differences between cell types. We download the HLCA core data from HCA [here](https://data.humancellatlas.org/hca-bio-networks/lung/atlases/lung-v1-0)

# In[1]:


scores = pd.read_csv(os.path.join(data_path, 'processed', '{}_candidate_catbias_perts.csv'.format(author)), 
                    index_col = 0)


# In[100]:


import copy
import joblib
import torch 
from tqdm import tqdm
from scLEMBAS.predict import get_prediction
from scLEMBAS import utilities as utils
import joblib
import itertools

sys.path.insert(1, '../../../.') 
from notebook_utils import get_split


# In[58]:


n_ensembles = 10
seed_multiplier = 21234


# In[59]:


pert = 'TGFB1'
cell_types = ['Basal', 'Club']
ct_rev_map = dict(zip(cell_types, cell_types[::-1]))

pls_fn_label = '_'.join(cell_types) + '^' + pert
pls_model = joblib.load(os.path.join(
    data_path, 'processed', 'PLS_subnetwork_{}.joblib'.format(pls_fn_label)))



# In[90]:


def load_model(fold, ensemble_idx, from_trainer = False):
    """Loads the model and training object.

    Two different ways to do so: from pickled training object (larger files) or from model state dict `.pt` file (smaller files to transfer).

    Parameters
    ----------
    fold : int
        fold split
    ensemble_idx : int
        ensemble index
    from_trainer : bool, optional
        whether to load from trainer object or model state dict, by default False
        if False, the training object is not returned
    """
    fn_base = os.path.join(data_path, 'processed', '{}_fold{}'.format(author, fold))
    if from_trainer:
        if ensemble_idx < n_ensembles - 1:
            fn_trainer =  os.path.join(fn_base + 'trainer_actual_ensemble{}.pickle'.format(ensemble_idx))
        else:
            fn_trainer = os.path.join(fn_base + 'trainer_actual.pickle')


        trainer = io.read_pickled_object(fn_trainer)
        mod = trainer.mod
    else:
        seed_ = seed + ensemble_idx + 1 + (seed_multiplier * ensemble_idx * fold) if ensemble_idx <= 3 else seed
        mod, trainer = initialize_mod_and_trainer(
            fold = fold,
            adversarial_penalty = True,
            randomize = False,
            seed = seed_
        )

        if ensemble_idx < n_ensembles - 1: # +1 of the originally trained model
            fn_mod = os.path.join(fn_base + 'model_actual_ensemble{}.pt'.format(ensemble_idx))
        else:
            fn_mod = os.path.join(fn_base + 'model_actual.pt')

        mod.load_state_dict(torch.load(fn_mod))
        trainer = None
    return mod, trainer

import functools, contextlib, os, warnings
from tqdm import tqdm


class suppress_tqdm:
    def __enter__(self):
        # silence prints (stdout)
        self._devnull = open(os.devnull, "w")
        self._redirect = contextlib.redirect_stdout(self._devnull)
        self._redirect.__enter__()

        # silence warnings
        self._warn_ctx = warnings.catch_warnings()
        self._warn_ctx.__enter__()
        warnings.simplefilter("ignore")   # or filter to specific categories

        # silence tqdm bars
        self._orig_init = tqdm.__init__
        tqdm.__init__ = functools.partialmethod(tqdm.__init__, disable=True)
        return self

    def __exit__(self, *args):
        tqdm.__init__ = self._orig_init
        self._warn_ctx.__exit__(*args)
        self._redirect.__exit__(*args)
        self._devnull.close()


def get_barcodes(ct, split):
    cond = ct + '^' + pert
    ctrl_cond = ct + '^' + ctrl_pert
    # use counterfactual either way to make comparisons as similar as possible
    #prediction_type = 'train_counterfactual' if cond in split['train_conds'] else 'test_counterfactual'
    prediction_type = 'train_nocounterfactual' if cond in split['train_conds'] else 'test_counterfactual'

    # subset train/test to condition of interest
    train_barcodes = split['train_barcodes']
    test_barcodes = split['test_barcodes']
    train_cond_mask = tf_adata[train_barcodes, :].obs.condition.isin([cond, ctrl_cond])
    test_cond_mask = tf_adata[test_barcodes, :].obs.condition.isin([cond, ctrl_cond])
    train_barcodes = train_cond_mask[train_cond_mask].index.tolist()
    test_barcodes = test_cond_mask[test_cond_mask].index.tolist()

    strg = {'train': train_barcodes, 'test': test_barcodes, 'prediction_type': prediction_type}

    return strg


def get_prediction_subnetwork(prediction_type, mod, train_barcodes, test_barcodes):
    with suppress_tqdm():
        if prediction_type == 'test_counterfactual':
            tf_adata_predicted = get_prediction(
                mod = mod,
                train_cells = train_barcodes,
                test_cells = test_barcodes,
                tf_adata = tf_adata,
                cat_col = cat_col,
                pert_col = pert_col,
                ctrl_pert = ctrl_pert,
                counterfactual = 'perturbation', # counterfactual from tests
                cat_counterfactual_map = None,
                remove_type = 'none',
                return_bias = False,
                max_cells = int(5e3),
                return_full = False,
                stim_label_map = None, # special use case for Kang
                check_forward = False,
            )
        elif prediction_type == 'train_nocounterfactual':
            tf_adata_predicted = get_prediction(
                mod = mod,
                train_cells = train_barcodes,
                test_cells = [],
                tf_adata = tf_adata,
                cat_col = cat_col,
                pert_col = pert_col,
                ctrl_pert = ctrl_pert,
                counterfactual = None,
                cat_counterfactual_map = None,
                remove_type = 'none',
                return_bias = False,
                max_cells = int(5e3),
                return_full = False,
                stim_label_map = None, # special use case for Kang
                 check_forward = False,

            )

        elif prediction_type == 'train_counterfactual':
            ctrl_mask = tf_adata[train_barcodes, :].obs[pert_col] == ctrl_pert
            train_barcodes_ctrl = pd.Series(train_barcodes)[ctrl_mask.tolist()]
            train_barcodes_pert = pd.Series(train_barcodes)[(~ctrl_mask).tolist()]

            tf_adata_predicted = get_prediction(
                mod = mod,
                train_cells = train_barcodes_ctrl,
                test_cells = train_barcodes_pert,
                tf_adata = tf_adata,
                cat_col = cat_col,
                pert_col = pert_col,
                ctrl_pert = ctrl_pert,
                counterfactual = 'perturbation',
                cat_counterfactual_map = None,
                remove_type = 'none',
                return_bias = False,
                max_cells = int(5e3),
                return_full = False,
                stim_label_map = None, # special use case for Kang
                check_forward = False,
            )

        return tf_adata_predicted



# In[91]:


mask_global_bias = False

node_scalers = [1, 10, 100, int(1e3), int(1e4)]
top_ns = [5, 10, 25, 40]



# In[93]:


all_predictions = []
n_iter = 5 * n_ensembles * len(cell_types) * len(node_scalers) * len(top_ns)

with tqdm(total=n_iter, desc="predictions") as pbar:
    for fold in range(5):
        split = get_split(fold, author)

        ct_barcode_map = {}
        for ct in cell_types:
            ct_barcode_map[ct] = get_barcodes(ct, split)

        for ensemble_idx in range(n_ensembles): # REPLACE
            mod, _ = load_model(fold = fold, ensemble_idx = ensemble_idx, from_trainer = False)

            for ct in cell_types:

#                 try:
                prediction_type = ct_barcode_map[ct]['prediction_type']
                if prediction_type != 'train_nocounterfactual': # not interested in test performance, just best performance
                    continue

                tf_adata_predicted = get_prediction_subnetwork(
                    prediction_type = prediction_type, 
                    mod = mod, 
                    train_barcodes = ct_barcode_map[ct]['train'],
                    test_barcodes = ct_barcode_map[ct]['test']
                )
                tf_adata_predicted.obs['alteration'] = False
                tf_adata_predicted.obs['fold'] = fold
                tf_adata_predicted.obs['ensemble_idx'] = ensemble_idx
                tf_adata_predicted.obs['prediction_type'] = prediction_type
                tf_adata_predicted.obs['node_scaler'] = np.nan
                tf_adata_predicted.obs['top_n_altered'] = np.nan

                # don't retain control predictions for memory purposes (won't be used)
                pert_mask = tf_adata_predicted.obs[pert_col] == pert
                tf_adata_predicted = tf_adata_predicted[pert_mask,:].copy()

                all_predictions.append(tf_adata_predicted)


                # get the alteration: for a cell type, make it look like the other cell type
                for node_scaler, top_n in itertools.product(node_scalers, top_ns):
                    pbar.set_postfix(fold=fold, ens=ensemble_idx, ct=ct, ns=node_scaler, top_n=top_n)
                    pbar.update(1)


                    alter_nodes = scores.index.tolist()[:top_n]
                    mod_alt = copy.deepcopy(mod)

                    from_ = ct
                    to_ = ct_rev_map[ct]

                    from_cat_idx = mod.signaling_network.cat_mapper[cat_col][from_]
                    to_cat_idx = mod.signaling_network.cat_mapper[cat_col][to_]

                    altered_nodes_idx = [mod.node_idx_map[node] for node in alter_nodes]

                    embedding = mod_alt.signaling_network.cat_embeddings[cat_col]
                    to_values = embedding.weight[to_cat_idx, altered_nodes_idx].clone()
                    to_values *= node_scaler

                    with torch.no_grad():
                        embedding.weight[from_cat_idx, altered_nodes_idx] = to_values
                        if mask_global_bias:
                            assert torch.all(~mod_alt.signaling_network.bias_mask[altered_nodes_idx]), 'Not all nodes are not masked'
                            mod_alt.signaling_network.bias_mask[altered_nodes_idx] = True


                    tf_adata_predicted_alt = get_prediction_subnetwork(
                        prediction_type = prediction_type, 
                        mod = mod_alt, 
                        train_barcodes = ct_barcode_map[ct]['train'],
                        test_barcodes = ct_barcode_map[ct]['test']
                    )
                    del mod_alt
                    utils.clear_memory()
                    tf_adata_predicted_alt.obs['alteration'] = True
                    tf_adata_predicted_alt.obs['fold'] = fold
                    tf_adata_predicted_alt.obs['ensemble_idx'] = ensemble_idx
                    tf_adata_predicted_alt.obs['prediction_type'] = prediction_type
                    tf_adata_predicted_alt.obs['node_scaler'] = node_scaler
                    tf_adata_predicted_alt.obs['top_n_altered'] = top_n

                    # don't retain control predictions for memory purposes (won't be used)
                    pert_mask = tf_adata_predicted_alt.obs[pert_col] == pert
                    tf_adata_predicted_alt = tf_adata_predicted_alt[pert_mask,:].copy()


                    all_predictions.append(tf_adata_predicted_alt)

            del mod
            utils.clear_memory()



tf_adata_predicted = sc.concat(
    all_predictions,
    join="outer"
)
del all_predictions       

fa = (author, pert, cell_types[0], cell_types[1])
tf_adata_predicted.write_h5ad(
    os.path.join(data_path, 'processed', '{}_subnetwork_insilico_alt_{}_{}_{}.h5ad'.format(*fa))
)


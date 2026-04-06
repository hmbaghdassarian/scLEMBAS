#!/usr/bin/env python
# coding: utf-8

# Here, we train scLEMBAS on the same split as Notebook 05B, but with different seeds to create an ensemble of learned models. We create an additional 4 per split in the 5-fold CV, giving a total of 25 models (5 of those originally trained in Notebook 05B and 20 from this split).

# In[1]:


import os

import numpy as np

import torch 

import sys
sclembas = '/home/hmbaghda/Projects/scLEMBAS'
sys.path.insert(1, os.path.join(sclembas))
from scLEMBAS import io
import scLEMBAS.utilities as utils

sys.path.insert(1, '../../.')
from McCauley_utils import initialize_mod_and_trainer, all_data

sys.path.insert(1, '../../../.') 
from notebook_utils import get_split


# In[2]:


seed = 888
data_path = '/home/hmbaghda/orcd/pool/scLEMBAS/analysis'
author = 'McCauley'

n_cores = 30
os.environ["OMP_NUM_THREADS"] = str(1)
os.environ["MKL_NUM_THREADS"] = str(1)
os.environ["OPENBLAS_NUM_THREADS"] = str(1)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(1)
os.environ["NUMEXPR_NUM_THREADS"] = str(1)


# In[3]:


(sn_ppis, tf_adata, adata, expr, source_label, target_label, weight_label, 
 stimulation_label, inhibition_label, cat_col, pert_col, ctrl_pert) = all_data

# 1% of real
num_stochastic_edges = int(np.round(0.01*sn_ppis.shape[0]))


n_ensembles_per_seed = 5
seed_multiplier = 21234

fold = 2
# -------------------- SINGLE-CELL MODELS --------------------
import itertools
bn_weights_l1s = [
    1e-7, 1e-5]
bn_weight_l2 = 0

iterations_ = itertools.product(range(n_ensembles_per_seed), bn_weights_l1s)
for (ensemble_idx, bn_weight_l1) in iterations_:
    l1_string = '{:.0E}'.format(bn_weight_l1).replace('E-0', 'E-')
    fn_base = os.path.join(data_path, 'processed', 'pruning_ensembles', '{}_fold{}_l20_l1{}'.format(author, fold, l1_string))

    def write_model(mod, trainer, ensemble_idx):
        io.write_pickled_object(trainer, fn_base + '_pruning_trainer_actual_ensemble{}.pickle'.format(ensemble_idx))
        torch.save(mod.state_dict(), fn_base + '_pruning_model_actual_ensemble{}.pt'.format(ensemble_idx))
        del mod, trainer
        utils.clear_memory()

    for ensemble_idx in range(n_ensembles_per_seed):
        curr_seed = seed + ensemble_idx + 1 + (seed_multiplier * ensemble_idx * fold)        
        if os.path.isfile(fn_base + '_pruning_model_actual_ensemble{}.pt'.format(ensemble_idx)):
            continue
        mod, trainer = initialize_mod_and_trainer(
            fold = fold, 
            adversarial_penalty = True, 
            randomize = False, 
            num_stochastic_edges = num_stochastic_edges, 
            bn_weights_lambda_L2 = bn_weight_l2, 
            bn_weights_lambda_L1 = bn_weight_l1, # 0
            seed = curr_seed)
        mod = trainer.train_model(verbose = False)
        write_model(mod, trainer, ensemble_idx = ensemble_idx)
#!/usr/bin/env python
# coding: utf-8

# Here, we train scLEMBAS on the same split as Notebook 05B, but with different seeds to create an ensemble of learned models. We create an additional 4 per split in the 5-fold CV, giving a total of 25 models (5 of those originally trained in Notebook 05B and 20 from this split).

# In[2]:


import os

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


# In[3]:


seed = 888
data_path = '/home/hmbaghda/orcd/pool/scLEMBAS/analysis'
author = 'McCauley'

n_cores = 30
os.environ["OMP_NUM_THREADS"] = str(1)
os.environ["MKL_NUM_THREADS"] = str(1)
os.environ["OPENBLAS_NUM_THREADS"] = str(1)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(1)
os.environ["NUMEXPR_NUM_THREADS"] = str(1)


# In[4]:


(sn_ppis, tf_adata, adata, expr, source_label, target_label, weight_label, 
 stimulation_label, inhibition_label, cat_col, pert_col, ctrl_pert) = all_data


# In[ ]:


n_ensembles_per_seed = 9#4 # because one exists already 
seed_multiplier = 21234

fold = 3
# -------------------- SINGLE-CELL MODELS --------------------
fn_base = os.path.join(data_path, 'processed', '{}_fold{}'.format(author, fold))

def write_model(mod, trainer, ensemble_idx):
    io.write_pickled_object(trainer, fn_base + 'trainer_actual_ensemble{}.pickle'.format(ensemble_idx))
    torch.save(mod.state_dict(), fn_base + 'model_actual_ensemble{}.pt'.format(ensemble_idx))
    del mod, trainer
    utils.clear_memory()

for ensemble_idx in range(n_ensembles_per_seed):
    curr_seed = seed + ensemble_idx + 1 + (seed_multiplier * ensemble_idx * fold)        
    if os.path.isfile(fn_base + 'model_actual_ensemble{}.pt'.format(ensemble_idx)):
        continue
    mod, trainer = initialize_mod_and_trainer(
        fold = fold, 
        adversarial_penalty = True, 
        randomize = False, 
        seed = curr_seed)
    mod = trainer.train_model(verbose = False)
    write_model(mod, trainer, ensemble_idx = ensemble_idx)


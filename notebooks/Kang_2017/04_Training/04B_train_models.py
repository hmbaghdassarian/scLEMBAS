#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--fold", type=int, required=True)
args = parser.parse_args()
fold = args.fold


# We will train the scLEMBAS model alongside 4 baselines:
# 
# 1) random baseline: trained on permuted features
# 2) no adversarial baseline: global bias does not receive an adversarial penalty for containing categorical / perturbation information
# 3) linear baseline:
# 4) fully connected neural network:
# 
# In assessment, we will also use the training mean as a baseline.

# In[1]:


import os

import torch

import sys
sclembas = '/home/hmbaghda/Projects/scLEMBAS'
sys.path.insert(1, os.path.join(sclembas))
from scLEMBAS import io
import scLEMBAS.utilities as utils

sys.path.insert(1, '../.')
from Kang_utils import initialize_mod_and_trainer, all_data

data_path = '/home/hmbaghda/orcd/pool/scLEMBAS/analysis'
author = 'Kang'
seed = 888


# In[2]:


(sn_ppis, tf_adata, adata, expr, source_label, target_label, weight_label, 
 stimulation_label, inhibition_label, cat_col, pert_col) = all_data


# # To do:
# - create baselins 3 and 4

# In[3]:


mod_types = {
    'actual': [True, False],
    'random': [True, True],
    'noadv': [False, False]
}


# In[ ]:


# fold = 0
# for fold in range(5)


# In[ ]:


fn_base = os.path.join(data_path, 'processed', '{}_fold{}'.format(author, fold))

def write_model(mod, trainer, mod_type: str):
    io.write_pickled_object(trainer, fn_base + 'trainer_{}.pickle'.format(mod_type))
    torch.save(mod.state_dict(), fn_base + 'model_{}.pt'.format(mod_type))
    del mod, trainer
    utils.clear_memory()

# Actual Model, Random Baseline, and No Adversarial Removal Baseline
for mod_type, mod_args in mod_types.items():
    if not os.path.isfile(fn_base + 'trainer_{}.pickle'.format(mod_type)):
        mod, trainer = initialize_mod_and_trainer(
            fold = fold, 
            adversarial_penalty = mod_args[0], 
            randomize = mod_args[1], 
            seed = seed)
        mod = trainer.train_model(verbose = False)
        write_model(mod, trainer, mod_type = mod_type)


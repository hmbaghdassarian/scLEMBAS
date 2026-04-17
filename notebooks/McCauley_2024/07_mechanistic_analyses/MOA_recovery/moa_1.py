#!/usr/bin/env python
# coding: utf-8

# Here, we train scLEMBAS on the same split as Notebook 05B, but with a subset of the known MOAs removed to test for this. We repeat this 5 times per fold, resulting in an ensemble of 25 models. 

# In[1]:


import os
import pandas as pd

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

sn_ppis_all = sn_ppis.copy()
sn_ppis_all['interaction_'] = sn_ppis_all[[source_label, target_label]].agg('^'.join, axis=1).tolist()  


# Define frequency with which to remove MOAs: 1% of known MOAs removed per iteration, matching the frequency of signs:

# In[4]:


subset_frac = 0.001

known_moa_mask = sn_ppis[weight_label].isin([1, -1])
known_moa = sn_ppis[known_moa_mask].copy()
sign_frequency = known_moa[weight_label].value_counts(normalize = True)

n_moa_remove = int(len(known_moa) * subset_frac)
n_per_class = (sign_frequency * n_moa_remove).round().astype(int)

print('{} MOAs will be removed per model, representing {:.1f}% of total known MOAs in the PKN'.format(n_moa_remove, subset_frac*100))


# In[5]:


def write_model(mod, trainer, moa_removed, fn_base, save_trainer = False):
    # model and trainer
    if save_trainer:
        io.write_pickled_object(trainer, fn_base + '_pruning_trainer_actual.pickle')
    torch.save(mod.state_dict(), fn_base + '_moa_model_actual.pt')

    # stats
    trainer.stats['train'].to_csv(fn_base + '_moa_trainstats_actual.csv')

    train_eval_df = trainer.stats['train_eval'].copy()
    test_df = trainer.stats['test'].copy()
    train_eval_df['loss_type'] = 'Train (Evaluation)'
    test_df['loss_type'] = 'Test'
    eval_df = pd.concat([train_eval_df, test_df], axis = 0)
    eval_df.reset_index(drop = True, inplace = True)
    eval_df.loss_type = pd.Categorical(eval_df.loss_type, ordered = True, 
                                      categories = ['Train (Evaluation)', 'Test'])
    eval_df.to_csv(fn_base + '_moa_evalstats_actual.csv')

    del mod, trainer
    utils.clear_memory()


    with open(fn_base + '_moa_removededges.txt', "w") as f:
        for item in moa_removed:
            f.write(f"{item}\n")


# In[5]:
fold = 1

n_ensembles_per_seed = 5
seed_multiplier = 21234


# for fold in range(5):

for ensemble_idx in range(n_ensembles_per_seed):
    frac_str = "{:d}".format(int(round(subset_frac * 1000)))  # 3 decimal precision
    fn_label = "{}_fold{}_ensemble{}_frac{}".format(author, fold, ensemble_idx, frac_str)
    fn_base = os.path.join(data_path, 'processed', 'moa_ensembles', fn_label)
    if os.path.isfile(fn_base + '_moa_removededges.txt'):
        continue

    curr_seed = seed + ensemble_idx + 1 + (seed_multiplier * ensemble_idx * fold) 


    ######################### REMOVE MOAs #########################
    # at frequency of population distribution

    # remove the 1% subset of moas
    moa_removed = (
        known_moa
        .groupby(weight_label, group_keys=False)
        .apply(
            lambda x: x.sample(n=n_per_class.loc[x.name], random_state=curr_seed),
            include_groups=False
        )
    )
    # known_moa.loc[moa_removed.index, weight_label].value_counts(normalize = True)
    moa_removed = moa_removed[[source_label, target_label]].agg('^'.join, axis=1).tolist()


    ######################### MODIFY INPUT NETWORK #########################
    # modify signaling network so all initializations and masking are correct
    sn_ppis = sn_ppis_all.copy()
    moa_removed_mask = sn_ppis.interaction_.apply(lambda x: x in (moa_removed))
    sn_ppis.loc[moa_removed_mask, weight_label] = 0.1


    mod, trainer = initialize_mod_and_trainer(
        fold = fold, 
        adversarial_penalty = True, 
        randomize = False, 
        seed = curr_seed, 
        net = sn_ppis # with removed MOAs
    )


    ######################### INITIALIZE REMOVED MOAS #########################
    # Unlike standard initialization, initialize with 0s to avoid any biases due to initialization
    n = len(moa_removed)
    sources = torch.empty(n, device=mod.device, dtype=torch.int32)
    targets = torch.empty(n, device=mod.device, dtype=torch.int32)

    for i, ppi in enumerate(moa_removed):
        source_id, target_id = ppi.split('^')
        sources[i] = mod.node_idx_map[source_id]
        targets[i] = mod.node_idx_map[target_id]

    mod.signaling_network.removed_moa = (targets, sources) 

    n = len(moa_removed)
    sources = torch.empty(n, device=mod.device, dtype=torch.int32)
    targets = torch.empty(n, device=mod.device, dtype=torch.int32)

    for i, ppi in enumerate(moa_removed):
        source_id, target_id = ppi.split('^')
        sources[i] = mod.node_idx_map[source_id]
        targets[i] = mod.node_idx_map[target_id]

    mod.signaling_network.removed_moa = (targets, sources) # store

    with torch.no_grad():
        mod.signaling_network.weights[mod.signaling_network.removed_moa] = torch.tensor(
            0, 
            device = mod.signaling_network.weights.device, 
            dtype = mod.signaling_network.weights.dtype
        )


    ######
    mod = trainer.train_model(verbose = False)


    write_model(
        mod=mod, 
        trainer=trainer, 
        moa_removed=moa_removed, 
        fn_base=fn_base, 
        save_trainer = False)


# # you are here:
# - What is the baseline?

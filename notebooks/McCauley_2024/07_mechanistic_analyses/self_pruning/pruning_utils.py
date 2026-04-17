import os
import sys 

import pandas as pd
import numpy as np
import torch


sys.path.insert(1, '../../.')
from McCauley_utils import initialize_mod_and_trainer, all_data

SEED_MULTIPLIER = 21234
SEED = 888

data_path = '/home/hmbaghda/orcd/pool/scLEMBAS/analysis'
author = 'McCauley'


(sn_ppis, tf_adata, adata, expr, source_label, target_label, weight_label, 
 stimulation_label, inhibition_label, cat_col, pert_col, ctrl_pert) = all_data

num_stochastic_edges = int(np.round(0.01*sn_ppis.shape[0]))

def load_model(fold, ensemble_idx, bn_weight_l2 = 0, bn_weight_l1 = 1e-3, from_trainer = False):
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
    curr_seed = SEED + ensemble_idx + 1 + (SEED_MULTIPLIER * ensemble_idx * fold)
    
    l2_string = '{:.0E}'.format(bn_weight_l2).replace('E-0', 'E-') if bn_weight_l2 != 0 else '0'
    l1_string = '{:.0E}'.format(bn_weight_l1).replace('E-0', 'E-') if bn_weight_l1 != 0 else '0'
    fn_base = os.path.join(data_path, 'processed', 'pruning_ensembles', '{}_fold{}_l2{}_l1{}'.format(author, fold, l2_string, l1_string))
    
    
    train_stats_df = pd.read_csv((fn_base + '_pruning_trainstats_actual_ensemble{}.csv'.format(ensemble_idx)), 
                                 index_col = 0)
    eval_df = pd.read_csv(fn_base + '_pruning_evalstats_actual_ensemble{}.csv'.format(ensemble_idx), index_col= 0)
    
    if from_trainer:
        fn_trainer =  os.path.join(fn_base + '_pruning_trainer_actual_ensemble{}.pickle'.format(ensemble_idx))
        trainer = io.read_pickled_object(fn_trainer)
        mod = trainer.mod
    else:
        mod, trainer = initialize_mod_and_trainer(
            fold = fold, 
            adversarial_penalty = True, 
            randomize = False, 
            num_stochastic_edges = num_stochastic_edges,
            seed = curr_seed, 
        )
        
        fn_mod = os.path.join(fn_base + '_pruning_model_actual_ensemble{}.pt'.format(ensemble_idx))
        mod.load_state_dict(torch.load(fn_mod))
        trainer = None
    return mod, trainer, train_stats_df, eval_df

def get_edge_weights(mod):
    """Returns the absolute value of the learned weights for the stochastic and real edges"""
    src, tar = mod.signaling_network.added_edges
    src_real, tar_real = mod.signaling_network.edge_real_edges

    # Extract data
    stochastic_edge_weights = torch.abs(mod.signaling_network.weights[src, tar].detach().cpu().flatten())
    real_edge_weights = torch.abs(mod.signaling_network.weights[src_real, tar_real].detach().cpu().flatten())

    return stochastic_edge_weights, real_edge_weights 
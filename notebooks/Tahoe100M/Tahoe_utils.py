#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from sklearn.model_selection import KFold
from tqdm import trange

def Tahoe100M_split(tf_adata, 
                   train_frac: float = 0.8, 
                   min_drug_frac: float = 0.6, 
                   min_cell_line_frac: float = 0.6, 
                    exclude_control: bool = False,
                   max_attempts: int = 1000, 
                   seed: int = 888 
                   ):
    """Create a train-test split for the Tahoe 100M data. 
    Splits by condition (drug + cell line), which leads to even splits in barcodes given balanced classes. 
    Also ensures a minimum fraction of total of each drug / cell line in conditions is in the train split 
    (this is separate and less than the condition split). 

    Parameters
    ----------
    tf_adata : _type_
        _description_
    train_frac : float, optional
        fraction of conditions to split into train, by default 0.8
    min_drug_frac : float, optional
        minimum fraction of conditions in train that should include each drug, by default 0.6
    min_cell_line_frac : float, optional
        minimum fraction of conditions in train that should include each cell line, by default 0.6
    exclude_control : bool, optional
        whether to ensure all control conditions are in the train (True) or not (False), by default False
    max_attempts : int, optional
        maximum number of tries to achieve a split that meets these requirements, by default 1000
    seed : int, optional
        random state, by default 888
    """
    
    if min_drug_frac > train_frac or min_cell_line_frac > train_frac:
        raise ValueError('The minimum perturbation/cell line fraction must be less than or equal to the train fraction')
    
    obs = tf_adata.obs.copy()
    unique_conditions = obs[['cell_line', 'drug', 'condition']].drop_duplicates().reset_index(drop = True)
    if unique_conditions.cell_line.value_counts().nunique() != 1 or unique_conditions.drug.value_counts().nunique() != 1:
        raise ValueError('This function was written expecting even drug and cell line counts')
        
    tot_drug_counts = unique_conditions.drug.value_counts().min()    
    min_train_count_drug = int(np.round(min_drug_frac*tot_drug_counts))
    min_train_count_cell_line = int(np.round(min_cell_line_frac*unique_conditions.cell_line.value_counts().min()))

    rng = np.random.default_rng(seed)
    
    test_frac = 1 - train_frac
    # used to calculate minimum train thresholds
    n_drugs = obs['drug'].nunique()
    n_cell_lines = obs['cell_line'].nunique()
    n_conds = obs['condition'].nunique()
    n_test = int(np.round(n_conds * test_frac))


    for attempt in range(max_attempts):
        # shuffle conditions
        unique_conditions = unique_conditions.sample(frac=1.0, random_state=rng.integers(0, 1e6)).reset_index(drop=True)


        test_conds = unique_conditions.iloc[:n_test]
        train_conds = unique_conditions.iloc[n_test:]

        train_drug_counts = train_conds['drug'].value_counts()
        train_cell_line_counts = train_conds['cell_line'].value_counts()

        a = np.all(train_drug_counts > min_train_count_drug)
        b = np.all(train_cell_line_counts > min_train_count_cell_line)
        c = True
        if exclude_control:
            c = (train_drug_counts['DMSO_TF'] == tot_drug_counts)
            

        if (a and b and c):
            break

    if attempt == max_attempts - 1:
        raise ValueError('Maximum attempts reached')
    
    train_split = {'conditions': train_conds, 
                  'drug_counts': train_drug_counts,
                  'cell_line_counts': train_cell_line_counts, 
                  'barcodes': obs[obs.condition.isin(train_conds.condition)].index.tolist()}
    
    test_split = {'conditions': test_conds, 
                 'drug_counts': test_conds['drug'].value_counts(), 
                 'cell_line_counts': test_conds['cell_line'].value_counts(), 
                 'barcodes': obs[obs.condition.isin(test_conds.condition)].index.tolist()}

    
    if len(set(train_conds.condition).intersection(test_conds.condition)) != 0:
        raise ValueError('Train and test conditions overlap')
    
    return train_split, test_split


def Tahoe100M_kfold(tf_adata, 
                    n_splits: int = 5,
                    min_drug_frac: float = 0.6,
                    min_cell_line_frac: float = 0.6,
                    exclude_control: bool = False, 
                    max_attempts: int = 1000,
                    seed: int = 888):
    """Creates k-fold cross-validation splits in the fashion of `Tahoe100M_split`.
    
    Splits by condition (drug + cell line), which leads to even splits in barcodes given balanced classes. 
    Also ensures a minimum fraction of total of each drug / cell line in conditions is in the train split 
    (this is separate and less than the condition split). 


    Parameters
    ----------
    tf_adata : _type_
        _description_
    n_splits : float, optional
        k-splits to create
    min_drug_frac : float, optional
        minimum fraction of conditions in train that should include each drug, by default 0.6
    min_cell_line_frac : float, optional
        minimum fraction of conditions in train that should include each cell line, by default 0.6
    exclude_control : bool, optional
        whether to ensure all control conditions are in the train (True) or not (False), by default False
    max_attempts : int, optional
        maximum number of tries to achieve a split that meets these requirements, by default 1000
    seed : int, optional
        random state, by default 888
    """
    
    train_frac = 1 - (1/n_splits)

    if min_drug_frac > train_frac or min_cell_line_frac > train_frac:
        raise ValueError('The minimum perturbation/cell line fraction must be less than or equal to the train fraction')
    
    obs = tf_adata.obs.copy()
    unique_conditions = obs[['cell_line', 'drug', 'condition']].drop_duplicates().reset_index(drop = True)
    if unique_conditions.cell_line.value_counts().nunique() != 1 or unique_conditions.drug.value_counts().nunique() != 1:
        raise ValueError('This function was written expecting even drug and cell line counts')
        
    tot_drug_counts = unique_conditions.drug.value_counts().min()    
    min_train_count_drug = int(np.round(min_drug_frac*tot_drug_counts))
    min_train_count_cell_line = int(np.round(min_cell_line_frac*unique_conditions.cell_line.value_counts().min()))

    rng = np.random.default_rng(seed)
    
    test_frac = 1 - train_frac
    # used to calculate minimum train thresholds
    n_drugs = obs['drug'].nunique()
    n_cell_lines = obs['cell_line'].nunique()
    n_conds = obs['condition'].nunique()
    n_test = int(np.round(n_conds * test_frac))
    


    for attempt in trange(max_attempts):
        folds = []
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=rng.integers(1e6))

        for train_idx, test_idx in kf.split(unique_conditions):
            train_conds = unique_conditions.iloc[train_idx]
            test_conds = unique_conditions.iloc[test_idx]
            # shuffle conditions
            unique_conditions = unique_conditions.sample(frac=1.0, random_state=rng.integers(0, 1e6)).reset_index(drop=True)


            test_conds = unique_conditions.iloc[:n_test]
            train_conds = unique_conditions.iloc[n_test:]

            train_drug_counts = train_conds['drug'].value_counts()
            train_cell_line_counts = train_conds['cell_line'].value_counts()

            a = np.all(train_drug_counts > min_train_count_drug)
            b = np.all(train_cell_line_counts > min_train_count_cell_line)
            c = True
            if exclude_control:
                c = (train_drug_counts['DMSO_TF'] == tot_drug_counts)


            if (a and b and c):
                folds.append({'train': train_conds, 
                             'test': test_conds})

        if len(folds) == n_splits:
            break
    if attempt == max_attempts - 1:
        raise ValueError('Maximum attempts reached')
        
    train_splits = []
    test_splits = []
    for k in folds:
        for split_type, conds in k.items():
            items = {'conditions': conds, 
                     'drug_counts': conds['drug'].value_counts(),
                     'cell_line_counts': conds['cell_line'].value_counts(), 
                     'barcodes': obs[obs.condition.isin(conds.condition)].index.tolist()}

            if split_type == 'train':
                train_splits.append(items)
            else:
                test_splits.append(items)

        

    return train_splits, test_splits

# USAGE

# train_split, test_split = Tahoe100M_split(tf_adata,
#                                           train_frac = 0.9, 
#                                           min_drug_frac = 0.7, 
#                                           min_cell_line_frac = 0.7,
#                                           exclude_control = True,
#                                           max_attempts = 1000, 
#                                           seed = 888)

# train_splits, test_splits = Tahoe100M_kfold(tf_adata, 
#                                             n_splits=10, 
#                                             min_drug_frac = 0.6, 
#                                             min_cell_line_frac = 0.6, 
#                                           exclude_control = True,
#                                             max_attempts = 1000, 
#                                             seed = 888)


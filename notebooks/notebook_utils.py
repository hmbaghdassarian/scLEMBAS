import os
import json

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

import torch

data_path = '/home/hmbaghda/orcd/pool/scLEMBAS/analysis'

def get_split(fold, author):
    with open(os.path.join(data_path, 'processed', author + '_5foldCV_splits.json'), "r") as f:
        all_splits = json.load(f)

    if type(fold) == int:
        fold = str(fold)
    return all_splits[fold]

#---------------------------- Csendes et al RF baseline ---------------------------

def load_csendes_go_pert_embedding(seed: int = 888):
    """GO perturbation embedding as generated in Csendes et al (https://doi.org/10.1186/s12864-025-11600-2)"""
    # perturbation embedding
    go_fn = os.path.join(data_path, 'interim', 'Csended_GO_embedding.csv')
    if not os.path.isfile(go_fn):
        #https://github.com/turbine-ai/PerturbSeqPredBenchmark/blob/main/data/go/go_raw_matched.csv
        go = pd.read_csv(os.path.join(data_path, 'raw', 'Csendes_go_raw_matched.csv'), index_col = 0)


        # add a control no perturbation
        go.loc['perturbation_control', :] = 0.0

        pca_mod = PCA(n_components=256, random_state = seed)
        pca_mod.fit(go)
        go_pca = pca_mod.transform(go)
        go_pca = pd.DataFrame(go_pca, index = go.index)
        go_pca.to_csv(go_fn)
    else:
        go_pca = pd.read_csv(go_fn, index_col = 0)
    return go_pca

def RF_y(adata, barcodes):
    """Generates the Csendes et al RF baseline y-block"""
    adata_sub = adata[barcodes, :].copy()
    y_out = adata_sub.to_df().copy()
    
    return y_out

def RF_X(adata, barcodes, embedding, author):
    """Generates the Csendes et al RF baseline X-block 
    y mapping input perturbations to embedding
   """
    adata_sub = adata[barcodes, :].copy()
    
    if author == 'McCauley':
        perts = adata_sub.obs['ligand'].cat.rename_categories({'CTRL': 'perturbation_control'}).astype(str).copy()
    elif author == 'Kang':
        pert_map = {'CTRL': 'perturbation_control', 'STIM': 'IFNB1'}
        perts = adata_sub.obs['stim'].cat.rename_categories(pert_map).astype(str).copy()
    X = embedding.loc[perts, :].copy()
    
    return X

def RF_baseline(fold, adata, author, n_cores, seed: int = 888):
    """
    Emulates Random Forest GO baseline from https://doi.org/10.1186/s12864-025-11600-2.
    
    Differences include:
        - GO binarized input to PCA includes the control perturbation (vector of all 0s), to allow train/test
        to contain controls and be directly comparable to scLEMBAS splits
        - Model X/y inputs are not psuedo-bulked by perturbation (because will results in very few "samples"). 
        Given that the X_block is a perturbation embedding, the RF model will internally psuedobulk the 
        output features anyways.
    """

    go_pca = load_csendes_go_pert_embedding(seed = seed)

    # data split
    split = get_split(fold = fold, author = author)
    y_train = RF_y(adata = adata, barcodes = split['train_barcodes'])
    y_test = RF_y(adata = adata, barcodes = split['test_barcodes'])

    X_train = RF_X(adata = adata, barcodes = split['train_barcodes'], embedding = go_pca, author = author)
    X_test = RF_X(adata = adata, barcodes = split['test_barcodes'], embedding = go_pca, author = author)

    # prediction
    rf_mod = RandomForestRegressor(n_jobs = n_cores, random_state = seed)
    rf_mod.fit(X_train, y_train)
    y_pred = rf_mod.predict(X_test)
    y_pred = pd.DataFrame(y_pred, index = y_test.index, columns=y_test.columns)
    return y_pred

#---------------------------- Csendes et al Mean baseline ---------------------------
def pert_psuedobulked_mean(adata, barcodes, pert_col):
    """
    Returns mean as in Csendes et al (https://doi.org/10.1186/s12864-025-11600-2) and Ahlmann-Eltze et al (https://doi.org/10.1038/s41592-025-02772-6)
    
    1) Psuedo-bulk by perturbation
    2) Take mean across perturbations
    
    Returns
    Y :
        Psueo-bulked data by perturbation (this is `Y_train` in Ahlmann-Eltze et al)
    b :
        Perturbation-wise mean for each feature (this is the mean baseline used in the manuscripts)
    
    """
    
    adata_sub = adata[barcodes, :].copy()
    Y = adata_sub.to_df()
    Y[pert_col] = adata_sub.obs[pert_col]
    Y = Y.groupby(pert_col, observed = False).mean().T

    b = Y.mean(axis = 1)
    
    return Y, b

#----------------------------  Ahlmann-Eltze et al linear baseline -------------------------
def linear_baseline(fold, 
                    adata,
                    pert_col: str, 
                    author: str, 
                    seed: int = 888):
    """
    Emulates the linear baseline in Ahlmann-Eltze et al (https://doi.org/10.1038/s41592-025-02772-6). 
    
    The following differences:
    - Because the gene embedding `G` is calculated on TF activity, it cannot be directly used for the perturbation embedding `P`. 
    Instead, `P` is retrieved from the GO Embedding as in  Csendes et al (https://doi.org/10.1186/s12864-025-11600-2)
    """

    split = get_split(fold = fold, author = author)

    # fit the linear model
    Y_train, b_train = pert_psuedobulked_mean(adata = adata, barcodes = split['train_barcodes'], pert_col = pert_col)

    # McCauley has 6 perturbations, Kang has 2
    if author == 'McCauley':
        author_n_components = 2 
    elif author == 'Kang':
        author_n_components = 1
    G = PCA(n_components=author_n_components, random_state = seed).fit_transform(Y_train)
    G = pd.DataFrame(G, index = Y_train.index)

    go_pca = load_csendes_go_pert_embedding(seed = seed)
    
    # don't need to subset to train because all perturbations are present in training with the way we split
    if author == 'McCauley':
        pert_map = {'CTRL': 'perturbation_control'}
        perts = adata.obs[pert_col].cat.rename_categories(pert_map).cat.categories.tolist()
    elif author == 'Kang':
        pert_map = {'CTRL': 'perturbation_control', 'STIM': 'IFNB1'}
        perts = adata.obs[pert_col].cat.rename_categories(pert_map).cat.categories.tolist()

        
    assert Y_train.columns.tolist() == pd.Series(perts).replace({v:k for k,v in pert_map.items()}).tolist(), 'Y and P must be in the same order'

    P_train = go_pca.loc[perts, :]

    lambda_reg = 0.1 # as in Ahlmann-Eltze
    resid = Y_train.values - b_train.values.reshape(-1,1)
    G_arr = G.values  # Convert DataFrame to numpy array
    P_arr = P_train.values
    W = (np.linalg.inv(G_arr.T @ G_arr + lambda_reg * np.eye(G_arr.shape[1])) 
         @ G_arr.T 
         @ resid 
         @ P_arr 
         @ np.linalg.inv(P_arr.T @ P_arr + lambda_reg * np.eye(P_arr.shape[1])))

    if author == 'McCauley':
        test_perts = adata[split['test_barcodes'], :].obs[pert_col].cat.rename_categories({'CTRL': 'perturbation_control'}).cat.categories.tolist()
    elif author == 'Kang':
        pert_map = {'CTRL': 'perturbation_control', 'STIM': 'IFNB1'}
        test_perts = adata[split['test_barcodes'], :].obs['stim'].cat.rename_categories(pert_map).cat.categories.tolist()

    P_test = P_train.loc[test_perts, :]
    y_pred = (G_arr @ W @ P_test.T.values) +  b_train.values.reshape(-1,1)
    y_pred = pd.DataFrame(y_pred, index = Y_train.index, columns = P_test.index)
    
    return y_pred



from typing import List, Literal

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def split_data(tf_adata,
               train_frac: float = 0.8,
               min_pert_frac: float = 0.4,
               min_cat_frac: float = 0.5,
               deviation_thresh: float = 0.025,
               max_attempts: int = 1000,
               exclude_pert_control: bool = True,
               pert_col: str = 'ligand',
               ctrl_pert: str ='CTRL',
               seed: int = 888 
                   ):
    """Create a train-test split for the Tahoe 100M data. 
    Splits by condition (pert + cell type) and even splits in barcodes. 
    Also ensures a minimum fraction of total of each drug / cell line in conditions is in the train split 
    (this is separate and less than the condition split). 

    Parameters
    ----------
    tf_adata : _type_
        _description_
    train_frac : float, optional
        fraction of conditions to split into train, by default 0.8
    min_pert_frac : float, optional
        minimum fraction of conditions in train that should include each drug, by default 0.6
    min_cat_frac : float, optional
        minimum fraction of conditions in train that should include each cell line, by default 0.6
    exclude_pert_control : bool, optional
        whether to ensure all control perturbations are in the train (True) or not (False), by default False
    max_attempts : int, optional
        maximum number of tries to achieve a split that meets these requirements, by default 1000
    seed : int, optional
        random state, by default 888
    """
        
    n_cells = tf_adata.n_obs

    obs = tf_adata.obs.copy()
    n_tot = obs.condition.nunique()
    if exclude_pert_control:
        ctrl_mask = obs[pert_col] == ctrl_pert
        ctrl_conds = obs.loc[ctrl_mask, 'condition'].unique().tolist()
    else: 
        ctrl_conds = []

    # exclude from split
    obs_noctrl = obs.loc[~obs.condition.isin(ctrl_conds)].copy()
    n_ctrl = len(ctrl_conds)
    train_frac_adj = ((train_frac * n_tot) - n_ctrl) / obs_noctrl.condition.nunique()
    test_frac_adj = 1 - train_frac_adj
    

    n_iter = 0
    conditions_unmet = True
    while conditions_unmet and (n_iter < max_attempts):
        train_conds, test_conds = train_test_split(obs_noctrl.condition.unique().tolist(), 
                                                   test_size = test_frac_adj, 
                                                   random_state = seed + n_iter, shuffle = True)
        train_conds += ctrl_conds


        # cell split matches condition split
        train_mask = obs.condition.isin(train_conds)
        n_train_cells = np.sum(train_mask)
        cell_train_frac = n_train_cells / n_cells
        deviation = np.abs(cell_train_frac - train_frac)
        outside_deviation = deviation > deviation_thresh


        # make sure it's not zero shot
        test_cats, test_perts = map(set, zip(*[cond.split('^') for cond in test_conds]))
        train_cats, train_perts = map(set, zip(*[cond.split('^') for cond in train_conds]))
        zero_shot = bool(test_cats - train_cats or test_perts - train_perts)

        pert_frac = len(test_perts)/len(test_perts | train_perts)
        under_pert_frac = pert_frac < min_pert_frac

        cat_frac = len(test_cats)/len(test_cats | train_cats)
        under_cat_frac = cat_frac < min_cat_frac

        conditions_unmet = outside_deviation or zero_shot or under_pert_frac or under_cat_frac
        n_iter += 1

    if n_iter >= max_attempts:
        return None
    else:
        split_dict = {
            'train_conds': train_conds, 
            'test_conds': test_conds, 
            'train_barcodes': obs.loc[train_mask, :].index.tolist(), 
            'test_barcodes': obs.loc[~train_mask, :].index.tolist()
        }
        return split_dict
    
reduction_type_map = {'pca': 'pc', 
                      'pls': 'pls', 
                      'umap': 'umap', 
                      'umap_pls': 'umap (pls)'}
def adata_dimviz(
        adata, 
        reduction_type: Literal['pca', 'pls', 'umap', 'umap_pls'], 
        cats: List[str], 
        subset_size: int = int(1e4), 
        seed: int = 888):
    """Formats for visualization of embedding

    Parameters
    ----------
    adata : _type_
        AnnData object
    reduction_type : Literal['pca', 'pls', 'umap', 'umap_pls']
        embedding to visualize
    cats : List[str]
        columns in adata.obs to retain
    subset_size : int, optional
        proportionally subsets across all cats, by default int(1e4)
    seed : int, optional
        random state, by default 888
    """


    reduction_type_ = reduction_type_map[reduction_type]

    if type(cats) != list:
        cats = [cats]

    viz_df = pd.DataFrame(adata.obsm['X_' + reduction_type])
    viz_df = pd.concat([viz_df, pd.DataFrame(adata.obs[cats]).reset_index(drop = True)], ignore_index = True, axis = 1)
    viz_df.columns = [reduction_type_.upper() + str(i+1) for i in range(viz_df.shape[1])][:-len(cats)] + cats


#     nmi = normalized_mutual_info_score(adata.obs.leiden, adata.obs[cat])

    if subset_size is not None and subset_size < viz_df.shape[0]:
        grouped = viz_df.groupby(cats, observed=False)
        cell_prop = viz_df[cats].value_counts(normalize = True)
        index_to_keep = []
        np.random.seed(seed)
        for cat_type, cat_df in grouped:
            mask = (viz_df[cats].values == cat_type)
            if len(cats) > 1:
                mask = mask.all(axis=1)
            all_barcodes = viz_df[mask].index
            
            cat_subset_size = np.round(max(1, np.round(cell_prop.loc[cat_type]*subset_size)))
            cat_subset_size = min(cat_subset_size, len(all_barcodes))
            cat_subset_size = int(cat_subset_size)

            
            index_to_keep += np.random.choice(all_barcodes, 
                         size = cat_subset_size,
                         replace = False).tolist()

        viz_df = viz_df.loc[index_to_keep, :]

    # shuffle
    viz_df = viz_df.sample(frac=1, random_state = seed).reset_index(drop=True)
    
    return viz_df
"""Common functions used across multiple scripts"""

def get_best_hyperparams(res, 
                         iter_cols = ['max_epochs', 'max_lr', 'train_batch_size'], 
                        loss_col = 'emd_loss_total'):
    
    res_nans = res.groupby(iter_cols)[loss_col].apply(lambda x: x.isna().any()).reset_index()
    res_nans = res_nans[res_nans[loss_col]]
    res_nans.set_index(iter_cols, inplace = True)

    mean_emd = res.groupby(iter_cols).emd_loss_total.mean().reset_index()
    mean_emd = mean_emd.set_index(iter_cols).drop(res_nans.index)
    best_emd_loss = mean_emd.emd_loss_total.min()
    best_emd = mean_emd[mean_emd.emd_loss_total == best_emd_loss]
    best_emd_mean = best_emd.copy()
    
    best_hyperparams = dict(zip(best_emd.index.names, best_emd.index[0]))
    best_emd = res[(res.max_epochs == best_hyperparams['max_epochs']) & 
                   (res.max_lr == best_hyperparams['max_lr']) & 
                   (res.train_batch_size == best_hyperparams['train_batch_size'])]
    
    return best_emd_mean, best_hyperparams, best_emd

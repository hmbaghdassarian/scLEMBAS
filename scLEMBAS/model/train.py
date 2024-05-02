"""
Train the signaling model.
"""
from typing import Dict, List, Union, Optional
import time
from tqdm import trange

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import scLEMBAS.utilities as utils


LR_PARAMS = {'max_epochs': 5000, 'learning_rate': 2e-3, 'reset_optimizer_epoch': 200}
OTHER_PARAMS = {'batch_size': 32, 'network_noise_scale': 10, 'gradient_noise_scale': 1e-9}
REGULARIZATION_PARAMS = {'param_lambda_L2': 1e-6, 'moa_lambda_L1': 0.1, 'ligand_lambda_L2': 1e-5, 'uniform_lambda_L2': 1e-4, 
                   'uniform_max': (1/1.2), 'spectral_loss_factor': 1e-5}
SPECTRAL_RADIUS_PARAMS = {'n_probes_spectral': 5, 'power_steps_spectral': 50, 'subset_n_spectral': 10}
HYPER_PARAMS = {**LR_PARAMS, **OTHER_PARAMS, **REGULARIZATION_PARAMS, **SPECTRAL_RADIUS_PARAMS}

def split_data(X_in: torch.Tensor, 
               y_out: torch.Tensor, 
               train_split_frac: Dict = {'train': 0.8, 'test': 0.2, 'validation': None}, 
              seed: int = 888):
    """Splits the data into train, test, and validation.

    Parameters
    ----------
    X_in : torch.Tensor
        input ligand concentrations. Index represents samples and columns represent a ligand. Values represent amount of ligand introduced (e.g., concentration). 
    y_out : torch.Tensor
        output TF activities. Index represents samples and columns represent TFs. Values represent activity of the TF.
    train_split_frac : Dict, optional
        fraction of samples to be assigned to each of train, test and split, by default 0.8, 0.2, and 0 respectively
    seed : int, optional
        seed value, by default 888
    """
    
    if not np.isclose(sum([v for v in train_split_frac.values() if v]), 1):
        raise ValueError('Train-test-validation split must sum to 1')
    
    if not train_split_frac['validation'] or train_split_frac['validation'] == 0:
        X_train, X_test, y_train, y_test = train_test_split(X_in, 
                                                        y_out, 
                                                        train_size=train_split_frac['train'],
                                                        random_state=seed)
        X_val, y_val = None, None
    else:
        X_train, _X, y_train, _y = train_test_split(X_in, 
                                                        y_out, 
                                                        train_size=train_split_frac['train'],
                                                        random_state=seed)
        X_test, X_val, y_test, y_val = train_test_split(_X, 
                                                    _y, 
                                                    train_size=train_split_frac['test']/(train_split_frac['test'] + train_split_frac['validation']),
                                                    random_state=seed)

    return X_train, X_test, X_val, y_train, y_test, y_val

class ModelData(Dataset):
    def __init__(self, X_in: torch.tensor, y_out: torch.tensor, covariates_idx: Optional[torch.tensor] = None):
            """_summary_

            Parameters
            ----------
            X_in : torch.tensor
                _description_
            y_out : torch.tensor
                _description_
            covariates_idx : Optional[torch.tensor], optional
                the numerical index representation of the categorical covariates for each sample, by default None
                can be obtained from `mod.signaling_network.covariates_idx`
            """
            self.X_in = X_in
            self.y_out = y_out
    #         if self.covariates is not None:
    #             self.covariates = covariates
    #         else:
    #             self.covariates = torch.full(X_train.shape, torch.nan)
            if covariates_idx is not None:
                self.covariates_idx = covariates_idx
            else:
                self.covariates_idx = torch.full(self.X_in.shape, torch.nan, device='cpu')

    def __len__(self) -> int:
        "Returns the total number of samples."
        return self.X_in.shape[0]
    def __getitem__(self, idx: int):
        "Returns one sample of data, data and label (X, y)."
        return self.X_in[idx, :], self.y_out[idx, :], self.covariates_idx[idx,:]

def train_signaling_model(mod,  
                          optimizer: torch.optim, 
                          loss_fn: torch.nn.modules.loss,
                          reset_epoch : int = 200,
                          hyper_params: Dict[str, Union[int, float]] = None,
                          train_split_frac: Dict = {'train': 0.8, 'test': 0.2, 'validation': None},
                          train_seed: int = None,
                          verbose: bool = True):
    """Trains the signaling model

    Parameters
    ----------
    mod : SignalingModel
        initialized signaling model. Suggested to also run `mod.signaling_network.prescale_weights` prior to training
    optimizer : torch.optim.adam.Adam
        optimizer to use during training
    loss_fn : torch.nn.modules.loss.MSELoss
        loss function to use during training
    hyper_params : Dict[str, Union[int, float]], optional
        various hyper parameter inputs for training
            - 'max_epochs' : the number of epochs, by default 5000
            - 'learning_rate' : the starting learning rate, by default 2e-3
            - 'reset_optimizer_epoch' : number of epochs upon which to reset the optimizer state, by default 200
            - 'batch_size' : number of samples per batch, by default 8
            - 'network_noise_scale' : noise added to signaling network input, by default 10. Set to 0 for no noise. Makes model more robust. 
            - 'gradient_noise_scale' : noise added to gradient after backward pass. Makes model more robust. 
            - 'reset_epoch' : number of epochs upon which to reset the optimizer state, by default 200
            - 'param_lambda_L2' : L2 regularization penalty term for most of the model weights and biases
            - 'moa_lambda_L1' : L1 regularization penalty term for incorrect interaction mechanism of action (inhibiting/stimulating)
            - 'ligand_lambda_L2' : L2 regularization penalty term for ligand biases
            - 'uniform_lambda_L2' : L2 regularization penalty term for 
            - 'uniform_max' : 
            - 'spectral_loss_factor' : regularization penalty term for 
            - 'n_probes_spectral' : 
            - 'power_steps_spectral' : 
            - 'subset_n_spectral' : 
    train_split_frac : Dict, optional
        fraction of samples to be assigned to each of train, test and split, by default 0.8, 0.2, and 0 respectively
    train_seed : int, optional
        seed value, by default mod.seed. By explicitly making this an argument, it allows different train-test splits even 
        with the same mod.seed, e.g., for cross-validation
    verbose : bool, optional
        whether to print various progress stats across training epochs


    Returns
    -------
    mod : SignalingModel
        a copy of the input model with trained parameters
    cur_loss : List[float], optional
        a list of the loss (excluding regularizations) across training iterations
    cur_eig : List[float], optional
        a list of the spectral_radius across training iterations
    mean_loss : torch.Tensor
        mean TF activity loss across samples (independent of training)
    split_data_dict : Dict[str, pd.DataFrame]
        key value pairs represent the output of the `split_data` function
    """
    if not mod.signaling_network._prescaled_weights:
        warnings.warn('Recommended to run `mod.signaling_network.prescale_weights()` prior to training')
    
    
    if not hyper_params:
        hyper_params = HYPER_PARAMS.copy()
    else:
        hyper_params = {k: v for k,v in {**HYPER_PARAMS, **hyper_params}.items() if k in HYPER_PARAMS} # give user input priority
    
    stats = utils.initialize_progress(hyper_params['max_epochs'])

    optimizer = optimizer(mod.parameters(), lr=1, weight_decay=0)
    reset_state = optimizer.state.copy()

    # set up data objects
    if not train_seed:
        train_seed = mod.seed

    X_train, X_test, X_val, y_train, y_test, y_val = split_data(mod.X_in, mod.y_out, train_split_frac, train_seed)
    split_data_dict = {'X_train': X_train, 'X_test': X_test, 'X_val': X_val, 
                      'y_train': y_train, 'y_test': y_test, 'y_val': y_val}

#     y_train = mod.df_to_sensor(y_train)
#     mean_loss = loss_fn(torch.mean(y_train, dim=0) * torch.ones(y_train.shape, device = y_train.device), y_train) # mean TF (across samples) loss
    
    
    if mod.signaling_network.covariates is not None:
        covariates_idx = mod.signaling_network.covariates_to_tensor(sample_ids = X_train.index)
    else:
        covariates_idx = None
    train_data = ModelData(X_in = mod.df_to_tensor(X_train).to('cpu'), 
                           y_out = mod.df_to_tensor(y_train).to('cpu'),
                           covariates_idx = covariates_idx)
#     if mod.device == 'cuda':
#         pin_memory = True
#     else:
#         pin_memory = False

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=hyper_params['batch_size'],
                                  # num_workers=n_cores_train,
                                  drop_last = False,
                                  pin_memory = False,#pin_memory,
                                  shuffle=True) 
    start_time = time.time()
    # begin iteration
    for e in trange(hyper_params['max_epochs']):
        # set learning rate
        cur_lr = utils.get_lr(e, hyper_params['max_epochs'], max_height = hyper_params['learning_rate'],
                              start_height=hyper_params['learning_rate']/10, end_height=1e-6, peak = 1000)
        optimizer.param_groups[0]['lr'] = cur_lr
        
        cur_loss = []
        cur_eig = []
        
        # iterate through batches
        if mod.seed:
            utils.set_seeds(mod.seed + e)
        for batch, (X_in_, y_out_, covariates_idx_) in enumerate(train_dataloader):
            mod.train()
            optimizer.zero_grad()

            X_in_, y_out_, covariates_idx_ = X_in_.to(mod.device), y_out_.to(mod.device), covariates_idx_.to(mod.device)
            covariates_idx_ = covariates_idx_.to(mod.device) if covariates_idx is not None else None

            # forward pass
            X_full = mod.input_layer(X_in_) # transform to full network with ligand input concentrations
            utils.set_seeds(mod.seed + mod._gradient_seed_counter)
            network_noise = torch.randn(X_full.shape, device = X_full.device)
            X_full = X_full + (hyper_params['network_noise_scale'] * cur_lr * network_noise) # randomly add noise to signaling network input, makes model more robust
            Y_full = mod.signaling_network(X_full, covariates_idx_) # train signaling network weights
            Y_hat = mod.output_layer(Y_full)
            
            # get prediction loss
            fit_loss = loss_fn(y_out_, Y_hat)
            
            # get regularization losses
            sign_reg = mod.signaling_network.sign_regularization(lambda_L1 = hyper_params['moa_lambda_L1']) # incorrect MoA
            ligand_reg = mod.ligand_regularization(lambda_L2 = hyper_params['ligand_lambda_L2']) # ligand biases
            stability_loss, spectral_radius = mod.signaling_network.get_SS_loss(Y_full = Y_full.detach(), spectral_loss_factor = hyper_params['spectral_loss_factor'],
                                                                                subset_n = hyper_params['subset_n_spectral'], n_probes = hyper_params['n_probes_spectral'], 
                                                                                power_steps = hyper_params['power_steps_spectral'])
            uniform_reg = mod.uniform_regularization(lambda_L2 = hyper_params['uniform_lambda_L2']*cur_lr, Y_full = Y_full, 
                                                     target_min = 0, target_max = hyper_params['uniform_max']) # uniform distribution
            param_reg = mod.L2_reg(hyper_params['param_lambda_L2']) # all model weights and signaling network biases
            
            total_loss = fit_loss + sign_reg + ligand_reg + param_reg + stability_loss + uniform_reg
    
            # gradient
            total_loss.backward()
            mod.add_gradient_noise(noise_level = hyper_params['gradient_noise_scale'])
            optimizer.step()
    
            # store
            cur_eig.append(spectral_radius)
            cur_loss.append(fit_loss.item())
    
        stats = utils.update_progress(stats, iter = e, loss = cur_loss, eig = cur_eig, learning_rate = cur_lr, 
                                     n_sign_mismatches = mod.signaling_network.count_sign_mismatch())
        
        if verbose and e % (hyper_params['max_epochs']/100) == 0:
            utils.print_stats(stats, iter = e)
        
        if np.logical_and(e % hyper_params['reset_optimizer_epoch'] == 0, e>0):
            optimizer.state = reset_state.copy()

    if verbose:
        mins, secs = divmod(time.time() - start_time, 60)
        print("Training ran in: {:.0f} min {:.2f} sec".format(mins, secs))

    return mod, cur_loss, cur_eig, mean_loss, stats, split_data_dict
"""
Train the signaling model.
"""
from typing import Dict, List, Union, Optional
import time
from tqdm import trange
import warnings
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import scLEMBAS.utilities as utils
from .model_utilities import update_with_defaults
from .bionetwork import BioNetSimple, BioNetCat
from .model_components import CatDiscriminator
from .lr_schedulers import WarmupCosineAnnealingWarmRestarts

# configure logger
if not logging.getLogger().hasHandlers():
    logging.basicConfig(filename='train.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
else:
    handler = logging.FileHandler('train.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.ERROR)
        
    
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
    
    
class TrainBase:
    """Base class for training the signaling model."""

    LR_PARAMS = {'max_epochs': 5000, 'maximum_learning_rate': 2e-3, 'minimum_learning_rate': 2e-4,
                 'lr_restart_epoch': 1000, 'reset_optimizer_epoch': 200, 
                'lr_decay': 0.9, 'lr_restart_factor': 1, 'warmup_epochs': 500}
    OTHER_PARAMS = {'train_batch_size': 512, 'test_batch_size': 512, 'validation_batch_size': 512, 
                    'network_noise_scale': 10, 'gradient_noise_scale': 1e-9}
    REGULARIZATION_PARAMS = {'input_lambda_L2': 1e-6, 'hidden_state_lambda_L2': 1e-6, 'bias_lambda_L2': 1e-6, 
                             'output_lambda_L2': 1e-6,
                             'moa_lambda_L1': 0.1, #'ligand_lambda_L2': 1e-5, 
                            'uniform_lambda_L2': 1e-4, 'uniform_max': (1/1.2), 'spectral_loss_factor': 1e-5}
    SPECTRAL_RADIUS_PARAMS = {'n_probes_spectral': 5, 'power_steps_spectral': 50, 'subset_n_spectral': 10}
    HYPER_PARAMS = {**LR_PARAMS, **OTHER_PARAMS, **REGULARIZATION_PARAMS, **SPECTRAL_RADIUS_PARAMS}

    def __init__(self, 
                 mod, 
                 prediction_optimizer: torch.optim, 
                 prediction_loss_fn: torch.nn.modules.loss,
                 hyper_params: Dict[str, Union[int, float]] = None,
                 # train_samples: Optional[List[str]] = None, 
                 train_split: Optional[Dict[str, Union[float, List[str]]]] = {'train': 0.8, 'test': 0.2, 'validation': None},
                 train_seed: int = None, 
                 track_test: bool = False,
                 track_validation: bool = False
                ):
        """Trains the signaling model

        Parameters
        ----------
        mod : SignalingModel
            initialized signaling model. Suggested to also run `mod.signaling_network.prescale_weights` prior to training
        prediction_optimizer : torch.optim.adam.Adam
            optimizer to use for the prediction during training (as opposed to the discriminator)
        prediction_loss_fn : torch.nn.modules.loss.MSELoss
            loss function for determining how well the model predicts the output (TF activities)
        hyper_params : Dict[str, Union[int, float]], optional
            various hyper parameter inputs for training
                - 'max_epochs' : the number of epochs, by default 5000
                - 'maximum_learning_rate' : the maximum learning rate for cosine annealing, by default 2e-3
                - 'minimum_learning_rate' : the minimum learning rate for cosine annealing, by default 2e-4. Equivalent to `eta_min` argument in `CosineAnnealingWarmRestarts`
                - 'lr_restart_epoch' : epoch at which to conduct a warm restart, by default 1000. Equivalent to `T_0` argument in `CosineAnnealingWarmRestarts`
                - 'lr_decay': amount to change the range in LR during warm restarts, by default 0.9
                - 'lr_restart_factor': amount to change the frequency of warm restarts, by default 1. Equivalent to `T_mult` argument in `CosineAnnealingWarmRestarts`
                - 'warmup_epochs' : number of epochs to linearly increase LR up to `maximum_learning_rate` prior to beginning
                cosine annealing with warm restarts, by default 500
                - 'reset_optimizer_epoch' : number of epochs upon which to reset the optimizer state, by default 200
                - 'train_batch_size' : number of samples/cells per batch for training data, by default 512
                - 'test_batch_size' : number of samples/cells per batch for test data, by default 512
                - 'validation_batch_size' : number of samples/cells per batch for test data, by default 512
                - 'network_noise_scale' : noise added to signaling network input, by default 10. Set to 0 for no noise. Makes model more robust. 
                - 'gradient_noise_scale' : noise added to gradient after backward pass. Makes model more robust. 
                - 'reset_epoch' : number of epochs upon which to reset the optimizer state, by default 200
                - 'param_lambda_L2' : L2 regularization penalty term for most of the model weights and biases
                - 'moa_lambda_L1' : L1 regularization penalty term for incorrect interaction mechanism of action (inhibiting/stimulating)
                - 'ligand_lambda_L2' : DEPRECATED, DO NOT ADD KEY/VALUE PAIR. L2 regularization penalty term for ligand biases. 
                - 'uniform_lambda_L2' : L2 regularization penalty term for 
                - 'uniform_max' : 
                - 'spectral_loss_factor' : regularization penalty term for 
                - 'n_probes_spectral' : 
                - 'power_steps_spectral' : 
                - 'subset_n_spectral' : 
        train_split : Optional[Dict[str, Union[float, List[str]]]], optional
            dictionary with values as either a float representing the fraction of samples ot be assigned to each of train, test, 
            and validation OR a list of the sample IDs representing each split, by default 0.8, 0.2, and 0 respectively
        train_seed : int, optional
            seed value, by default mod.seed. By explicitly making this an argument, it allows different train-test splits even 
            with the same mod.seed, e.g., for cross-validation
        track_test : bool, optional
            whether to run the predictions on the test model for at each epoch
        track_validation : bool, optional
            whether to run the predictions on the validation model for at each epoch
        verbose : bool, optional
            whether to print various progress stats across training epochs
        Returns
        -------
        split_data_dict : Dict[str, pd.DataFrame]
            key value pairs represent the output of the `TrainBase.split_data` function
        """
        if track_validation and not train_split['validation']:
            raise ValueError('Specified to track validation statistics, but there is no validation data specified')
        if track_test and not train_split['test']:
            raise ValueError('Specified to track test statistics, but there is no test data specified')
            
        if not mod.signaling_network._prescaled_weights:
            warnings.warn('Recommended to run `self.mod.signaling_network.prescale_weights()` prior to training')

        if not hyper_params:
            self.hyper_params = self.HYPER_PARAMS.copy()
        else:
            self.hyper_params = {k: v for k,v in {**self.HYPER_PARAMS, **hyper_params}.items() if k in self.HYPER_PARAMS}
        
        self.mod = mod

        self.prediction_loss_fn = prediction_loss_fn
        self.prediction_optimizer = prediction_optimizer(self.mod.parameters(), 
                                                 lr=self.hyper_params['maximum_learning_rate'], 
                                                 weight_decay=0)
        self.lr_scheduler = WarmupCosineAnnealingWarmRestarts(optimizer = self.prediction_optimizer,
                                                              T_0 = self.hyper_params['lr_restart_epoch'],
                                                              T_mul = self.hyper_params['lr_restart_factor'], 
                                                              gamma = self.hyper_params['lr_decay'],
                                                              eta_min = self.hyper_params['minimum_learning_rate'],
                                                              max_lr=self.hyper_params['maximum_learning_rate'],
                                                              warmup_steps = self.hyper_params['warmup_epochs'],
                                                              last_epoch = -1)
        self.reset_state = self.prediction_optimizer.state.copy()
        self.track_test = track_test
        self.track_validation = track_validation

 # give user input 

        # self.stats = utils.initialize_progress(hyper_params['max_epochs'])

        # set up data objects
        if not train_seed:
            self.train_seed = self.mod.seed
        else:
             self.train_seed = train_seed

        if isinstance(train_split['train'], float):
            self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val = self.split_data(self.mod.X_in, 
                                                                                                           self.mod.y_out, 
                                                                                                           train_split, 
                                                                                                           train_seed)
        elif isinstance(train_split['train'], list):
            self.X_train = self.mod.X_in.loc[train_split['train'], :]
            self.y_train = self.mod.y_out.loc[train_split['train'], :]
            if 'test' in train_split and train_split['test'] is not None:
                # for storing
                self.X_test = self.mod.X_in.loc[train_split['test'], :]
                self.y_test = self.mod.y_out.loc[train_split['test'], :]
#                 # for running through model
#                 self._X_test = self.mod.df_to_tensor(self.X_test)
#                 self._y_test = self.mod.df_to_tensor(self.y_test)
            if 'validation' in train_split and train_split['validation'] is not None:
                # for storing
                self.X_val = self.mod.X_in.loc[train_split['validation'], :]
                self.y_val = self.mod.y_out.loc[train_split['validation'], :]
#                 # for running through model
#                 self._X_val = self.mod.df_to_tensor(self.X_val)
#                 self._y_val = self.mod.df_to_tensor(self.y_val)
                
        stats_df_cols = ['learning_rate', 'iter_time', 'eig_mean', 'eig_sigma', 'n_moa_violations',
                         'train_loss_with_reg', 'train_loss_mean', 'train_loss_sigma', 
                         'train_pearson_mean', 'train_pearson_sigma']

        if self.track_test: 
            stats_df_cols += ['test_loss_mean', 'test_loss_sigma', 'test_pearson_mean', 'test_pearson_sigma']
        if self.track_validation:
            stats_df_cols += ['validation_loss_mean', 'validation_loss_sigma', 'validation_pearson_mean', 'validation_pearson_sigma']

        self.stats_df = pd.DataFrame(columns = stats_df_cols)

    def create_data_loader(self, include_covariates = True):
        covariates_idx = None

        if include_covariates:
            covariates_idx = self.mod.signaling_network.covariates_to_tensor(sample_ids = self.X_train.index)
        model_data = ModelData(X_in = self.mod.df_to_tensor(self.X_train).to('cpu'), 
                           y_out = self.mod.df_to_tensor(self.y_train).to('cpu'),
                           covariates_idx = covariates_idx)
        self.train_dataloader = DataLoader(dataset=model_data,
                                           batch_size=self.hyper_params['train_batch_size'],
                                           drop_last = False,
                                           pin_memory = False,#pin_memory,
                                           shuffle=True) 
        if self.track_test:
            if include_covariates:
                covariates_idx = self.mod.signaling_network.covariates_to_tensor(sample_ids = self.X_test.index)
            model_data = ModelData(X_in = self.mod.df_to_tensor(self.X_test).to('cpu'), 
                               y_out = self.mod.df_to_tensor(self.y_test).to('cpu'),
                               covariates_idx = covariates_idx)
            self.test_dataloader = DataLoader(dataset=model_data,
                                               batch_size=self.hyper_params['test_batch_size'],
                                               drop_last = False,
                                               pin_memory = False,#pin_memory,
                                               shuffle=False)
        if self.track_validation:
            if include_covariates:
                covariates_idx = self.mod.signaling_network.covariates_to_tensor(sample_ids = self.X_val.index)
            model_data = ModelData(X_in = self.mod.df_to_tensor(self.X_val).to('cpu'), 
                               y_out = self.mod.df_to_tensor(self.y_val).to('cpu'),
                               covariates_idx = covariates_idx)
            self.validation_dataloader = DataLoader(dataset=model_data,
                                               batch_size=self.hyper_params['validation_batch_size'],
                                               drop_last = False,
                                               pin_memory = False,#pin_memory,
                                               shuffle=False)
 
    @staticmethod
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
    
    @staticmethod
    def get_pearson_correlation(tensor_a, tensor_b, axis=0, return_mean=True):
        """Takes the row- or column-wise Pearson correlation between two torch tensors

        Parameters
        ----------
        tensor_a : _type_
            _description_
        tensor_b : _type_
            _description_
        axis : int, optional
            row-wise if 0, column-wise if 1, by default 0
        return_mean : bool, optional
            whether to take the mean across all correlations, by default True
        """
        with torch.no_grad():
            if axis == 1:
                tensor_a = tensor_a.T
                tensor_b = tensor_b.T

    #         correlations = np.array([np.corrcoef(tensor_a[i], tensor_b[i])[0, 1] for i in range(tensor_a.shape[0])])
            correlations = torch.tensor([torch.corrcoef(torch.stack([tensor_a[i], tensor_b[i]]))[0, 1] for i in range(tensor_a.shape[0])])

            if return_mean:
                return torch.nanmean(correlations).item()
            else:
                return correlations


class TrainSimple(TrainBase):
    """Training the signaling model for bulk data with no categorical covariates."""

    def __init__(self, 
                  mod, 
                 prediction_optimizer: torch.optim, 
                 prediction_loss_fn: torch.nn.modules.loss,
                 hyper_params: Dict[str, Union[int, float]] = None,
                 train_split: Optional[Dict[str, Union[float, List[str]]]] = {'train': 0.8, 'test': 0.2, 'validation': None},
                 train_seed: int = None,
                 track_test: bool = False,
                 track_validation: bool = False):
        """See `TrainBase` for parameters."""
        
        if not (type(mod.signaling_network) is BioNetSimple):
            msg = 'You must use the correct training class to match the BioNet class.'
            msg += ' Do you have categorical covariates?'
            raise ValueError(msg)
        super().__init__(mod = mod, 
                           prediction_optimizer = prediction_optimizer, 
                           prediction_loss_fn = prediction_loss_fn, 
                           hyper_params=hyper_params, 
                           train_split=train_split, 
                           train_seed = train_seed, 
                        track_test = track_test, 
                        track_validation = track_validation)
        
        self.create_data_loader(include_covariates = False)
  
    def train_model(self,
                    verbose: bool = True):
        """Train the model

        Parameters
        ----------
        verbose : bool, optional
            print stats during training, by default True

        Returns
        -------
        mod : SignalingModel
            the model with trained parameters
        cur_loss : List[float], optional
            a list of the loss (excluding regularizations) across training iterations
        cur_eig : List[float], optional
            a list of the spectral_radius across training iterations
        """
        start_time = time.time()
        
        #TODELETE:
        with torch.no_grad():
            self.sn_weights = {'mean': [np.mean(self.mod.signaling_network.weights.to('cpu').detach().numpy().flatten())], 
                      'median': [np.median(self.mod.signaling_network.weights.to('cpu').detach().numpy().flatten())], 
                      'std': [np.std(self.mod.signaling_network.weights.to('cpu').detach().numpy().flatten())]}
            self.sn_bias = {'mean': [np.mean(self.mod.signaling_network.bias_basal.to('cpu').detach().numpy().flatten())], 
                          'median': [np.median(self.mod.signaling_network.bias_basal.to('cpu').detach().numpy().flatten())], 
                          'std': [np.std(self.mod.signaling_network.bias_basal.to('cpu').detach().numpy().flatten())]}
        
    
        
        
        for e in trange(self.hyper_params['max_epochs']):
            cur_lr = self.prediction_optimizer.param_groups[0]['lr']
            # set learning rate
#             cur_lr = utils.get_lr(e, self.hyper_params['max_epochs'], max_height = self.hyper_params['maximum_learning_rate'],
#                                 start_height=self.hyper_params['minimum_learning_rate'], end_height=1e-6, peak = self.hyper_params['lr_restart_epoch'])
#             self.prediction_optimizer.param_groups[0]['lr'] = cur_lr
            
            cur_loss = []
            cur_eig = []
            cur_loss_with_reg = []
            cur_pearson = []
            
            # iterate through batches
            if self.mod.seed:
                utils.set_seeds(self.mod.seed + e)
            for batch, (X_in_, y_out_, covariates_idx_) in enumerate(self.train_dataloader):
                self.mod.train()
                self.prediction_optimizer.zero_grad()
        
                X_in_, y_out_, covariates_idx_ = X_in_.to(self.mod.device), y_out_.to(self.mod.device), covariates_idx_.to(self.mod.device)
        
                # forward pass
                X_full = self.mod.input_layer(X_in_) # transform to full network with ligand input concentrations
                utils.set_seeds(self.mod.seed + self.mod._gradient_seed_counter)
                network_noise = torch.randn(X_full.shape, device = X_full.device)
                X_full = X_full + (self.hyper_params['network_noise_scale'] * cur_lr * network_noise) # randomly add noise to signaling network input, makes model more robust
                Y_full = self.mod.signaling_network(X_full) # train signaling network weights
                Y_hat = self.mod.output_layer(Y_full)
                
                # get prediction loss
                fit_loss = self.prediction_loss_fn(y_out_, Y_hat)
                train_pearson_r = self.get_pearson_correlation(y_out_, Y_hat, axis = 0, return_mean = True)
                
                # get regularization losses
                sign_reg = self.mod.signaling_network.sign_regularization(lambda_L1 = self.hyper_params['moa_lambda_L1']) # incorrect MoA
        #             ligand_reg = self.mod.ligand_regularization(lambda_L2 = self.hyper_params['ligand_lambda_L2']) # ligand biases
                stability_loss, spectral_radius = self.mod.signaling_network.get_SS_loss(Y_full = Y_full.detach(), spectral_loss_factor = self.hyper_params['spectral_loss_factor'],
                                                                                    subset_n = self.hyper_params['subset_n_spectral'], n_probes = self.hyper_params['n_probes_spectral'], 
                                                                                    power_steps = self.hyper_params['power_steps_spectral'])
                uniform_reg = self.mod.uniform_regularization(lambda_L2 = self.hyper_params['uniform_lambda_L2']*cur_lr, Y_full = Y_full, 
                                                        target_min = 0, target_max = self.hyper_params['uniform_max']) # uniform distribution
                param_reg = self.mod.L2_reg(input_lambda_L2=self.hyper_params['input_lambda_L2'],
                                            hidden_state_lambda_L2=self.hyper_params['hidden_state_lambda_L2'], 
                                            bias_lambda_L2=self.hyper_params['bias_lambda_L2'], 
                                            output_lambda_L2=self.hyper_params['output_lambda_L2'])
        #             total_loss = fit_loss + sign_reg + ligand_reg + param_reg + stability_loss + uniform_reg
                total_loss = fit_loss + sign_reg + param_reg + stability_loss + uniform_reg
        
                # gradient
                total_loss.backward()
                self.mod.add_gradient_noise(noise_level = self.hyper_params['gradient_noise_scale'])
                self.prediction_optimizer.step()
        
                # store
                cur_eig.append(spectral_radius)
                cur_loss.append(fit_loss.item())
                cur_loss_with_reg.append(total_loss.item())
                cur_pearson.append(train_pearson_r)
                
                # free up CUDA mem
                del sign_reg, stability_loss, uniform_reg, param_reg, fit_loss, train_pearson_r
                del X_in_, y_out_, covariates_idx_, X_full, Y_full, Y_hat
                
            self.lr_scheduler.step()
            
            #TO DELETE:
            with torch.no_grad():
                self.sn_weights['mean'] += [np.mean(self.mod.signaling_network.weights.to('cpu').detach().numpy().flatten())]
                self.sn_weights['median'] += [np.median(self.mod.signaling_network.weights.to('cpu').detach().numpy().flatten())]
                self.sn_weights['std'] += [np.std(self.mod.signaling_network.weights.to('cpu').detach().numpy().flatten())]

                self.sn_bias['mean'] += [np.mean(self.mod.signaling_network.bias_basal.to('cpu').detach().numpy().flatten())]
                self.sn_bias['median'] += [np.median(self.mod.signaling_network.bias_basal.to('cpu').detach().numpy().flatten())]
                self.sn_bias['std'] += [np.std(self.mod.signaling_network.bias_basal.to('cpu').detach().numpy().flatten())]
            
#            cur_lr = lr_scheduler.get_lr()[0]
            # test/validation
            if self.track_validation or self.track_test:
                self.mod.eval()
                with torch.inference_mode(): 
                    if self.track_validation:
                        loss_val_all = []
                        pearson_val_all = []
                        for batch, (X_in_val, y_out_val, covariates_idx_val) in enumerate(self.validation_dataloader): 
                            X_in_val, y_out_val, covariates_idx_val = X_in_val.to(self.mod.device), y_out_val.to(self.mod.device), covariates_idx_val.to(self.mod.device)
                            self.mod.signaling_network.mask = self.mod.signaling_network.mask.to(X_in_val.device)
                            y_pred_val, _ = self.mod(X_in = X_in_val, covariates_idx = covariates_idx_val)
                            loss_val = self.prediction_loss_fn(y_out_val, y_pred_val).detach().item()
                            pearson_val = self.get_pearson_correlation(y_out_val,  y_pred_val)
                            loss_val_all.append(loss_val)
                            pearson_val_all.append(pearson_val)
                            del y_pred_val, _
                    if self.track_test:
                        loss_test_all = []
                        pearson_test_all = []
                        for batch, (X_in_test, y_out_test, covariates_idx_test) in enumerate(self.test_dataloader):
                            X_in_test, y_out_test, covariates_idx_test = X_in_test.to(self.mod.device), y_out_test.to(self.mod.device), covariates_idx_test.to(self.mod.device)
                            y_pred_test, _ = self.mod(X_in = X_in_test, covariates_idx = covariates_idx_test)
                            loss_test = self.prediction_loss_fn(y_out_test, y_pred_test).detach().item()
                            pearson_test = self.get_pearson_correlation(y_out_test, y_pred_test)
                            loss_test_all.append(loss_test)
                            pearson_test_all.append(pearson_test)
                            del y_pred_test, _
        
            # tracking
            sv = [cur_lr, time.time() - start_time, np.mean(cur_eig), np.std(cur_eig),
                  self.mod.signaling_network.count_sign_mismatch(), 
                  np.mean(cur_loss_with_reg), np.mean(cur_loss), np.std(cur_loss), 
                  np.mean(cur_pearson), np.std(cur_pearson)]
            sv += [np.mean(loss_test_all), np.std(loss_test_all), np.mean(pearson_test_all), np.std(pearson_test_all)] if self.track_test else []
            sv += [np.mean(loss_val_all), np.std(loss_val_all), np.mean(pearson_val_all), np.std(pearson_val_all)] if self.track_validation else []
            self.stats_df.loc[e, :] = sv
            
            if e % (self.hyper_params['max_epochs']/100) == 0:
                param_names = []
                for name, param in self.mod.named_parameters():
                    if torch.isnan(param).any():
                        param_names.append(name)
                if len(param_names) > 0:
                    log_error = 'NaN values found in model parameters at epoch {}'.format(e)
                    log_error += ' for layers ' + ', '.join(param_names)
                    logging.error(log_error)
                    raise ValueError('NaN values found in model parameters at epoch {}'.format(e))
                if verbose:
                    utils.print_stats(self.stats_df)
            if np.logical_and(e % self.hyper_params['reset_optimizer_epoch'] == 0, e>0):
                self.prediction_optimizer.state = self.reset_state.copy()
        
        if verbose:
            mins, secs = divmod(time.time() - start_time, 60)
            print("Training ran in: {:.0f} min {:.2f} sec".format(mins, secs))
            
        return self.mod, cur_loss, cur_eig, self.stats_df

class TrainCat(TrainBase):
    """Training the signaling model for bulk data, accounting for categorical covariates of the samples (e.g. cell line, genetic background, etc.)."""
    
    HYPER_PARAMS = TrainBase.HYPER_PARAMS
    
    def __init__(self,
                 mod, 
                 prediction_optimizer: torch.optim, 
                 prediction_loss_fn: torch.nn.modules.loss,
                 hyper_params: Dict[str, Union[int, float]] = None,
                 train_split: Optional[Dict[str, Union[float, List[str]]]] = {'train': 0.8, 'test': 0.2, 'validation': None},
                 train_seed: int = None,
                 track_test: bool = False,
                 track_validation: bool = False
                 ):
        """See `TrainBase` for parameters. Additional parameters include:
        
        Parameters
        ----------
        discriminator_params : Dict
            key word arguments to pass to `CatDiscriminator`
        
        
        """
        if not (type(mod.signaling_network) is BioNetCat):
            raise ValueError('You must use the correct training class to match the BioNet class.')

        super().__init__(mod = mod, 
                           prediction_optimizer = prediction_optimizer, 
                           prediction_loss_fn = prediction_loss_fn, 
                           hyper_params=hyper_params, 
                           train_split=train_split, 
                           train_seed = train_seed, 
                        track_validation = track_validation, 
                        track_test = track_test)

        self.create_data_loader(include_covariates = True)

    def train_model(self, verbose: bool = True):
        """Train the model

        Parameters
        ----------
        verbose : bool, optional
           print stats during trainin, by default True

        Returns
        -------
        mod : SignalingModel
            the model with trained parameters
        cur_loss : List[float], optional
            a list of the loss (excluding regularizations) across training iterations
        cur_eig : List[float], optional
            a list of the spectral_radius across training iterations
       """
        start_time = time.time()
        for e in trange(self.hyper_params['max_epochs']):
            # set learning rate
            cur_lr = self.prediction_optimizer.param_groups[0]['lr']

#             cur_lr = utils.get_lr(e, self.hyper_params['max_epochs'], max_height = self.hyper_params['maximum_learning_rate'],
#                                 start_height=self.hyper_params['minimum_learning_rate'], end_height=1e-6, peak = self.hyper_params['lr_restart_epoch'])
#             self.prediction_optimizer.param_groups[0]['lr'] = cur_lr

            cur_loss = []
            cur_eig = []
            cur_loss_with_reg = []
            cur_pearson = []


            # iterate through batches
            if self.mod.seed:
                utils.set_seeds(self.mod.seed + e)
            for batch, (X_in_, y_out_, covariates_idx_) in enumerate(self.train_dataloader):
                self.mod.train()
                
                self.prediction_optimizer.zero_grad()

                X_in_, y_out_, covariates_idx_ = X_in_.to(self.mod.device), y_out_.to(self.mod.device), covariates_idx_.to(self.mod.device)

                # forward pass
                X_full = self.mod.input_layer(X_in_) # transform to full network with ligand input concentrations
                utils.set_seeds(self.mod.seed + self.mod._gradient_seed_counter)
                network_noise = torch.randn(X_full.shape, device = X_full.device)
                X_full = X_full + (self.hyper_params['network_noise_scale'] * cur_lr * network_noise) # randomly add noise to signaling network input, makes model more robust
                Y_full = self.mod.signaling_network(X_full, covariates_idx_) # train signaling network weights
                Y_hat = self.mod.output_layer(Y_full)

                # get prediction loss
                prediction_loss = self.prediction_loss_fn(y_out_, Y_hat)
                train_pearson_r = self.get_pearson_correlation(y_out_, Y_hat, axis = 0, return_mean = True)

                # regularization
                sign_reg = self.mod.signaling_network.sign_regularization(lambda_L1 = self.hyper_params['moa_lambda_L1']) # incorrect MoA
        #             ligand_reg = self.mod.ligand_regularization(lambda_L2 = self.hyper_params['ligand_lambda_L2']) # ligand biases
                stability_loss, spectral_radius = self.mod.signaling_network.get_SS_loss(Y_full = Y_full.detach(), spectral_loss_factor = self.hyper_params['spectral_loss_factor'],
                                                                                    subset_n = self.hyper_params['subset_n_spectral'], n_probes = self.hyper_params['n_probes_spectral'], 
                                                                                    power_steps = self.hyper_params['power_steps_spectral'])
                uniform_reg = self.mod.uniform_regularization(lambda_L2 = self.hyper_params['uniform_lambda_L2']*cur_lr, Y_full = Y_full, 
                                                        target_min = 0, target_max = self.hyper_params['uniform_max']) # uniform distribution
                param_reg = self.mod.L2_reg(input_lambda_L2=self.hyper_params['input_lambda_L2'],
                                            hidden_state_lambda_L2=self.hyper_params['hidden_state_lambda_L2'], 
                                            bias_lambda_L2=self.hyper_params['bias_lambda_L2'], 
                                            output_lambda_L2=self.hyper_params['output_lambda_L2'])
                tot_pred_loss = prediction_loss + sign_reg + param_reg + stability_loss + uniform_reg
                
                # gradient
                tot_pred_loss.backward()
                self.mod.add_gradient_noise(noise_level = self.hyper_params['gradient_noise_scale'])
                self.prediction_optimizer.step()

                # store
                cur_eig.append(spectral_radius)
                cur_loss.append(prediction_loss.item())
                cur_loss_with_reg.append(tot_pred_loss.item())
                cur_pearson.append(train_pearson_r)
                
                # free up CUDA mem
                del sign_reg, stability_loss, uniform_reg, param_reg, prediction_loss, train_pearson_r
                del X_in_, y_out_, covariates_idx_, X_full, Y_full, Y_hat

            self.lr_scheduler.step()
#            cur_lr = lr_scheduler.get_lr()[0]                
            # test/validation
            if self.track_validation or self.track_test:
                self.mod.eval()
                with torch.inference_mode(): 
                    if self.track_validation:
                        loss_val_all = []
                        pearson_val_all = []
                        for batch, (X_in_val, y_out_val, covariates_idx_val) in enumerate(self.validation_dataloader): 
                            X_in_val, y_out_val, covariates_idx_val = X_in_val.to(self.mod.device), y_out_val.to(self.mod.device), covariates_idx_val.to(self.mod.device)
                            self.mod.signaling_network.mask = self.mod.signaling_network.mask.to(X_in_val.device)
                            y_pred_val, _ = self.mod(X_in = X_in_val, covariates_idx = covariates_idx_val)
                            loss_val = self.prediction_loss_fn(y_out_val, y_pred_val).detach().item()
                            pearson_val = self.get_pearson_correlation(y_out_val,  y_pred_val)
                            loss_val_all.append(loss_val)
                            pearson_val_all.append(pearson_val)
                            del y_pred_val, _
                    if self.track_test:
                        loss_test_all = []
                        pearson_test_all = []
                        for batch, (X_in_test, y_out_test, covariates_idx_test) in enumerate(self.test_dataloader):
                            X_in_test, y_out_test, covariates_idx_test = X_in_test.to(self.mod.device), y_out_test.to(self.mod.device), covariates_idx_test.to(self.mod.device)
                            y_pred_test, _ = self.mod(X_in = X_in_test, covariates_idx = covariates_idx_test)
                            loss_test = self.prediction_loss_fn(y_out_test, y_pred_test).detach().item()
                            pearson_test = self.get_pearson_correlation(y_out_test, y_pred_test)
                            loss_test_all.append(loss_test)
                            pearson_test_all.append(pearson_test)
                            del y_pred_test, _
        
            # tracking
            # tracking
            sv = [cur_lr, time.time() - start_time, np.mean(cur_eig), np.std(cur_eig),
                  self.mod.signaling_network.count_sign_mismatch(), 
                  np.mean(cur_loss_with_reg), np.mean(cur_loss), np.std(cur_loss), 
                  np.mean(cur_pearson), np.std(cur_pearson)]
            sv += [np.mean(loss_test_all), np.std(loss_test_all), np.mean(pearson_test_all), np.std(pearson_test_all)] if self.track_test else []
            sv += [np.mean(loss_val_all), np.std(loss_val_all), np.mean(pearson_val_all), np.std(pearson_val_all)] if self.track_validation else []
            self.stats_df.loc[e, :] = sv

            if e % (self.hyper_params['max_epochs']/100) == 0:
                param_names = []
                for name, param in self.mod.named_parameters():
                    if torch.isnan(param).any():
                        param_names.append(name)
                if len(param_names) > 0:
                    log_error = 'NaN values found in model parameters at epoch {}'.format(e)
                    log_error += ' for layers ' + ', '.join(param_names)
                    logging.error(log_error)
                    raise ValueError('NaN values found in model parameters at epoch {}'.format(e))
                if verbose:
                    utils.print_stats(self.stats_df)

            if np.logical_and(e % self.hyper_params['reset_optimizer_epoch'] == 0, e>0):
                self.prediction_optimizer.state = self.reset_state.copy()

        if verbose:
            mins, secs = divmod(time.time() - start_time, 60)
            print("Training ran in: {:.0f} min {:.2f} sec".format(mins, secs))

        return self.mod, cur_loss, cur_eig, self.stats_df    
    
    

class TrainSC(TrainBase):
    """Training the signaling model for single-cell data."""
    
    HYPER_PARAMS = {**TrainBase.HYPER_PARAMS, **{'discriminator_lambda_L2': 1e-5}}
    
    def __init__(self,
                 mod, 
                 prediction_optimizer: torch.optim, 
                 prediction_loss_fn: torch.nn.modules.loss,
                 discriminator_optimizer: torch.optim,
                 discriminator_params: Dict = CatDiscriminator.DEFAULT_HYPER_PARAMS,
                 hyper_params: Dict[str, Union[int, float]] = None,
                 train_split: Optional[Dict[str, Union[float, List[str]]]] = {'train': 0.8, 'test': 0.2, 'validation': None},
                 train_seed: int = None,
                 track_test: bool = False,
                 track_validation: bool = False
                 ):
        """See `TrainBase` for parameters. Additional parameters include:
        
        Parameters
        ----------
        discriminator_params : Dict
            key word arguments to pass to `CatDiscriminator`
        
        
        """
        if not (type(mod.signaling_network) is BioNetCat):
            raise ValueError('You must use the correct training class to match the BioNet class.')

        super().__init__(mod = mod, 
                           prediction_optimizer = prediction_optimizer, 
                           prediction_loss_fn = prediction_loss_fn, 
                           hyper_params=hyper_params,
                           train_split=train_split, 
                           train_seed = train_seed,
                         track_validation = track_validation,
                         track_test = track_test)

        self.create_data_loader(include_covariates = True)
        self.initialize_discriminator(discriminator_params, discriminator_optimizer)

    def initialize_discriminator(self, discriminator_params, discriminator_optimizer):
        discriminator_params['batch_momentum'] = None # bias is a vector in bulk; this should be eliminated in single-cell
        self.discriminators = nn.ModuleDict(
                        {
                            covariate_cat: CatDiscriminator(n_features_in = cat_embedding.weight.shape[1],
                                              n_labels = cat_embedding.weight.shape[0], 
                                              dtype = self.mod.dtype, 
                                              device = self.mod.device,
                                              **discriminator_params)
                            for covariate_cat, cat_embedding in self.mod.signaling_network.cat_embeddings.items()}
                    )
        
        # prediction optimizer contains the parameters for the the union of the generator and the signaling model
        # note, this combines all discriminator parameters into one optimizer
        # may want to check back into this
        discriminator_params = []
        for discriminator in self.discriminators.values():
            discriminator_params += list(discriminator.parameters())
        self.discriminator_optimizer = discriminator_optimizer(discriminator_params, lr = 1, weight_decay = 0)

    def train_model(self, verbose: bool = True):
        """Train the model

        Parameters
        ----------
        verbose : bool, optional
           print stats during trainin, by default True

        Returns
        -------
        mod : SignalingModel
            the model with trained parameters
        cur_loss : List[float], optional
            a list of the loss (excluding regularizations) across training iterations
        cur_eig : List[float], optional
            a list of the spectral_radius across training iterations
       """
        start_time = time.time()
        for e in trange(self.hyper_params['max_epochs']):
            cur_lr = self.prediction_optimizer.param_groups[0]['lr']

            # set learning rate
#             cur_lr = utils.get_lr(e, self.hyper_params['max_epochs'], max_height = self.hyper_params['maximum_learning_rate'],
#                                 start_height=self.hyper_params['minimum_learning_rate'], end_height=1e-6, peak = self.hyper_params['lr_restart_epoch'])
            self.prediction_optimizer.param_groups[0]['lr'] = cur_lr
            raise ValueError('Need to change the discriminator optimizer with lr scheduler')
            self.discriminator_optimizer.param_groups[0]['lr'] = cur_lr

            cur_loss = []
            cur_eig = []
            cur_loss_with_reg = []
            cur_pearson = []


            # iterate through batches
            if self.mod.seed:
                utils.set_seeds(self.mod.seed + e)
            for batch, (X_in_, y_out_, covariates_idx_) in enumerate(self.train_dataloader):
                self.mod.train()
                
                self.prediction_optimizer.zero_grad()
                self.discriminator_optimizer.zero_grad()

                X_in_, y_out_, covariates_idx_ = X_in_.to(self.mod.device), y_out_.to(self.mod.device), covariates_idx_.to(self.mod.device)

                # forward pass
                X_full = self.mod.input_layer(X_in_) # transform to full network with ligand input concentrations
                utils.set_seeds(self.mod.seed + self.mod._gradient_seed_counter)
                network_noise = torch.randn(X_full.shape, device = X_full.device)
                X_full = X_full + (self.hyper_params['network_noise_scale'] * cur_lr * network_noise) # randomly add noise to signaling network input, makes model more robust
                Y_full = self.mod.signaling_network(X_full, covariates_idx_) # train signaling network weights
                Y_hat = self.mod.output_layer(Y_full)

                # get prediction loss
                prediction_loss = self.prediction_loss_fn(y_out_, Y_hat)
                train_pearson_r = self.get_pearson_correlation(y_out_, Y_hat, axis = 0, return_mean = True)

                # discriminator prediction and loss
                discriminator_loss = torch.tensor([0], device = self.mod.device, dtype = self.mod.dtype)
                for cat_group_idx, (cat, discriminator) in enumerate(self.discriminators.items()):
                    bias_basal_prediction = discriminator(self.mod.signaling_network.bias_basal.T) # predicted logits
                    # TO DO: should this expansion of the vector into samples be done after getting the prediction or 
                    # should it be done prior to 
                    # if prior, should the batch_norm = False line in the initilize_discriminator be removed?
                    bias_basal_prediction = bias_basal_prediction.repeat(covariates_idx_.shape[0], 1) # expand vector to # of samples

                    target = covariates_idx_[:, cat_group_idx]
                    if discriminator.n_labels == 2:
                        target = target.to(self.mod.dtype).unsqueeze(1)

                    discriminator_loss += discriminator.loss_fn(bias_basal_prediction, target)
                    prediction_loss -= discriminator.loss_fn(bias_basal_prediction, target) 

                # regularization - SCL
                sign_reg = self.mod.signaling_network.sign_regularization(lambda_L1 = self.hyper_params['moa_lambda_L1']) # incorrect MoA
        #             ligand_reg = self.mod.ligand_regularization(lambda_L2 = self.hyper_params['ligand_lambda_L2']) # ligand biases
                stability_loss, spectral_radius = self.mod.signaling_network.get_SS_loss(Y_full = Y_full.detach(), spectral_loss_factor = self.hyper_params['spectral_loss_factor'],
                                                                                    subset_n = self.hyper_params['subset_n_spectral'], n_probes = self.hyper_params['n_probes_spectral'], 
                                                                                    power_steps = self.hyper_params['power_steps_spectral'])
                uniform_reg = self.mod.uniform_regularization(lambda_L2 = self.hyper_params['uniform_lambda_L2']*cur_lr, Y_full = Y_full, 
                                                        target_min = 0, target_max = self.hyper_params['uniform_max']) # uniform distribution
                param_reg = self.mod.L2_reg(input_lambda_L2=self.hyper_params['input_lambda_L2'],
                                            hidden_state_lambda_L2=self.hyper_params['hidden_state_lambda_L2'], 
                                            bias_lambda_L2=self.hyper_params['bias_lambda_L2'], 
                                            output_lambda_L2=self.hyper_params['output_lambda_L2'])
                tot_pred_loss = prediction_loss + sign_reg + param_reg + stability_loss + uniform_reg

                # regularization - Discriminator
                for discriminator in self.discriminators.values():
                    discriminator_loss += discriminator.L2_reg(self.hyper_params['discriminator_lambda_L2'])
                
                # gradient
                discriminator_loss.backward(retain_graph = True)
                tot_pred_loss.backward()
                self.mod.add_gradient_noise(noise_level = self.hyper_params['gradient_noise_scale'])
                self.prediction_optimizer.step()
                self.discriminator_optimizer.step()

                # store
                cur_eig.append(spectral_radius)
                cur_loss.append(prediction_loss.item())
                cur_loss_with_reg.append(tot_pred_loss.item())
                cur_pearson.append(train_pearson_r)
                
                # free up CUDA mem
                del sign_reg, stability_loss, uniform_reg, param_reg, prediction_loss, train_pearson_r
                del X_in_, y_out_, covariates_idx_, X_full, Y_full, Y_hat

            self.lr_scheduler.step()
#            cur_lr = lr_scheduler.get_lr()[0]
            # test/validation
            if self.track_validation or self.track_test:
                self.mod.eval()
                with torch.inference_mode(): 
                    if self.track_validation:
                        loss_val_all = []
                        pearson_val_all = []
                        for batch, (X_in_val, y_out_val, covariates_idx_val) in enumerate(self.validation_dataloader): 
                            X_in_val, y_out_val, covariates_idx_val = X_in_val.to(self.mod.device), y_out_val.to(self.mod.device), covariates_idx_val.to(self.mod.device)
                            self.mod.signaling_network.mask = self.mod.signaling_network.mask.to(X_in_val.device)
                            y_pred_val, _ = self.mod(X_in = X_in_val, covariates_idx = covariates_idx_val)
                            loss_val = self.prediction_loss_fn(y_out_val, y_pred_val).detach().item()
                            pearson_val = self.get_pearson_correlation(y_out_val,  y_pred_val)
                            loss_val_all.append(loss_val)
                            pearson_val_all.append(pearson_val)
                            del y_pred_val, _
                    if self.track_test:
                        loss_test_all = []
                        pearson_test_all = []
                        for batch, (X_in_test, y_out_test, covariates_idx_test) in enumerate(self.test_dataloader):
                            X_in_test, y_out_test, covariates_idx_test = X_in_test.to(self.mod.device), y_out_test.to(self.mod.device), covariates_idx_test.to(self.mod.device)
                            y_pred_test, _ = self.mod(X_in = X_in_test, covariates_idx = covariates_idx_test)
                            loss_test = self.prediction_loss_fn(y_out_test, y_pred_test).detach().item()
                            pearson_test = self.get_pearson_correlation(y_out_test, y_pred_test)
                            loss_test_all.append(loss_test)
                            pearson_test_all.append(pearson_test)
                            del y_pred_test, _
        
            # tracking
            # tracking
            sv = [cur_lr, time.time() - start_time, np.mean(cur_eig), np.std(cur_eig),
                  self.mod.signaling_network.count_sign_mismatch(), 
                  np.mean(cur_loss_with_reg), np.mean(cur_loss), np.std(cur_loss), 
                  np.mean(cur_pearson), np.std(cur_pearson)]
            sv += [np.mean(loss_test_all), np.std(loss_test_all), np.mean(pearson_test_all), np.std(pearson_test_all)] if self.track_test else []
            sv += [np.mean(loss_val_all), np.std(loss_val_all), np.mean(pearson_val_all), np.std(pearson_val_all)] if self.track_validation else []
            self.stats_df.loc[e, :] = sv

            if e % (self.hyper_params['max_epochs']/100) == 0:
                param_names = []
                for name, param in self.mod.named_parameters():
                    if torch.isnan(param).any():
                        param_names.append(name)
                if len(param_names) > 0:
                    log_error = 'NaN values found in model parameters at epoch {}'.format(e)
                    log_error += ' for layers ' + ', '.join(param_names)
                    logging.error(log_error)
                    raise ValueError('NaN values found in model parameters at epoch {}'.format(e))
                if verbose:
                    utils.print_stats(self.stats_df)

            if np.logical_and(e % self.hyper_params['reset_optimizer_epoch'] == 0, e>0):
                self.prediction_optimizer.state = self.reset_state.copy()
                self.discriminator_optimizer.state = self.reset_state.copy()

        if verbose:
            mins, secs = divmod(time.time() - start_time, 60)
            print("Training ran in: {:.0f} min {:.2f} sec".format(mins, secs))

        return self.mod, cur_loss, cur_eig, self.stats_df
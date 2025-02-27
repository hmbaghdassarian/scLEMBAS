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
from .bionetwork import BioNetSimple, BioNetCat, BioNetSC
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
    def __init__(self, 
                 X_in: torch.tensor, 
                 y_out: torch.tensor, 
                 covariates_idx: Optional[torch.tensor] = None, 
                expr: Optional[torch.tensor] = None):
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
            expr : torch.tensor
                the gene expression matrix, can be obtained from mod.expr
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
                
            if expr is not None:
                self.expr = expr
            else:
                self.expr = torch.full(self.X_in.shape, torch.nan, device='cpu')

    def __len__(self) -> int:
        "Returns the total number of samples."
        return self.X_in.shape[0]
    def __getitem__(self, idx: int):
        "Returns one sample of data, data and label (X, y)."
        return self.X_in[idx, :], self.y_out[idx, :], self.covariates_idx[idx,:], self.expr[idx,:]
    
    
class TrainBase:
    """Base class for training the signaling model."""

    LR_PARAMS = {'max_epochs': 5000, 'maximum_learning_rate': 2e-3, 'minimum_learning_rate': 2e-4,
                 'lr_restart_epoch': 1000, 'reset_optimizer_epoch': 200, 
                'lr_decay': 0.9, 'lr_restart_factor': 1, 'warmup_epochs': 500}
    OTHER_PARAMS = {'train_batch_size': 512, 'test_batch_size': 512, 'validation_batch_size': 512, 
                    'network_noise_scale': 10, 'gradient_noise_scale': 1e-9}
    REGULARIZATION_PARAMS = {'input_lambda_L2': 1e-6, 'bn_weights_lambda_l2': 1e-6, 'bn_bias_lambda_L2': 1e-6, 
                             'output_weights_lambda_L2': 1e-6,
                             'output_bias_lambda_L2': 1e-6,
                             'moa_lambda_L1': 0.1, #'ligand_lambda_L2': 1e-5, 
                            'uniform_lambda_L2': 1e-4, 'uniform_min': 0, 'uniform_max': (1/1.2), 'spectral_loss_factor': 1e-5, 
                            'vae_lambda_l2': 1e-5, 
                            'vae_scaling_KL': 1e-2}
    SPECTRAL_RADIUS_PARAMS = {'n_probes_spectral': 5, 'power_steps_spectral': 5, 'subset_n_spectral': 5}
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
                - '<param>_lambda_L2' : L2 regularization penalty term for most of the model weights and biases. Note, recommend setting bn_bias_lambda_l2 to 0 when using TrainSC/BioNetSC singce the bias term is regularized by the KL divergence instead. 
                - 'moa_lambda_L1' : L1 regularization penalty term for incorrect interaction mechanism of action (inhibiting/stimulating)
                - 'ligand_lambda_L2' : DEPRECATED, DO NOT ADD KEY/VALUE PAIR. L2 regularization penalty term for ligand biases. 
                - 'uniform_lambda_L2' : L2 regularization penalty term for a uniform distribution fo the RNN adjacency matrix weight values
                - 'uniform_max' : max value of uniform distribution
                - 'uniform_min' : min value of uniform distribution
                - 'vae_lambda_l2': L2 regularization to weights and biases of linear layer of the VAE
                - 'vae_scaling_KL': multiplies the KL divergence by this value to scale it
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
            self.X_train, self.X_test, self.X_validation, self.y_train, self.y_test, self.y_validation = self.split_data(self.mod.X_in, 
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
                self.X_validation = self.mod.X_in.loc[train_split['validation'], :]
                self.y_validation = self.mod.y_out.loc[train_split['validation'], :]
#                 # for running through model
#                 self._X_val = self.mod.df_to_tensor(self.X_val)
#                 self._y_val = self.mod.df_to_tensor(self.y_val)
        self._initialize_tracking()

    def _initialize_tracking(self):
        self._stats_cols = ['epoch', 'batch_index', 
                            'learning_rate', 'iter_time', 'spectral_radius', #'eig_sigma', 
                         'n_moa_violations', 
                         'train_loss_total', 'train_loss_prediction', 'train_pearson', 
                        'sign_reg_loss', 'stability_reg_loss', 'uniform_reg_loss', 
                         'input_param_reg_loss', 'sn_param_reg_loss', 'output_param_reg_loss']        

        self.stats = {}
        self.stats['train'] = np.empty((0, len(self._stats_cols)))

        if self.track_test: 
            self.stats['test'] = np.empty((0, 4))
        if self.track_validation:
            self.stats['validation'] = np.empty((0, 4))

    def create_data_loader(self, include_covariates = True, include_expr = True):
        covariates_idx = None
        expr = None
        for data_type in ['train', 'test', 'validation']:
            if data_type == 'train' or self.__dict__['track_' + data_type]:
                if include_covariates:
                    covariates_idx = self.mod.signaling_network.covariates_to_tensor(sample_ids = self.__dict__['X_' + data_type].index)
                if include_expr:
                    self.__dict__['expr_' + data_type] = self.mod.expr.loc[self.__dict__['X_' + data_type].index, :]
                    expr = self.mod.df_to_tensor(self.__dict__['expr_' + data_type])

                model_data = ModelData(X_in = self.mod.df_to_tensor(self.__dict__['X_' + data_type]).to('cpu'), 
                                       y_out = self.mod.df_to_tensor(self.__dict__['y_' + data_type]).to('cpu'),
                                       covariates_idx = covariates_idx,
                                       expr = expr)
                self.__dict__[data_type + '_dataloader'] = DataLoader(dataset=model_data,
                                                                      batch_size=self.hyper_params[data_type + '_batch_size'],
                                                                      drop_last = False,
                                                                      pin_memory = False,#pin_memory,
                                                                      shuffle=True if data_type == 'train' else False) 
 
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
            X_validation, y_validation = None, None
        else:
            X_train, _X, y_train, _y = train_test_split(X_in, 
                                                            y_out, 
                                                            train_size=train_split_frac['train'],
                                                            random_state=seed)
            X_test, X_validation, y_test, y_validation = train_test_split(_X, 
                                                        _y, 
                                                        train_size=train_split_frac['test']/(train_split_frac['test'] + train_split_frac['validation']),
                                                        random_state=seed)

        return X_train, X_test, X_validation, y_train, y_test, y_validation
    
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
    def print_stats(self, e):
        """Prints various stats of the progress of training the model.

        Parameters
        ----------
        stats : dict
            a dictionary of progress statistics
        iter : int
            the current training iteration
        """
        temp_df = pd.DataFrame(data = self.stats['train'], columns = self._stats_cols)
        temp_df = temp_df[temp_df.epoch == e + 1].mean()

        msg = 'i={:.0f}'.format(e)
        msg += ', l(tr)={:.5f}'.format(temp_df.train_loss_prediction)
        if self.track_test:
            test_df = pd.DataFrame(self.stats['test'], columns = ['epoch', 'batch', 'test_loss_prediction', 'test_pearson'])
            test_df = test_df[test_df.epoch == e + 1].mean()
            msg += ', l(te)={:.5f}'.format(test_df.test_loss_prediction)
        if self.track_validation:
            val_df = pd.DataFrame(self.stats['validation'], columns = ['epoch', 'batch', 'val_loss_prediction', 'val_pearson'])
            val_df = val_df[val_df.epoch == e + 1].mean()
            msg += ', l(v)={:.5f}'.format(val_df.val_loss_prediction)
        msg += ', s={:.5f}'.format(temp_df.spectral_radius)
        msg += ', r={:.5f}'.format(temp_df.learning_rate)
        msg += ', v={:.5f}'.format(temp_df.n_moa_violations)
        print(msg)


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
        
        self.create_data_loader(include_covariates = False, include_expr = False)
  
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
        self.mod.signaling_network.implement_mask()
        for e in trange(self.hyper_params['max_epochs']):
            cur_lr = self.prediction_optimizer.param_groups[0]['lr']
            # set learning rate
#             cur_lr = utils.get_lr(e, self.hyper_params['max_epochs'], max_height = self.hyper_params['maximum_learning_rate'],
#                                 start_height=self.hyper_params['minimum_learning_rate'], end_height=1e-6, peak = self.hyper_params['lr_restart_epoch'])
#             self.prediction_optimizer.param_groups[0]['lr'] = cur_lr
            
 

            # iterate through batches
            if self.mod.seed:
                utils.set_seeds(self.mod.seed + e)
            for batch, (X_in_, y_out_, covariates_idx_, expr_) in enumerate(self.train_dataloader):
                self.mod.train()
                self.prediction_optimizer.zero_grad()
        
                X_in_, y_out_, covariates_idx_ = X_in_.to(self.mod.device), y_out_.to(self.mod.device), covariates_idx_.to(self.mod.device)
        
                # forward pass
                X_full = self.mod.input_layer(X_in_) # transform to full network with ligand input concentrations
                utils.set_seeds(self.mod.seed + self.mod._gradient_seed_counter)
                network_noise = torch.randn(X_full.shape, device = X_full.device)
                X_full = X_full + (self.hyper_params['network_noise_scale'] * cur_lr * network_noise) # randomly add noise to signaling network input, makes model more robust
                Y_full, _ = self.mod.signaling_network(X_full) # train signaling network weights
                Y_hat = self.mod.output_layer(Y_full)
                
                # get prediction loss
                prediction_loss = self.prediction_loss_fn(y_out_, Y_hat)
                train_pearson_r = self.get_pearson_correlation(y_out_, Y_hat, axis = 0, return_mean = True)
                
                # get regularization losses
                sign_reg = self.mod.signaling_network.sign_regularization(lambda_L1 = self.hyper_params['moa_lambda_L1']) # incorrect MoA
        #             ligand_reg = self.mod.ligand_regularization(lambda_L2 = self.hyper_params['ligand_lambda_L2']) # ligand biases
                stability_loss, spectral_radius = self.mod.signaling_network.get_SS_loss(Y_full = Y_full.detach(), spectral_loss_factor = self.hyper_params['spectral_loss_factor'],
                                                                                    subset_n = self.hyper_params['subset_n_spectral'], n_probes = self.hyper_params['n_probes_spectral'], 
                                                                                    power_steps = self.hyper_params['power_steps_spectral'])
                uniform_reg = self.mod.uniform_regularization(lambda_L2 = self.hyper_params['uniform_lambda_L2']*cur_lr, Y_full = Y_full, 
                                                        target_min = self.hyper_params['uniform_min'], target_max = self.hyper_params['uniform_max']) # uniform distribution
                input_param_reg, sn_param_reg, output_param_reg = self.mod.L2_reg(input_lambda_L2=self.hyper_params['input_lambda_L2'],
                                            bn_weights_lambda_l2=self.hyper_params['bn_weights_lambda_l2'], 
                                            bn_bias_lambda_L2=self.hyper_params['bn_bias_lambda_L2'], 
                                            output_weights_lambda_L2=self.hyper_params['output_weights_lambda_L2'],
                                            output_bias_lambda_L2=self.hyper_params['output_bias_lambda_L2'])
                param_reg = input_param_reg + sn_param_reg + output_param_reg
                total_loss = prediction_loss + sign_reg + param_reg + stability_loss + uniform_reg
        
                # gradient
                total_loss.backward()
                self.mod.add_gradient_noise(noise_level = self.hyper_params['gradient_noise_scale'])
                self.prediction_optimizer.step()
                self.mod.signaling_network.implement_mask() # moved out of forward pass to ensure after last backpass these are 0

                sv = np.array([e + 1, batch, cur_lr, time.time() - start_time, spectral_radius, 
                    self.mod.signaling_network.count_sign_mismatch(), 
                    tot_pred_loss.item(), prediction_loss.item(), train_pearson_r, 
                    sign_reg.item(), stability_loss.item(), uniform_reg.item(), 
                    input_param_reg.item(), sn_param_reg.item(), output_param_reg.item()
                    ])
                self.stats['train'] = np.vstack((self.stats['train'], sv))

                # free up CUDA mem
                del sign_reg, stability_loss, uniform_reg, param_reg, fit_loss, train_pearson_r
                del X_in_, y_out_, covariates_idx_, X_full, Y_full, Y_hat
                torch.cuda.empty_cache()
                
            self.lr_scheduler.step()

            # test/validation
            if self.track_validation or self.track_test:
                self.mod.eval()
                with torch.inference_mode(): 
                    if self.track_validation:
                        # loss_val_all = []
                        # pearson_val_all = []
                        for batch, (X_in_, y_out_, covariates_idx_, expr_val) in enumerate(self.validation_dataloader): 
                            X_in_val, y_out_val, covariates_idx_val = X_in_val.to(self.mod.device), y_out_val.to(self.mod.device), covariates_idx_val.to(self.mod.device)
                            self.mod.signaling_network.mask = self.mod.signaling_network.mask.to(X_in_val.device)
                            y_pred_val, _, _ = self.mod(X_in = X_in_val, covariates_idx = covariates_idx_val, expr = expr_val)
                            loss_val = self.prediction_loss_fn(y_out_val, y_pred_val).detach().item()
                            pearson_val = self.get_pearson_correlation(y_out_val,  y_pred_val)
                            # loss_val_all.append(loss_val)
                            # pearson_val_all.append(pearson_val)
                            self.stats['validation'] = np.vstack((self.stats['validation'], np.array([e+1, batch, loss_val,pearson_val])))
                            del y_pred_val, _
                    if self.track_test:
                        # loss_test_all = []
                        # pearson_test_all = []
                        for batch, (X_in_test, y_out_test, covariates_idx_test, expr_test) in enumerate(self.test_dataloader):
                            X_in_test, y_out_test, covariates_idx_test = X_in_test.to(self.mod.device), y_out_test.to(self.mod.device), covariates_idx_test.to(self.mod.device)
                            y_pred_test, _, _ = self.mod(X_in = X_in_test, covariates_idx = covariates_idx_test, expr = expr_test)
                            loss_test = self.prediction_loss_fn(y_out_test, y_pred_test).detach().item()
                            pearson_test = self.get_pearson_correlation(y_out_test, y_pred_test)
                            loss_test_all.append(loss_test)
                            pearson_test_all.append(pearson_test)
                            del y_pred_test, _     

        if self.track_test: 
            stats_df_cols += ['test_loss_prediction', 'test_pearson']
        if self.track_validation:
            stats_df_cols += ['validation_loss_prediction', 'validation_pearson']

            # sv = [cur_lr, time.time() - start_time, np.mean(cur_eig), self.mod.signaling_network.count_sign_mismatch(), 
            #       np.mean(cur_loss_tot_train), np.mean(cur_loss_pred_train), np.mean(cur_pearson_train), 
            #       np.mean(cur_sign_loss), np.mean(cur_stab_loss), np.mean(cur_uni_loss), 
            #       np.mean(cur_in_loss), np.mean(cur_sn_loss), np.mean(cur_out_loss)
            #       ]
            # sv += [np.mean(loss_test_all), np.mean(pearson_test_all)] if self.track_test else []
            # sv += [np.mean(loss_val_all), np.mean(pearson_val_all)] if self.track_validation else []

            # self.stats_df.loc[e, :] = sv
            
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
                    self.print_stats(e)
            if np.logical_and(e % self.hyper_params['reset_optimizer_epoch'] == 0, e>0):
                self.prediction_optimizer.state = self.reset_state.copy()
        
        self.stats['train'] = pd.DataFrame(data = self.stats['train'], columns = self._stats_cols)
        if self.track_test:
            self.stats['test'] = pd.DataFrame(data = self.stats['test'], 
                                            columns = ['epoch', 'batch', 'test_loss_prediction', 'test_pearson'])
        if self.track_validation:
            self.stats['validation'] = pd.DataFrame(data = self.stats['validation'], 
                                            columns = ['epoch', 'batch', 'val_loss_prediction', 'val_pearson'])        
        if verbose:
            mins, secs = divmod(time.time() - start_time, 60)
            print("Training ran in: {:.0f} min {:.2f} sec".format(mins, secs))
            
        return self.mod

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

        self.create_data_loader(include_covariates = True, include_expr = False)

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
        self.mod.signaling_network.implement_mask()
        for e in trange(self.hyper_params['max_epochs']):
            # set learning rate
            cur_lr = self.prediction_optimizer.param_groups[0]['lr']

#             cur_lr = utils.get_lr(e, self.hyper_params['max_epochs'], max_height = self.hyper_params['maximum_learning_rate'],
#                                 start_height=self.hyper_params['minimum_learning_rate'], end_height=1e-6, peak = self.hyper_params['lr_restart_epoch'])
#             self.prediction_optimizer.param_groups[0]['lr'] = cur_lr

            # iterate through batches
            if self.mod.seed:
                utils.set_seeds(self.mod.seed + e)
            for batch, (X_in_, y_out_, covariates_idx_, expr_) in enumerate(self.train_dataloader):
                self.mod.train()
                
                self.prediction_optimizer.zero_grad()

                X_in_, y_out_, covariates_idx_ = X_in_.to(self.mod.device), y_out_.to(self.mod.device), covariates_idx_.to(self.mod.device)

                # forward pass
                X_full = self.mod.input_layer(X_in_) # transform to full network with ligand input concentrations
                utils.set_seeds(self.mod.seed + self.mod._gradient_seed_counter)
                network_noise = torch.randn(X_full.shape, device = X_full.device)
                X_full = X_full + (self.hyper_params['network_noise_scale'] * cur_lr * network_noise) # randomly add noise to signaling network input, makes model more robust
                Y_full, _ = self.mod.signaling_network(X_full, covariates_idx_) # train signaling network weights
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
                input_param_reg, sn_param_reg, output_param_reg = self.mod.L2_reg(input_lambda_L2=self.hyper_params['input_lambda_L2'],
                                            bn_weights_lambda_l2=self.hyper_params['bn_weights_lambda_l2'], 
                                            bn_bias_lambda_L2=self.hyper_params['bn_bias_lambda_L2'], 
                                            output_weights_lambda_L2=self.hyper_params['output_weights_lambda_L2'],
                                            output_bias_lambda_L2=self.hyper_params['output_bias_lambda_L2'])
                param_reg = input_param_reg + sn_param_reg + output_param_reg
                tot_pred_loss = prediction_loss + sign_reg + param_reg + stability_loss + uniform_reg
                
                # gradient
                tot_pred_loss.backward()
                self.mod.add_gradient_noise(noise_level = self.hyper_params['gradient_noise_scale'])
                self.prediction_optimizer.step()
                self.mod.signaling_network.implement_mask() # moved out of forward pass to ensure after last backpass these are 0

                sv = np.array([e + 1, batch, cur_lr, time.time() - start_time, spectral_radius, 
                    self.mod.signaling_network.count_sign_mismatch(), 
                    tot_pred_loss.item(), prediction_loss.item(), train_pearson_r, 
                    sign_reg.item(), stability_loss.item(), uniform_reg.item(), 
                    input_param_reg.item(), sn_param_reg.item(), output_param_reg.item()
                    ])
                self.stats['train'] = np.vstack((self.stats['train'], sv))

                # free up CUDA mem
                del sign_reg, stability_loss, uniform_reg, param_reg, prediction_loss, train_pearson_r
                del input_param_reg, sn_param_reg, output_param_reg
                del X_in_, y_out_, covariates_idx_, X_full, Y_full, Y_hat
                torch.cuda.empty_cache()

            self.lr_scheduler.step()
#            cur_lr = lr_scheduler.get_lr()[0]                
            # test/validation
            if self.track_validation or self.track_test:
                self.mod.eval()
                with torch.inference_mode(): 
                    if self.track_validation:
                        # loss_val_all = []
                        # pearson_val_all = []
                        for batch, (X_in_val, y_out_val, covariates_idx_val, expr_val) in enumerate(self.validation_dataloader): 
                            X_in_val, y_out_val, covariates_idx_val = X_in_val.to(self.mod.device), y_out_val.to(self.mod.device), covariates_idx_val.to(self.mod.device)
                            self.mod.signaling_network.mask = self.mod.signaling_network.mask.to(X_in_val.device)
                            y_pred_val, _, _ = self.mod(X_in = X_in_val, covariates_idx = covariates_idx_val, expr = expr_val)
                            loss_val = self.prediction_loss_fn(y_out_val, y_pred_val).detach().item()
                            pearson_val = self.get_pearson_correlation(y_out_val,  y_pred_val)
                            # loss_val_all.append(loss_val)
                            # pearson_val_all.append(pearson_val)
                            self.stats['validation'] = np.vstack((self.stats['validation'], np.array([e+1, batch, loss_val,pearson_val])))
                            del y_pred_val, _
                    if self.track_test:
                        # loss_test_all = []
                        # pearson_test_all = []
                        for batch, (X_in_test, y_out_test, covariates_idx_test, expr_test) in enumerate(self.test_dataloader):
                            X_in_test, y_out_test, covariates_idx_test = X_in_test.to(self.mod.device), y_out_test.to(self.mod.device), covariates_idx_test.to(self.mod.device)
                            y_pred_test, _, _ = self.mod(X_in = X_in_test, covariates_idx = covariates_idx_test, expr = expr_test)
                            loss_test = self.prediction_loss_fn(y_out_test, y_pred_test).detach().item()
                            pearson_test = self.get_pearson_correlation(y_out_test, y_pred_test)
                            loss_test_all.append(loss_test)
                            pearson_test_all.append(pearson_test)
                            del y_pred_test, _
        
            # tracking
            # tracking
            # sv = [cur_lr, time.time() - start_time, np.mean(cur_eig), self.mod.signaling_network.count_sign_mismatch(), 
            #       np.mean(cur_loss_tot_train), np.mean(cur_loss_pred_train), np.mean(cur_pearson_train), 
            #       np.mean(cur_sign_loss), np.mean(cur_stab_loss), np.mean(cur_uni_loss), 
            #       np.mean(cur_in_loss), np.mean(cur_sn_loss), np.mean(cur_out_loss)
            #       ]
            # sv += [np.mean(loss_test_all), np.mean(pearson_test_all)] if self.track_test else []
            # sv += [np.mean(loss_val_all), np.mean(pearson_val_all)] if self.track_validation else []
            # self.stats_df.loc[e, :] = sv

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
                    self.print_stats(e)

            if np.logical_and(e % self.hyper_params['reset_optimizer_epoch'] == 0, e>0):
                self.prediction_optimizer.state = self.reset_state.copy()

        self.stats['train'] = pd.DataFrame(data = self.stats['train'], columns = self._stats_cols)
        if self.track_test:
            self.stats['test'] = pd.DataFrame(data = self.stats['test'], 
                                            columns = ['epoch', 'batch', 'test_loss_prediction', 'test_pearson'])
        if self.track_validation:
            self.stats['validation'] = pd.DataFrame(data = self.stats['validation'], 
                                            columns = ['epoch', 'batch', 'val_loss_prediction', 'val_pearson'])        
        if verbose:
            mins, secs = divmod(time.time() - start_time, 60)
            print("Training ran in: {:.0f} min {:.2f} sec".format(mins, secs))

        return self.mod    
    
    

class TrainSC(TrainBase):
    """Training the signaling model for single-cell data."""

    DISCRIMINATOR_PARAMS = {**CatDiscriminator.DEFAULT_HYPER_PARAMS, 
                             **{k: v for k,v in TrainBase.LR_PARAMS.items() if k != 'max_epochs'},
                             **{'optimizer': torch.optim.Adam,
                                'discriminator_lambda_L2': 1e-5, 
                               'discriminator_penalty_weight': 1}}
    HYPER_PARAMS = TrainBase.HYPER_PARAMS
    HYPER_PARAMS['bn_bias_lambda_L2'] = 0 # KL divergence regularization deals with this
    
    def __init__(self,
                 mod, 
                 prediction_optimizer: torch.optim, 
                 prediction_loss_fn: torch.nn.modules.loss,
                 discriminator_params: Dict = None,
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
            key word arguments to pass to `CatDiscriminator`. see TrainSC.DISCRIMINATOR_PARAMS for defaults
            as well as:
                'optimizer': the optimizer to use
                'discriminator_labmda_L2': L2 regularization penalty for discriminator parameters
                'discriminator_penalty_weight': float | List[float]
                    scales the dscriminator loss to incorporate into the adverserial training this can either be a float, 
                    or a list of floats, where each element corresponds to the scaling for that epoch
        
        
        """
#         if not (type(mod.signaling_network) is BioNetSC):
#             raise ValueError('You must use the correct training class to match the BioNet class.')

        super().__init__(mod = mod, 
                           prediction_optimizer = prediction_optimizer, 
                           prediction_loss_fn = prediction_loss_fn, 
                           hyper_params=hyper_params,
                           train_split=train_split, 
                           train_seed = train_seed,
                         track_validation = track_validation,
                         track_test = track_test)
        
        self.create_data_loader(include_covariates = True, include_expr = True)
        self.initialize_discriminator(discriminator_params)
        
    def _initialize_tracking(self):
        self._stats_cols = ['epoch', 'batch_index', 
                            'learning_rate', 'discriminator_learning_rate', 'iter_time', 'spectral_radius', #'eig_sigma', 
                         'n_moa_violations', 
                         'train_loss_total', 'train_loss_prediction', 
                        'sign_reg_loss', 'stability_reg_loss', 'uniform_reg_loss', 
                         'input_param_reg_loss', 'sn_param_reg_loss', 'output_param_reg_loss', 
                          'vae_param_reg_loss', 'kl_divergence', 'adverserial_loss','discriminator_loss_total', 
                                'discriminator_loss_prediction', 'discriminator_param_reg_loss' ]        

        self.stats = {}
        self.stats['train'] = np.empty((0, len(self._stats_cols)))

        if self.track_test: 
            self.stats['test'] = np.empty((0, 3))
        if self.track_validation:
            self.stats['validation'] = np.empty((0, 3))

    def initialize_discriminator(self, discriminator_params):
        # self.discriminator['params']['batch_momentum'] = None # bias is a vector in bulk; this should be eliminated in single-cell
        self.discriminator = {}
        self.discriminator['params'] = update_with_defaults(self.DISCRIMINATOR_PARAMS, discriminator_params)
        if isinstance(self.discriminator['params']['discriminator_penalty_weight'], float):
            self.discriminator['params']['discriminator_penalty_weight'] = [self.discriminator['params']['discriminator_penalty_weight']]*self.hyper_params['max_epochs']
        elif isinstance(self.discriminator['params']['discriminator_penalty_weight'], list):
            if len(self.discriminator['params']['discriminator_penalty_weight']) != self.hyper_params['max_epochs']:
                raise ValueError('Must specify a discriminator penalty weight for each epoch')
        else:
            raise ValueError("'discriminator_penalty_weight' must be a float or list")
        
        self.discriminator['discriminators'] = nn.ModuleDict(
                            {
                                covariate_cat: CatDiscriminator(n_features_in = cat_embedding.weight.shape[1],
                                                  n_labels = cat_embedding.weight.shape[0], 
                                                  dtype = self.mod.dtype, 
                                                  device = self.mod.device,
                                                                batch_momentum = self.discriminator['params']['batch_momentum'], 
                                                                layer_norm = self.discriminator['params']['layer_norm'], 
                                                               dropout_rate = self.discriminator['params']['dropout_rate'], 
                                                               activation_fn = self.discriminator['params']['activation_fn'], 
                                                               n_hidden_nodes = self.discriminator['params']['n_hidden_nodes'])
                                for covariate_cat, cat_embedding in self.mod.signaling_network.cat_embeddings.items()}
                        )

        # NOTE: combines all discriminators into one optimizer; in the future, may want to separate or create a multi-task classifier
        discriminator_mod_params = []
        for discriminator in self.discriminator['discriminators'].values():
            discriminator_mod_params += list(discriminator.parameters())

        self.discriminator['optimizer'] = self.discriminator['params']['optimizer'](discriminator_mod_params, 
                                                                            lr = self.discriminator['params']['maximum_learning_rate'],
                                                                            weight_decay = 0)
        self.discriminator['lr_scheduler'] = WarmupCosineAnnealingWarmRestarts(optimizer = self.discriminator['optimizer'], 
                                                                               T_0 = self.discriminator['params']['lr_restart_epoch'],
                                                              T_mul = self.discriminator['params']['lr_restart_factor'], 
                                                              gamma = self.discriminator['params']['lr_decay'],
                                                              eta_min = self.discriminator['params']['minimum_learning_rate'],
                                                              max_lr=self.discriminator['params']['maximum_learning_rate'],
                                                              warmup_steps = self.discriminator['params']['warmup_epochs'],
                                                              last_epoch = -1)



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
        self.mod.signaling_network.implement_mask()
        for e in trange(self.hyper_params['max_epochs']):
            cur_lr = self.prediction_optimizer.param_groups[0]['lr']
            self.discriminator['_cur_lr'] = self.discriminator['optimizer'].param_groups[0]['lr']
            cur_disc_lambda = self.discriminator['params']['discriminator_penalty_weight'][e]
 
            cur_vae_loss, cur_kl_loss, disc_loss_tot_train, disc_loss_pred_train, disc_param_loss = [], [], [], [], []

            # iterate through batches
            if self.mod.seed:
                utils.set_seeds(self.mod.seed + e)
            for batch, (X_in_, y_out_, covariates_idx_, expr_) in enumerate(self.train_dataloader):
                self.mod.train()
                for mod_discriminator in self.discriminator['discriminators'].values():
                    mod_discriminator.train()
                
                self.prediction_optimizer.zero_grad()
                self.discriminator['optimizer'].zero_grad()

                X_in_, y_out_, covariates_idx_ = X_in_.to(self.mod.device), y_out_.to(self.mod.device), covariates_idx_.to(self.mod.device)

                # forward pass
                X_full = self.mod.input_layer(X_in_) # transform to full network with ligand input concentrations
                utils.set_seeds(self.mod.seed + self.mod._gradient_seed_counter)
                network_noise = torch.randn(X_full.shape, device = X_full.device)
                X_full = X_full + (self.hyper_params['network_noise_scale'] * cur_lr * network_noise) # randomly add noise to signaling network input, makes model more robust
                Y_full, bias_terms = self.mod.signaling_network(X_full = X_full, 
                                                                 covariates_idx = covariates_idx_, 
                                                                 expr = expr_) # train signaling network weights
                bias_global, bias_mu, bias_log_sigma_squared = bias_terms
                
                Y_hat = self.mod.output_layer(Y_full)

                ############ DISCRIMINATOR ############
                # discriminator prediction and loss
                discriminator_loss_accuracy = torch.tensor(0, device = self.mod.device, dtype = self.mod.dtype)
                for cat_group_idx, (cat, discriminator) in enumerate(self.discriminator['discriminators'].items()):
                    bias_global_prediction = discriminator(bias_global) # predicted logits

                    target = covariates_idx_[:, cat_group_idx]
                    if discriminator.n_labels == 2:
                        target = target.to(self.mod.dtype).unsqueeze(1)

                    discriminator_loss_accuracy += discriminator.loss_fn(bias_global_prediction, target)   # if don't use retain_graph = True, then use bias_global_prediction.detach() here
#                     prediction_loss -= discriminator.loss_fn(bias_global_prediction, target) 

                # discriminator regularization
                discriminator_reg = torch.tensor(0, device = self.mod.device, dtype = self.mod.dtype)
                for discriminator in self.discriminator['discriminators'].values():
                    discriminator_reg += discriminator.L2_reg(self.discriminator['params']['discriminator_lambda_L2'])
                discriminator_loss = discriminator_loss_accuracy + discriminator_reg
                
                # discriminator optimization
                # NOTE: discriminator is optimized prior to advererial training (and loss re-calculated)
                # TODO: need to reset optimizer? or anything
                discriminator_loss.backward(retain_graph = True)
                self.discriminator['optimizer'].step()

                ############ LEMBAS and generator ############
                # reconstruction loss
                prediction_loss = self.prediction_loss_fn(y_out_, Y_hat)

                # lembas regularization
                sign_reg = self.mod.signaling_network.sign_regularization(lambda_L1 = self.hyper_params['moa_lambda_L1']) # incorrect MoA
        #             ligand_reg = self.mod.ligand_regularization(lambda_L2 = self.hyper_params['ligand_lambda_L2']) # ligand biases
                stability_loss, spectral_radius = self.mod.signaling_network.get_SS_loss(Y_full = Y_full.detach(), spectral_loss_factor = self.hyper_params['spectral_loss_factor'],
                                                                                    subset_n = self.hyper_params['subset_n_spectral'], n_probes = self.hyper_params['n_probes_spectral'], 
                                                                                    power_steps = self.hyper_params['power_steps_spectral'])
                uniform_reg = self.mod.uniform_regularization(lambda_L2 = self.hyper_params['uniform_lambda_L2']*cur_lr, Y_full = Y_full, 
                                                        target_min = 0, target_max = self.hyper_params['uniform_max']) # uniform distribution
                input_param_reg, sn_param_reg, output_param_reg = self.mod.L2_reg(input_lambda_L2=self.hyper_params['input_lambda_L2'],
                                            bn_weights_lambda_l2=self.hyper_params['bn_weights_lambda_l2'], 
                                            bn_bias_lambda_L2=self.hyper_params['bn_bias_lambda_L2'], 
                                            bias_global = bias_global,
                                            output_weights_lambda_L2=self.hyper_params['output_weights_lambda_L2'],
                                            output_bias_lambda_L2=self.hyper_params['output_bias_lambda_L2'])
                param_reg = input_param_reg + sn_param_reg + output_param_reg
                vae_reg = self.mod.signaling_network.vae.L2_reg(lambda_L2=self.hyper_params['vae_lambda_l2']) # VAE loss
                param_reg += vae_reg
                   
                # NOTE: KL divergence is scaled to match loss magnitudes; no bias regularization given KL regularization
                # can use MMD in the future if KL unstable
                kl_divergence = self.mod.signaling_network.vae.KL_divergence(z_mu = bias_mu, 
                                                                             z_log_sigma_squared = bias_log_sigma_squared, 
                                                                             scaling_factor = self.hyper_params['vae_scaling_KL'])
                
                tot_pred_loss = prediction_loss + sign_reg + param_reg + stability_loss + uniform_reg + kl_divergence

                # adverserial portion -- same as discriminator, but recalculating on trained model
                # TODO: does discriminator need to be in .eval/.inference mode?
                adverserial_loss = torch.tensor(0, device = self.mod.device, dtype = self.mod.dtype)
                for cat_group_idx, (cat, discriminator) in enumerate(self.discriminator['discriminators'].items()):
                    bias_global_prediction = discriminator(bias_global) # predicted logits

                    target = covariates_idx_[:, cat_group_idx]
                    if discriminator.n_labels == 2:
                        target = target.to(self.mod.dtype).unsqueeze(1)

                    adverserial_loss += discriminator.loss_fn(bias_global_prediction, target)   # if don't use retain_graph = True, then use bias_global_prediction.detach() here
#                     prediction_loss -= discriminator.loss_fn(bias_global_prediction, target) 
                tot_pred_loss -= (cur_disc_lambda*adverserial_loss) # adversarial portion

                # gradient
                tot_pred_loss.backward()
                self.mod.add_gradient_noise(noise_level = self.hyper_params['gradient_noise_scale'])
                self.prediction_optimizer.step()
                self.mod.signaling_network.implement_mask() # moved out of forward pass to ensure after last backpass these are 0
                # bias global masking can stay in forward pass because it is generated during the forward pass and 
                # won't be updated in the back pass


                sv = np.array([e + 1, batch, cur_lr, self.discriminator['_cur_lr'], 
                               time.time() - start_time, spectral_radius, 
                    self.mod.signaling_network.count_sign_mismatch(), 
                    tot_pred_loss.item(), prediction_loss.item(), #train_pearson_r, 
                    sign_reg.item(), stability_loss.item(), uniform_reg.item(), 
                    input_param_reg.item(), sn_param_reg.item(), output_param_reg.item(),
                    vae_reg.item(), kl_divergence.item(), 
                               cur_disc_lambda*adverserial_loss.item(), discriminator_loss.item(), discriminator_loss_accuracy.item(), discriminator_reg.item()])
                self.stats['train'] = np.vstack((self.stats['train'], sv))
                
                # free up CUDA mem
                del sign_reg, stability_loss, uniform_reg, param_reg, prediction_loss
                del input_param_reg, sn_param_reg, output_param_reg
                del vae_reg, kl_divergence, discriminator_loss, discriminator_loss_accuracy, discriminator_reg
                del X_in_, y_out_, covariates_idx_, X_full, Y_full, Y_hat
                torch.cuda.empty_cache()

            self.lr_scheduler.step()
            self.discriminator['lr_scheduler'].step()

            # test/validation
            if self.track_validation or self.track_test:
                self.mod.eval()
                with torch.inference_mode(): 
                    if self.track_validation:
                        # loss_val_all = []
                        # pearson_val_all = []
                        for batch, (X_in_val, y_out_val, covariates_idx_val, expr_val) in enumerate(self.validation_dataloader): 
                            X_in_val, y_out_val, covariates_idx_val = X_in_val.to(self.mod.device), y_out_val.to(self.mod.device), covariates_idx_val.to(self.mod.device)
                            self.mod.signaling_network.mask = self.mod.signaling_network.mask.to(X_in_val.device)
                            y_pred_val, _, _ = self.mod(X_in = X_in_val, covariates_idx = covariates_idx_val, expr = expr_val)
                            loss_val = self.prediction_loss_fn(y_out_val, y_pred_val).detach().item()
                            # loss_val_all.append(loss_val)
                            # pearson_val_all.append(pearson_val)
                            self.stats['validation'] = np.vstack((self.stats['validation'], np.array([e+1, batch, loss_val])))
                            del y_pred_val, _
                    if self.track_test:
                        # loss_test_all = []
                        # pearson_test_all = []
                        for batch, (X_in_test, y_out_test, covariates_idx_test, expr_test) in enumerate(self.test_dataloader):
                            X_in_test, y_out_test, covariates_idx_test = X_in_test.to(self.mod.device), y_out_test.to(self.mod.device), covariates_idx_test.to(self.mod.device)
                            y_pred_test, _, _ = self.mod(X_in = X_in_test, covariates_idx = covariates_idx_test, expr = expr_test)
                            loss_test = self.prediction_loss_fn(y_out_test, y_pred_test).detach().item()
                            # loss_test_all.append(loss_test)
                            self.stats['test'] = np.vstack((self.stats['test'], np.array([e +1, batch, loss_test])))
                            del y_pred_test, _

            if e % (self.hyper_params['max_epochs']/100) == 0:
                param_names = []
                for name, param in self.mod.named_parameters():
                    if torch.isnan(param).any():
                        param_names.append(name)
                if len(param_names) > 0:
                    log_error = 'NaN values found in model parameters at epoch {}'.format(e)
                    log_error += ' for layers ' + ', '.join(param_names)
                    logging.error(log_error)
                    raise ValueError(log_error)
                if verbose:
                    self.print_stats(e)

            if np.logical_and(e % self.hyper_params['reset_optimizer_epoch'] == 0, e>0):
                self.prediction_optimizer.state = self.reset_state.copy()
            if (e % self.discriminator['params']['reset_optimizer_epoch'] == 0) and e > 0:
                self.discriminator['optimizer'].state = self.reset_state.copy()
        
        # format the tracking metrics
        self.stats['train'] = pd.DataFrame(data = self.stats['train'], columns = self._stats_cols)
        
        # sanity check
        loss_cols = [
               'train_loss_prediction', 'sign_reg_loss',
               'stability_reg_loss', 'uniform_reg_loss', 'input_param_reg_loss',
               'sn_param_reg_loss', 'output_param_reg_loss', 'vae_param_reg_loss', 'kl_divergence']

        reg_tot = self.stats['train'][loss_cols].sum(axis = 1)
        if not np.allclose(reg_tot - self.stats['train']['adverserial_loss'], self.stats['train']['train_loss_total']):
            warnings.warn('Training loss tracking is incorrect')
        
        if not np.allclose(self.stats['train']['discriminator_loss_total'], 
                           self.stats['train'][['discriminator_loss_prediction', 
                                                'discriminator_param_reg_loss']].sum(axis = 1)):
            warnings.warn('Discriminator loss tracking is incorrect')
        
        
        if self.track_test:
            self.stats['test'] = pd.DataFrame(data = self.stats['test'], 
                                            columns = ['epoch', 'batch', 'test_loss_prediction'])

        if self.track_validation:
            self.stats['validation'] = pd.DataFrame(data = self.stats['validation'], 
                                            columns = ['epoch', 'batch', 'val_loss_prediction'])
        if verbose:
            mins, secs = divmod(time.time() - start_time, 60)
            print("Training ran in: {:.0f} min {:.2f} sec".format(mins, secs))

        return self.mod

    def print_stats(self, e):
        """Prints various stats of the progress of training the model.

        Parameters
        ----------
        stats : dict
            a dictionary of progress statistics
        iter : int
            the current training iteration
        """
        temp_df = pd.DataFrame(data = self.stats['train'], columns = self._stats_cols)
        temp_df = temp_df[temp_df.epoch == e + 1].mean()

        msg = 'i={:.0f}'.format(e)
        msg += ', l(tr)={:.5f}'.format(temp_df.train_loss_prediction)
        if self.track_test:
            test_df = pd.DataFrame(self.stats['test'], columns = ['epoch', 'batch', 'test_loss_prediction'])
            test_df = test_df[test_df.epoch == e + 1].mean()
            msg += ', l(te)={:.5f}'.format(test_df.test_loss_prediction)
        if self.track_validation:
            val_df = pd.DataFrame(self.stats['validation'], columns = ['epoch', 'batch', 'val_loss_prediction'])
            val_df = val_df[val_df.epoch == e + 1].mean()
            msg += ', l(v)={:.5f}'.format(val_df.val_loss_prediction)
        msg += ', s={:.5f}'.format(temp_df.spectral_radius)
        msg += ', r={:.5f}'.format(temp_df.learning_rate)
        msg += ', v={:.5f}'.format(temp_df.n_moa_violations)
        print(msg)
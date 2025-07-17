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
from torch.nn import MSELoss
from geomloss import SamplesLoss

import scLEMBAS.utilities as utils
from .model_utilities import update_with_defaults, kl_divergence_normal, freeze_model, unfreeze_model
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
                 'lr_restart_epoch': 1000, 'n_optimizer_resets': 0, 
                'lr_decay': 0.9, 'lr_restart_factor': 1, 'warmup_epochs': 500}
    BATCH_PARAMS = {'train_batch_size': 512, 'test_batch_size': 512, 'validation_batch_size': 512}
    NOISE_PARAMS = {'network_noise_scale': 0.01, # adjust according to projection_amplitude_in, this assumes default projection_amplitude_in = 3
                   'min_network_noise': 0.0025, # 1/4 of the network noise_scaler (in case LR range is larger than 4x, which by default it is 10x)
                   'gradient_noise_scale': 1e-9}
    REGULARIZATION_PARAMS = {'input_lambda_L2': 0, # assuming frozen weights
                             'bn_weights_lambda_L2': 1e-6,  
                             'output_weights_lambda_L2': 1e-6,
                             'output_bias_lambda_L2': 1e-6,
                             'moa_lambda_L1': 0.1, #'ligand_lambda_L2': 1e-5, 
                            'uniform_lambda_L2': 1e-4, 'uniform_min': 0, 'uniform_max': (1/1.2), 'spectral_loss_factor': 1e-5, 
                            'adj_scaling_KL': 0, 'adj_prior_mu': 0, 'adj_prior_sigma': 0.2
                            }
    SPECTRAL_RADIUS_PARAMS = {'n_probes_spectral': 5, 'power_steps_spectral': 5, 'subset_n_spectral': 5}
    HYPER_PARAMS = {**LR_PARAMS, 
                    **BATCH_PARAMS, 
                    **NOISE_PARAMS, 
                    **REGULARIZATION_PARAMS, 
                    **SPECTRAL_RADIUS_PARAMS}

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
                - 'network_noise_scale' : noise added to signaling network input, by default 0.01. Value should change in accordance to `projections_amplitude_in` argument used for the `SignalingModel`. Noise scale is network_noise_scale * cur_lr Set to 0 for no noise. Makes model more robust. 
                - 'min_network_noise' : minimum noise to add per epoch (since a function of LR), by default 0.0025
                - 'gradient_noise_scale' : noise added to gradient after backward pass. Makes model more robust. 
                - 'reset_epoch' : number of epochs upon which to reset the optimizer state, by default 200
                - '<param>_lambda_L2' : L2 regularization penalty term for most of the model weights and biases. Note, recommend setting bn_bias_lambda_l2 to 0 when using TrainSC/BioNetSC singce the bias term is regularized by the KL divergence instead. 
                - 'moa_lambda_L1' : L1 regularization penalty term for incorrect interaction mechanism of action (inhibiting/stimulating)
                - 'ligand_lambda_L2' : DEPRECATED, DO NOT ADD KEY/VALUE PAIR. L2 regularization penalty term for ligand biases. 
                - 'uniform_lambda_L2' : L2 regularization penalty term for a uniform distribution for the node activity output
                - 'uniform_max' : max value of uniform distribution
                - 'uniform_min' : min value of uniform distribution
                - 'vae_lambda_l2': L2 regularization to weights and biases of linear layer of the VAE
                - 'vae_scaling_KL': multiplies the KL divergence by this value to scale it
                - 'spectral_loss_factor' : regularization penalty term for 
                - 'n_probes_spectral' : 
                - 'power_steps_spectral' : 
                - 'subset_n_spectral' : 
                - Implementation of a KL divergence regularization between unmasked bionetwork weights and a target normal distribution
                    - 'adj_scaling_KL': scaling of the KL divergence value (multiplicative factor)
                    - 'adj_prior_mu': target normal distribution mean
                    - 'adj_prior_sigma': target normal distribution standard deviation
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

        self.mod = mod
        
        if not hyper_params:
            self.hyper_params = self.HYPER_PARAMS.copy()
        else:
            self.hyper_params = {k: v for k,v in {**self.HYPER_PARAMS, **hyper_params}.items() if k in self.HYPER_PARAMS}

        self.hyper_params['reset_optimizer_epoch'] = self.hyper_params['max_epochs'] // (self.hyper_params['n_optimizer_resets'] + 1) #if self.hyper_params['n_optimizer_resets'] > 0 else torch.inf
        
        for param, param_value in self.hyper_params.items():
            if 'lambda' in param and not torch.is_tensor(param_value):
                    self.hyper_params[param] = torch.tensor(param_value, device=self.mod.device, dtype=self.mod.dtype)   

        if hasattr(self.mod.signaling_network, 'vae'):
            vae_params = set(p for p in self.mod.signaling_network.vae.parameters())
            non_vae_params = [p for p in self.mod.parameters() if p not in vae_params]
        else:
            non_vae_params = list(self.mod.parameters())
        self.prediction_loss_fn = prediction_loss_fn
        self.prediction_optimizer = prediction_optimizer(non_vae_params, 
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
                            'input_param_reg_loss',
                            'sn_param_reg_weights_kl_divergence',
                            'sn_param_reg_weights_L2_loss', 
                            'sn_param_reg_bias_L2_loss', 'sn_param_reg_bias_L1_loss',
                            'output_param_reg_weights_loss', 'output_param_reg_bias_loss']      
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
 
    @staticmethod
    def make_weights(counts, mode="sqrt", beta=0.995):
        """Returns a vector of weights proportional to counts"""
        if mode == "uniform": # standard average -- rare cells may be overweighted
            w = torch.ones_like(counts, dtype=torch.float)
        elif mode == "size": # weight every cell equally -- rare cells may underperofrm
            w = counts.float()
        elif mode == "sqrt": # balances rare and abundant cells
            w = torch.sqrt(counts.float())
        elif mode == "effnum":  # balances rare and abundant cells -- as beta goes to 1, w goes to uniform                     
            w = 1.0 / ((1 - beta ** counts) / (1 - beta))
        else:
            raise ValueError("unknown mode")
        return w / w.sum() # normalize to sum to 1
    
    @staticmethod
    def get_global_l2_norm(mod: torch.nn.Module) -> float:
        """Returns the global L2 norm of gradients for all parameters in a model. For tracking purposes only."""
        grad_norm_sq = 0.0
        for param in mod.parameters():
            if param.grad is not None:
                grad_norm_sq += param.grad.detach().norm(2).item() ** 2
        return grad_norm_sq ** 0.5

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
    HYPER_PARAMS = {**TrainBase.HYPER_PARAMS, 
                    **{'global_bias_lambda_L2': 1e-6, 
                       'global_bias_lambda_L1': 0} 
                   }
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
                
                # randomly add noise to signaling network input, makes model more robust
                utils.set_seeds(self.mod.seed + self.mod._gradient_seed_counter)
                network_noise = torch.randn(X_full.shape, device = X_full.device)
                noise_scale_factor = self.hyper_params['network_noise_scale'] * (cur_lr/self.lr_scheduler.max_lr)
                noise_scale_factor = max(noise_scale_factor, self.hyper_params['min_network_noise'])
                X_full = X_full + (noise_scale_factor * network_noise) 
                
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
                                                                                  bn_weights_lambda_L2=self.hyper_params['bn_weights_lambda_L2'], 
                                            global_bias_lambda_L2=self.hyper_params['global_bias_lambda_L2'], 
                                            bias_global = None, # unused argument
                                            cat_bias_lambda_L2=None, # unused argument
                                            output_weights_lambda_L2=self.hyper_params['output_weights_lambda_L2'],
                                            output_bias_lambda_L2=self.hyper_params['output_bias_lambda_L2'])
                sn_bias_l1_reg = self.mod.signaling_network.L1_reg_bias(global_bias_lambda_L1 = self.hyper_params['global_bias_lambda_L1'])
                sn_param_reg = {**sn_param_reg, **sn_bias_l1_reg}
                param_reg = input_param_reg + sum(sn_param_reg.values()) + sum(output_param_reg.values())
                
                # NOTE: KL divergence is scaled to match loss magnitudes
                # can use MMD in the future if KL unstable
                
                # for adj matrix 
                unmasked_weights = self.mod.signaling_network.weights[~self.mod.signaling_network.mask]
                kl_divergence_adj = self.hyper_params['adj_scaling_KL'] *kl_divergence_normal(empirical_values = unmasked_weights, 
                                                         mu=self.hyper_params['adj_prior_mu'], 
                                                         sigma=self.hyper_params['adj_prior_sigma'], 
                                                         eps=1e-8)
                
                total_loss = prediction_loss + sign_reg + param_reg + stability_loss + uniform_reg + kl_divergence_adj
        
                # gradient
                total_loss.backward()
                self.mod.add_gradient_noise(noise_level = self.hyper_params['gradient_noise_scale'])
                self.prediction_optimizer.step()
                self.mod.signaling_network.implement_mask() # moved out of forward pass to ensure after last backpass these are 0

                sv = np.array([e + 1, batch, cur_lr, time.time() - start_time, spectral_radius, 
                    self.mod.signaling_network.count_sign_mismatch(), 
                    tot_pred_loss.detach().item(), prediction_loss.detach().item(), train_pearson_r, 
                    sign_reg.detach().item(), stability_loss.detach().item(), uniform_reg.detach().item(), 
                    input_param_reg.detach().item(), kl_divergence_adj.detach().item()])
                sv = np.concatenate([sv, 
                                     np.array([v.detach().item() for v in sn_param_reg.values()]), 
                                     np.array([v.detach().item() for v in output_param_reg.values()])])
                self.stats['train'] = np.vstack((self.stats['train'], sv))

                # free up CUDA mem
                del sign_reg, stability_loss, uniform_reg, param_reg, kl_divergence_adj, fit_loss, train_pearson_r
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
                    
            if e % (self.hyper_params['max_epochs']/100) == 0 or e == self.hyper_params['max_epochs']:
                param_names = []
                for name, param in self.mod.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        param_names.append(name)
                if len(param_names) > 0:
                    log_error = 'NaN/inf values found in model parameters at epoch {}'.format(e)
                    log_error += ' for layers ' + ', '.join(param_names)
                    logging.error(log_error)
                    raise ValueError('NaN values found in model parameters at epoch {}'.format(e))
                
                # masks implemented correctly
                correct_masking = True
                mask_coords = torch.nonzero(self.mod.signaling_network.mask.detach())
                if not (self.mod.signaling_network.weights.detach()[mask_coords[:, 0], mask_coords[:, 1]] == 0).all():
                    correct_masking = False                    
                mask_coords = torch.nonzero(self.mod.signaling_network.bias_mask.detach())
                if not (self.bias_global.detach()[mask_coords[:, 0], mask_coords[:, 1]] == 0).all():
                    correct_masking = False
                if not correct_masking:
                    log_error = 'Masking is not being implemented correctly'
                    logging.error(log_error)
                    raise ValueError(log_error)
                    
                if verbose:
                    self.print_stats(e)
            if np.logical_and(e % self.hyper_params['reset_optimizer_epoch'] == 0, e>0):
                self.prediction_optimizer.state = self.reset_state.copy()
        
        self.stats['train'] = pd.DataFrame(data = self.stats['train'], columns = self._stats_cols)
        reg_idx = [col_idx for col_idx, col_name in enumerate(self.stats['train'].columns.tolist()) if col_name.startswith('sn_param_reg')]
        self.stats['train'].insert(max(reg_idx) + 1, 'sn_param_reg_tot_loss', self.stats['train'].iloc[:, reg_idx].sum(axis = 1))

        reg_idx = [col_idx for col_idx, col_name in enumerate(self.stats['train'].columns.tolist()) if col_name.startswith('output_param_reg')]
        self.stats['train'].insert(max(reg_idx) + 1, 'output_param_reg_tot_loss', self.stats['train'].iloc[:, reg_idx].sum(axis = 1))
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
    
    HYPER_PARAMS = {**TrainBase.HYPER_PARAMS, 
                    **{'cat_bias_lambda_L2': 0, # since cat max norm has been implemented
                       'cat_bias_lambda_L1': 0, 
                       'cat_bias_orthogonality_scaler': 0} 
                   }
    
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
        cat_discriminator_params : Dict
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
        
        # add a column to the tracking dataframe
        self._stats_cols.insert(self._stats_cols.index('sn_param_reg_bias_L1_loss')+1, 
                                'sn_param_reg_cat_bias_orthogonality')
        self.stats['train'] = np.empty((0, len(self._stats_cols)))

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
                noise_scale_factor = self.hyper_params['network_noise_scale'] * (cur_lr/self.lr_scheduler.max_lr)
                noise_scale_factor = max(noise_scale_factor, self.hyper_params['min_network_noise'])
                X_full = X_full + (noise_scale_factor * network_noise) # randomly add noise to signaling network input, makes model more robust                Y_full, _ = self.mod.signaling_network(X_full, covariates_idx_) # train signaling network weights
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
                                            bn_weights_lambda_L2=self.hyper_params['bn_weights_lambda_L2'], 
                                            global_bias_lambda_L2=None,# unused argument 
                                            #self.hyper_params['global_bias_lambda_L2'], 
                                            bias_global = None, # unused argument
                                            cat_bias_lambda_L2=self.hyper_params['cat_bias_lambda_L2'],
                                            output_weights_lambda_L2=self.hyper_params['output_weights_lambda_L2'],
                                            output_bias_lambda_L2=self.hyper_params['output_bias_lambda_L2'])
                
                if self.hyper_params['cat_bias_lambda_L1'] == 0:
                    sn_bias_l1_reg = torch.tensor(0.0, device=self.mod.device, dtype=self.mod.dtype)
                else:
                    sn_bias_l1_reg = self.mod.signaling_network.L1_reg_bias(cat_bias_lambda_L1 = self.hyper_params['cat_bias_lambda_L1'])

                if self.hyper_params['cat_bias_orthogonality_scaler'] == 0:
                    sn_cat_bias_orthogonality_reg = torch.tensor(0.0, device=self.mod.device, dtype=self.mod.dtype)
                else:
                    sn_cat_bias_orthogonality_reg = self.mod.signaling_network.cat_orthogonality_regularization(covariates_idx = covariates_idx_, 
                                                                                                            X_in = X_in_, 
                                                                                                            regularization_scaler = self.hyper_params['cat_bias_orthogonality_scaler'])
                sn_param_reg = {**sn_param_reg, **sn_bias_l1_reg, **sn_cat_bias_orthogonality_reg}
                param_reg = input_param_reg + sum(sn_param_reg.values()) + sum(output_param_reg.values())
                
                # NOTE: KL divergence is scaled to match loss magnitudes
                # can use MMD in the future if KL unstable
                
                # for adj matrix 
                unmasked_weights = self.mod.signaling_network.weights[~self.mod.signaling_network.mask]
                kl_divergence_adj = self.hyper_params['adj_scaling_KL'] *kl_divergence_normal(empirical_values = unmasked_weights, 
                                                         mu=self.hyper_params['adj_prior_mu'], 
                                                         sigma=self.hyper_params['adj_prior_sigma'], 
                                                         eps=1e-8)
                
                tot_pred_loss = prediction_loss + sign_reg + param_reg + stability_loss + uniform_reg + kl_divergence_adj
                
                # gradient
                tot_pred_loss.backward()
                self.mod.add_gradient_noise(noise_level = self.hyper_params['gradient_noise_scale'])
                self.prediction_optimizer.step()
                self.mod.signaling_network.implement_mask() # moved out of forward pass to ensure after last backpass these are 0

                sv = np.array([e + 1, batch, cur_lr, time.time() - start_time, spectral_radius, 
                    self.mod.signaling_network.count_sign_mismatch(), 
                    tot_pred_loss.detach().item(), prediction_loss.detach().item(), train_pearson_r, 
                    sign_reg.detach().item(), stability_loss.detach().item(), uniform_reg.detach().item(), 
                    input_param_reg.detach().item(), kl_divergence_adj.items()])
                sv = np.concatenate([sv, 
                                     np.array([v.detach().item() for v in sn_param_reg.values()]), 
                                     np.array([v.detach().item() for v in output_param_reg.values()])])
                self.stats['train'] = np.vstack((self.stats['train'], sv))

                # free up CUDA mem
                del sign_reg, stability_loss, uniform_reg, param_reg, prediction_loss, train_pearson_r
                del input_param_reg, kl_divergence_adj, sn_param_reg, output_param_reg
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

                    
            if e % (self.hyper_params['max_epochs']/100) == 0 or e == self.hyper_params['max_epochs']:
                # exploding/vanishing gradients
                param_names = []
                for name, param in self.mod.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        param_names.append(name)
                if len(param_names) > 0:
                    log_error = 'NaN/inf values found in model parameters at epoch {}'.format(e)
                    log_error += ' for layers ' + ', '.join(param_names)
                    logging.error(log_error)
                    raise ValueError('NaN values found in model parameters at epoch {}'.format(e))
                
                # masks implemented correctly
                correct_masking = True
                for idx, cat_group in enumerate(self.mod.signaling_network.cat_embeddings.keys()):
                    mask_coords = torch.nonzero(self.mod.signaling_network.cat_embeddings_mask[cat_group].detach())
                    embedding_vals = self.mod.signaling_network.cat_embeddings[cat_group].weight.detach()
                    if not (embedding_vals[mask_coords[:, 0], mask_coords[:, 1]] == 0).all():
                        correct_masking = False
                mask_coords = torch.nonzero(self.mod.signaling_network.mask.detach())
                if not (self.mod.signaling_network.weights.detach()[mask_coords[:, 0], mask_coords[:, 1]] == 0).all():
                    correct_masking = False
                if not correct_masking:
                    log_error = 'Masking is not being implemented correctly'
                    logging.error(log_error)
                    raise ValueError(log_error)
                if verbose:
                    self.print_stats(e)

            if np.logical_and(e % self.hyper_params['reset_optimizer_epoch'] == 0, e>0):
                self.prediction_optimizer.state = self.reset_state.copy()

        self.stats['train'] = pd.DataFrame(data = self.stats['train'], columns = self._stats_cols)
        reg_idx = [col_idx for col_idx, col_name in enumerate(self.stats['train'].columns.tolist()) if col_name.startswith('sn_param_reg')]
        self.stats['train'].insert(max(reg_idx) + 1, 'sn_param_reg_tot_loss', self.stats['train'].iloc[:, reg_idx].sum(axis = 1))

        reg_idx = [col_idx for col_idx, col_name in enumerate(self.stats['train'].columns.tolist()) if col_name.startswith('output_param_reg')]
        self.stats['train'].insert(max(reg_idx) + 1, 'output_param_reg_tot_loss', self.stats['train'].iloc[:, reg_idx].sum(axis = 1))
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

    CAT_DISCRIMINATOR_PARAMS = {**CatDiscriminator.DEFAULT_HYPER_PARAMS, 
                             **{k: v for k,v in TrainBase.LR_PARAMS.items() if k != 'max_epochs'},
                             **{'optimizer': torch.optim.Adam,
                                'discriminator_lambda_L2': 1e-5, 
                               'discriminator_penalty_weight': 1}}
    PERT_DISCRIMINATOR_PARAMS = CAT_DISCRIMINATOR_PARAMS.copy()

    # these are params for training, separate from the DEFAULT_HYPER_PARAMS for building the vae
    # discriminators are both built and trained here, so they include both
    VAE_PARAMS = {**{'lambda_l2': 1e-5, 
                  'scaling_KL': 1e-2,
                  'prior_mu': 0, 
                  'prior_sigma': 1, 
                  'optimizer': torch.optim.Adam}, 
                  **{k: v for k,v in TrainBase.LR_PARAMS.items() if k != 'max_epochs'}}

    # other training params
    HYPER_PARAMS = {**TrainCat.HYPER_PARAMS,
                    **{'global_bias_lambda_L2': 0, 'global_bias_lambda_L1': 0}, # KL divergence regularization deals with this
                     **{'prediction_loss_fn_scaler': 1}, # regularizer/multipler for prediction loss output
                    **{'include_gradient_noise_vae': True, #add noise to vae params
                       'include_gradient_noise_embedding': True, # add noise to embedding params
                      'constant_gradient_noise': True}
                   }
    
    def __init__(self,
                 mod, 
                 prediction_optimizer: torch.optim, 
                 prediction_loss_fn: torch.nn.modules.loss,
                 n_adversarial_start: int = 0, # start immediately
                 n_discriminator_train: int = 5, # train generator as frequently as discriminator
                 per_condition_loss: bool = False,
                 gradient_ascent: bool = False, 
                 cat_discriminator_params: Dict = None,
                 pert_discriminator_params: Dict = None, 
                 vae_params: Dict = None,
                 hyper_params: Dict[str, Union[int, float]] = None,
                 train_split: Optional[Dict[str, Union[float, List[str]]]] = {'train': 0.8, 'test': 0.2, 'validation': None},
                 train_seed: int = None,
                 track_test: bool = False,
                 track_validation: bool = False, 
                 n_eval_cells: int = 30, 
                 n_eval_bootstrap: int = 3
                 ):
        """See `TrainBase` for parameters. Additional parameters include:
        
        Parameters
        ----------
        gradient_ascent: bool, optional
            whether to use the gradient ascent label flipping trick for maintaining generator gradients (True) 
            or the standard GAN loss (False), by default False
        per_condition_loss : bool
            whether to calculate the loss on all conditions simultaneously, or each condition separately, by default True
            particularly useful for EMD loss, which may transport conditions across each other
            can also be useful for MSE, which may weigh abundant conditions more heavily when all calculated simultaneously
        n_adversarial_start : int, optional
            when to start adversarial training (before this, only the adj and categorical bias are learned)
            allows the model to learn stimulation / categorical information without incorporating gene expression and the generator
        n_discriminator_train : int, optional
            how frequently to train the generator relative to the discriminator, by default 5
            will train the discriminator every epoch, and generator every n_discriminator_train epochs
        cat_discriminator_params : Dict
            key word arguments to pass to `CatDiscriminator`. see TrainSC.cat_discriminator_params for defaults
            as well as:
                'optimizer': the optimizer to use
                'discriminator_labmda_L2': L2 regularization penalty for discriminator parameters
                'discriminator_penalty_weight': float | List[float]
                    scales the dscriminator loss to incorporate into the adverserial training this can either be a float, 
                    or a list of floats, where each element corresponds to the scaling for that epoch
        pert_discriminator_params : Dict
            same as `cat_discriminator_params`
        n_eval_cells: int
            only relevant for EMD loss
            downsample train/test/validation batches to this # of cells when evaluating loss
            which enables direct comparison between train and test/val for EMD loss
            EMD loss scales with sample size (more samples --> less loss given two IID datasets)
        n_eval_bootstrap: int
            only relevant for EMD loss
            run downsampling this many times when downsampling for evaluation

        n_optimizer_resets is deprecated 
        
        """
        if not (type(mod.signaling_network) is BioNetSC):
            raise ValueError('You must use the correct training class to match the BioNet class.')

        super().__init__(mod = mod, 
                           prediction_optimizer = prediction_optimizer, 
                           prediction_loss_fn = prediction_loss_fn, 
                           hyper_params=hyper_params,
                           train_split=train_split, 
                           train_seed = train_seed,
                         track_validation = track_validation,
                         track_test = track_test)
        
        # X_in and per condition EMD loss check
        if sorted(pd.unique(self.X_train.values.ravel())) != [0,1]:
            raise ValueError('The current per-condition EMD loss can only handle categorical (e.g. binary) perturbation information encoded as 0 for no perturbation and 1 for perturbation')
        if self.mod.X_in.sum(axis = 1).max() != 1:
            raise ValueError('Currently, model training can only handle a single perturbation per cell')
        if self.mod.X_in.sum(axis = 1).min() != 0:
            warnings.warn('The input perturbation matrix does not contain unperturbed/ctrl cells as currently formatted')
        
        assert n_adversarial_start >= 0, "adverserial start must be non-negative"
        self.n_adversarial_start = n_adversarial_start
        self.n_discriminator_train = n_discriminator_train

        if self.hyper_params['n_optimizer_resets'] != 0:
            warnings.warn('The n_optimizer_resets parameter is deprecated, will not be used')

        self.per_condition_loss = per_condition_loss
        if type(self.prediction_loss_fn) == SamplesLoss:
            self._prediction_loss_name = 'EMD'
            self._bootstrap_counter = 0
        elif type(self.prediction_loss_fn) == MSELoss:
            self._prediction_loss_name = 'MSE'
            self._bootstrap_counter = None
        else:
            raise ValueError('Single-cell LEMBAS is only optimized for MSE or EMD currently')

        if not self.per_condition_loss:
            self.compute_loss = self.compute_loss_all
            if self._prediction_loss_name == 'EMD':
                self.evaluate_loss = self.evaluate_emd_loss_all
        else:
            self.compute_loss = self.compute_loss_per_condition

        if self._prediction_loss_name == 'EMD':
            if not self.per_condition_loss:
                self.evaluate_loss = self.evaluate_emd_loss_all
            else:
                self.evaluate_loss = self.evaluate_emd_loss_per_condition
        else:
            self.evaluate_loss = self.evaluate_mse_loss
        
        # for per-condition EMD tracking
        self.n_eval_cells = n_eval_cells
        self.n_eval_bootstrap = n_eval_bootstrap
        
        self.create_data_loader(include_covariates = True, include_expr = True)
        self.gradient_ascent = gradient_ascent
        self.initialize_cat_discriminator(cat_discriminator_params)
        self.initialize_pert_discriminator(pert_discriminator_params) 

        self.initialize_vae(vae_params)
        
        if self.hyper_params['constant_gradient_noise']:
            self.hyper_params['gradient_noise_scale'] = torch.tensor([self.hyper_params['gradient_noise_scale']], 
                                                                     device = self.mod.device, 
                                                                     dtype = self.mod.dtype)

    def _initialize_tracking(self): 
        self._stats_cols = ['epoch', 'batch_index', 
                            'learning_rate', 'cat_discriminator_learning_rate', 'pert_discriminator_learning_rate',
                            'vae_learning_rate',
                            'iter_time', 'spectral_radius', #'eig_sigma', 
                            'n_moa_violations',
                            'train_loss_total', 'train_loss_prediction',
                            'sign_reg_loss', 'stability_reg_loss', 'uniform_reg_loss',
                            'input_param_reg_loss', 
                            'sn_param_reg_weights_kl_divergence',
                            'sn_param_reg_weights_L2_loss', 
                            'sn_param_reg_global_bias_L2_loss', 'sn_param_reg_cat_bias_L2_loss',
                            'sn_param_reg_global_bias_L1_loss', 'sn_param_reg_cat_bias_L1_loss', 'sn_param_reg_cat_bias_orthogonality',
                            'output_param_reg_weights_loss', 'output_param_reg_bias_loss',
                            'vae_param_reg_loss', 'vae_grad_l2_norm', 'global_bias_kl_divergence',
                            'cat_adverserial_loss','cat_discriminator_loss_total',
                            'cat_discriminator_loss_prediction', 'cat_discriminator_param_reg_loss']
        self._stats_cols.extend(['cat_' + cat_type + '_discriminator_grad_l2_norm' for cat_type in list(self.mod.signaling_network.cat_embeddings.keys())])
        self._stats_cols.extend(['pert_adverserial_loss','pert_discriminator_loss_total',
                            'pert_discriminator_loss_prediction', 'pert_discriminator_param_reg_loss' ,
                            'pert_discriminator_grad_l2_norm'
                            ])      

        self.stats = {}
        self.stats['train'] = np.empty((0, len(self._stats_cols)))
        
        self._noise_cols = ['epoch', 'batch_index'] 
        self._noise_cols += [param_name + suffix for param_name, param in self.mod.named_parameters() 
         if param.requires_grad for suffix in ('_norm', '_noise_scale')]

        if not self.hyper_params['include_gradient_noise_vae']:
            self._noise_cols = [col for col in self._noise_cols if 'vae' not in col]
        if not self.hyper_params['include_gradient_noise_embedding']:
            self._noise_cols = [col for col in self._noise_cols if 'cat_embeddings' not in col]    

        self._noise_cols = [col.replace('.', '_') for col in self._noise_cols]  

        self.stats['gradient_noise'] = np.empty((0, len(self._noise_cols)))

        if self.track_test or self.track_validation:
            if type(self.prediction_loss_fn) == SamplesLoss: #downsampling needed
                self._stats_cols_eval = ['epoch', 'batch_index', 'loss_full', 'size_full', 
                                          'bootstrap_index', 'loss_downsample', 'size_downsample']
                self._verbose_loss = 'loss_downsample'
            else:
                self._stats_cols_eval = ['epoch', 'batch_index', 'loss_full']
                self._verbose_loss = 'loss_full'
            
            self.stats['train_eval'] = np.empty((0, len(self._stats_cols_eval)))
            if self.track_test:
                self.stats['test'] = np.copy(self.stats['train_eval'])
            if self.track_validation:
                self.stats['validation'] = np.copy(self.stats['train_eval'])

    def add_gradient_noise(self, cur_lr, e, batch):
        """Adds noise to gradients as a function of the LR and parameter gradient norm."""
        noise_tracker = [e, batch]
        if self.mod.seed:
            utils.set_seeds(self.mod.seed + self.mod._gradient_seed_counter)
        for param_name, param in self.mod.named_parameters():
            if not self.hyper_params['include_gradient_noise_vae'] and 'vae' in param_name:
                continue
            if not self.hyper_params['include_gradient_noise_embedding'] and 'cat_embeddings' in param_name:
                continue
            if self.hyper_params['include_gradient_noise_vae'] and self._no_vae and 'vae' in param_name:
                noise_tracker.extend([0,0])
            if param.requires_grad:
                # TODO: technically this should use the associated param mask
                # but this should be VERY similar if not idetical
                grad_norm = param.grad[param!=0].norm()
                # tot_noise scale is a function of the grad norm and the cur_lr
                # tot_noise = noise_scale # default behavior
                if self.hyper_params['constant_gradient_noise']:
                    tot_noise = self.hyper_params['gradient_noise_scale']
                else:            
                    tot_noise = self.hyper_params['gradient_noise_scale'] * grad_norm * ((cur_lr/self.lr_scheduler.max_lr) ** 0.5)
#                     tot_noise = torch.min(tot_noise, grad_norm*self.hyper_params['max_gradient_noise_fraction'])

                param.grad += torch.randn_like(param.grad) * tot_noise
                noise_tracker.extend([grad_norm.item(), tot_noise.item()])

        self.mod._gradient_seed_counter += 1
        
        self.stats['gradient_noise'] = np.vstack((self.stats['gradient_noise'], 
                                                  np.array(noise_tracker)))  

  
    def initialize_vae(self, vae_params):

        self.vae_learning = {}
        self.vae_learning['params'] = update_with_defaults(self.VAE_PARAMS, vae_params)
        if self.vae_learning['params']['n_optimizer_resets'] != 0:
            warnings.warn('The n_optimizer_resets parameter is deprecated, will not be used')

        if self.hyper_params['max_epochs'] > self.n_adversarial_start:
            self.vae_learning['params']['reset_optimizer_epoch'] = (self.hyper_params['max_epochs'] - self.n_adversarial_start) // (self.vae_learning['params']['n_optimizer_resets'] + 1) # if self.vae_learning['params']['n_optimizer_resets']  > 0 else torch.inf
        else:
            self.vae_learning['params']['reset_optimizer_epoch'] = self.hyper_params['max_epochs'] + 1

        torch_type_params = ['lambda_l2', 'scaling_KL', 'prior_mu', 'prior_sigma']
        for param, param_value in self.vae_learning['params'].items():
            if param in torch_type_params and not torch.is_tensor(param_value):
                    self.hyper_params[param] = torch.tensor(param_value, device=self.mod.device, dtype=self.mod.dtype)  

        self.vae_learning['optimizer'] = self.vae_learning['params']['optimizer'](self.mod.signaling_network.parameters(),
                                                                                  lr = self.vae_learning['params']['maximum_learning_rate'],
                                                                                  weight_decay = 0)
        self.vae_learning['lr_scheduler'] = WarmupCosineAnnealingWarmRestarts(optimizer = self.vae_learning['optimizer'], 
                                                                               T_0 = self.vae_learning['params']['lr_restart_epoch'],
                                                                               T_mul = self.vae_learning['params']['lr_restart_factor'],
                                                                               gamma = self.vae_learning['params']['lr_decay'],
                                                                               eta_min = self.vae_learning['params']['minimum_learning_rate'],
                                                              max_lr=self.vae_learning['params']['maximum_learning_rate'],
                                                              warmup_steps = self.vae_learning['params']['warmup_epochs'],
                                                              last_epoch = -1)
        self.vae_learning['reset_state'] = self.vae_learning['optimizer'].state.copy()
        
        # for training without adversarial portion
        freeze_model(model = self.pert_discriminator['discriminator'])   
        
    def initialize_pert_discriminator(self, pert_discriminator_params):
        if sorted(pd.unique(self.X_train.values.ravel())) != [0,1]:
            raise ValueError('The current global bias perturbation discriminator can only handle categorical (e.g. binary) perturbation information encoded as 0 for no perturbation and 1 for perturbation')
        if not (self.X_train.sum(axis=1) <= 1).all():
            raise ValueError('The current global bias perturbation discriminator can only handle one perturbation condition per cell or sample')
            
            
        # create a mapping of the perturbation conditions to a integer label 
        # TODO: this may be better done in the bionet or scl __init__, but leaving here for now to allow modifications depending on allowing other types of classification (see error checks above)
#         pert_labels = self.mod.X_in.idxmax(axis = 1)
#         if pert_labels.dtype.name == 'category' and pert_labels.dtype.ordered:
#             pert_labels = pert_labels.cat.categories.tolist()
#         else:
#             pert_labels = sorted(set(pert_labels))

#         pert_labels = self.mod.X_in.columns # ensures the X_in columns are called in the order they are input
#         self.pert_mapper = dict(zip(pert_labels, range(self.mod.X_in.shape[1]))) 
#         self.pert_mapper['no_pert'] = self.mod.X_in.shape[1]
        
        self.pert_discriminator = {}
        self.pert_discriminator['params'] = update_with_defaults(self.PERT_DISCRIMINATOR_PARAMS, pert_discriminator_params)
        
        if self.pert_discriminator['params']['n_optimizer_resets'] != 0:
            warnings.warn('The n_optimizer_resets parameter is deprecated, will not be used')
        if self.hyper_params['max_epochs'] > self.n_adversarial_start:
            self.pert_discriminator['params']['reset_optimizer_epoch'] = (self.hyper_params['max_epochs'] - self.n_adversarial_start) // (self.pert_discriminator['params']['n_optimizer_resets']+ 1) #if self.pert_discriminator[['n_optimizer_resets']] > 0 else torch.inf
        else:
            self.pert_discriminator['params']['reset_optimizer_epoch'] = self.hyper_params['max_epochs'] + 1

        self.pert_discriminator['params']['discriminator_penalty_weight'] = self._check_discriminator_pw(self.pert_discriminator['params']['discriminator_penalty_weight'])
        if self.pert_discriminator['params']['discriminator_lambda_L2'] != 0 and self.pert_discriminator['params']['spectral_norm']:
            raise ValueError('Do not apply L2 regularization if using spectral norm.')        
        
        self.pert_discriminator['discriminator'] = CatDiscriminator(n_features_in = self.mod.signaling_network.n_network_nodes_in,
                          n_labels = self.mod.X_in.shape[1] + 1, # + 1 for the no perturbation label 
                          dtype = self.mod.dtype, 
                          device = self.mod.device,
                         batch_momentum = self.pert_discriminator['params']['batch_momentum'], 
                         layer_norm = self.pert_discriminator['params']['layer_norm'], 
                                                                    spectral_norm = self.pert_discriminator['params']['spectral_norm'],
                         dropout_rate = self.pert_discriminator['params']['dropout_rate'], 
                         activation_fn = self.pert_discriminator['params']['activation_fn'], 
                         bionet_activation = self.pert_discriminator['params']['bionet_activation'], 
                         rnn_params = {'activation_function': self.mod.signaling_network.activation_function, 
                                       'leak': self.mod.signaling_network.bionet_params['leak']},
                         smooth_labels = self.pert_discriminator['params']['smooth_labels'],
                         epsilon_smooth = self.pert_discriminator['params']['epsilon_smooth'],                                
                        #  initialize = self.pert_discriminator['params']['initialize'],
                         n_hidden_nodes = self.pert_discriminator['params']['n_hidden_nodes'], 
                         seed = self.train_seed)
    
        self.pert_discriminator['optimizer'] = self.pert_discriminator['params']['optimizer'](self.pert_discriminator['discriminator'].parameters(), 
                                                                            lr = self.pert_discriminator['params']['maximum_learning_rate'],
                                                                            weight_decay = 0)
        self.pert_discriminator['lr_scheduler'] = WarmupCosineAnnealingWarmRestarts(optimizer = self.pert_discriminator['optimizer'], 
                                                                               T_0 = self.pert_discriminator['params']['lr_restart_epoch'],
                                                              T_mul = self.pert_discriminator['params']['lr_restart_factor'], 
                                                              gamma = self.pert_discriminator['params']['lr_decay'],
                                                              eta_min = self.pert_discriminator['params']['minimum_learning_rate'],
                                                              max_lr=self.pert_discriminator['params']['maximum_learning_rate'],
                                                              warmup_steps = self.pert_discriminator['params']['warmup_epochs'],
                                                              last_epoch = -1)
        self.pert_discriminator['reset_state'] = self.pert_discriminator['optimizer'].state.copy()
        
        # for training without adversarial portion
        freeze_model(model = self.pert_discriminator['discriminator'])
        
        # label flipping
        if self.gradient_ascent and self.pert_discriminator['discriminator'].n_labels > 2:
            X_train = self.mod.df_to_tensor(self.X_train)
            target = X_train.argmax(dim=1)
            no_pert = X_train.sum(dim=1) == 0  
            target[no_pert] = self.pert_discriminator['discriminator'].n_labels - 1

            self.pert_class_probs = {}
            self.pert_class_probs['probs'] = {}

            classes, class_counts = torch.unique(target, return_counts = True)
            # class_probs = class_counts / class_counts.sum()

            self.pert_class_probs['classes'] = classes

            for cls in classes:
                cls = cls.item()
                # get probabilities when excluding true class
                mask = torch.ones(self.pert_discriminator['discriminator'].n_labels, 
                                  dtype=torch.bool, device=self.mod.device)
                mask[cls] = False
                self.pert_class_probs['probs'][cls] = class_counts[mask] / class_counts[mask].sum()
    
    
    def initialize_cat_discriminator(self, cat_discriminator_params):
        # self.cat_discriminator['params']['batch_momentum'] = None # bias is a vector in bulk; this should be eliminated in single-cell
        self.cat_discriminator = {}
        self.cat_discriminator['params'] = update_with_defaults(self.CAT_DISCRIMINATOR_PARAMS, cat_discriminator_params)

        if self.cat_discriminator['params']['n_optimizer_resets'] != 0:
            warnings.warn('The n_optimizer_resets parameter is deprecated, will not be used')
        if self.hyper_params['max_epochs'] > self.n_adversarial_start:
            self.cat_discriminator['params']['reset_optimizer_epoch'] = (self.hyper_params['max_epochs'] - self.n_adversarial_start) // (self.cat_discriminator['params']['n_optimizer_resets']+ 1) #if self.cat_discriminator[['n_optimizer_resets']] > 0 else torch.inf
        else:
            self.cat_discriminator['params']['reset_optimizer_epoch'] = self.hyper_params['max_epochs'] + 1

        self.cat_discriminator['params']['discriminator_penalty_weight'] = self._check_discriminator_pw(self.cat_discriminator['params']['discriminator_penalty_weight'])
        if self.cat_discriminator['params']['discriminator_lambda_L2'] != 0 and self.cat_discriminator['params']['spectral_norm']:
            raise ValueError('Do not apply L2 regularization if using spectral norm.')

        self.cat_discriminator['discriminators'] = nn.ModuleDict(
                            {
                                covariate_cat: CatDiscriminator(n_features_in = cat_embedding.weight.shape[1],
                                                  n_labels = cat_embedding.weight.shape[0], 
                                                  dtype = self.mod.dtype, 
                                                  device = self.mod.device,
                                                    batch_momentum = self.cat_discriminator['params']['batch_momentum'], 
                                                    layer_norm = self.cat_discriminator['params']['layer_norm'],
                                                    spectral_norm = self.cat_discriminator['params']['spectral_norm'],
                                                    dropout_rate = self.cat_discriminator['params']['dropout_rate'], 
                                                    activation_fn = self.cat_discriminator['params']['activation_fn'], 
                                                    n_hidden_nodes = self.cat_discriminator['params']['n_hidden_nodes'], 
                                                    bionet_activation = self.cat_discriminator['params']['bionet_activation'], 
                                                    rnn_params = {'activation_function': self.mod.signaling_network.activation_function, 
                                                                'leak': self.mod.signaling_network.bionet_params['leak']},
                         smooth_labels = self.cat_discriminator['params']['smooth_labels'],
                         epsilon_smooth = self.cat_discriminator['params']['epsilon_smooth'], 
#                                                                initialize = self.cat_discriminator['params']['initialize'],
                                                               seed = self.train_seed)
                                for covariate_cat, cat_embedding in self.mod.signaling_network.cat_embeddings.items()}
                        )

        # TODO: combines all discriminators into one optimizer; in the future, may want to separate or create a multi-task classifier
        discriminator_mod_params = []
        for discriminator in self.cat_discriminator['discriminators'].values():
            discriminator_mod_params += list(discriminator.parameters())

        self.cat_discriminator['optimizer'] = self.cat_discriminator['params']['optimizer'](discriminator_mod_params, 
                                                                            lr = self.cat_discriminator['params']['maximum_learning_rate'],
                                                                            weight_decay = 0)
        self.cat_discriminator['lr_scheduler'] = WarmupCosineAnnealingWarmRestarts(optimizer = self.cat_discriminator['optimizer'], 
                                                                               T_0 = self.cat_discriminator['params']['lr_restart_epoch'],
                                                              T_mul = self.cat_discriminator['params']['lr_restart_factor'], 
                                                              gamma = self.cat_discriminator['params']['lr_decay'],
                                                              eta_min = self.cat_discriminator['params']['minimum_learning_rate'],
                                                              max_lr=self.cat_discriminator['params']['maximum_learning_rate'],
                                                              warmup_steps = self.cat_discriminator['params']['warmup_epochs'],
                                                              last_epoch = -1)
        self.cat_discriminator['reset_state'] = self.cat_discriminator['optimizer'].state.copy()
        
        # for training without adversarial portion
        for discriminator in self.cat_discriminator['discriminators'].values():
            freeze_model(model = discriminator)
        
        # label flipping
        if self.gradient_ascent:
            covariates_train = self.mod.signaling_network.covariates_to_tensor(sample_ids = self.X_train.index)

            self.cat_class_probs = {}
            for cat_group_idx, (covariate_cat, discriminator) in enumerate(self.cat_discriminator['discriminators'].items()):
                if discriminator.n_labels > 2:
                    self.cat_class_probs[covariate_cat] = {}
                    self.cat_class_probs[covariate_cat]['probs'] = {}


                    classes, class_counts = torch.unique(covariates_train[:, cat_group_idx], return_counts = True)
                    # class_probs = class_counts / class_counts.sum()

                    self.cat_class_probs[covariate_cat]['classes'] = classes

                    for cls in classes:
                        cls = cls.item()
                        # get probabilities when excluding true class
                        mask = torch.ones(discriminator.n_labels, dtype=torch.bool, device=self.mod.device)
                        mask[cls] = False
                        self.cat_class_probs[covariate_cat]['probs'][cls] = class_counts[mask] / class_counts[mask].sum()
        
    @staticmethod
    def flip_labels_uniform(true_labels, n_labels):
        """Equal probability of drawing any class"""
        flipped_labels = torch.randint_like(true_labels, low=0, high=n_labels)
        match = flipped_labels == true_labels
        while match.any():
            flipped_labels[match] = torch.randint_like(flipped_labels[match], low=0, high=n_labels)
            match = flipped_labels == true_labels

        return flipped_labels

    def flip_labels_proportional(self, true_labels, n_labels, classes, probs):
        """Flip labels, drawing in proportion to the training class label"""
        flipped_labels = torch.empty_like(true_labels)
        for cls in classes:
            cls = cls.item()
            mask_cls = true_labels == cls
            
            if mask_cls.any():
                mask = torch.ones(n_labels, dtype=torch.bool, device=self.mod.device)
                mask[cls] = False
                wrong_classes = classes[mask]
                sampled = wrong_classes[torch.multinomial(probs[cls], num_samples=mask_cls.sum().item(), replacement=True)]
                flipped_labels[mask_cls] = sampled
        return flipped_labels
        
    def _check_discriminator_pw(self, discriminator_penalty_weight):
        if self.hyper_params['max_epochs'] > self.n_adversarial_start:
            if isinstance(discriminator_penalty_weight, float):
                discriminator_penalty_weight = [discriminator_penalty_weight]*(self.hyper_params['max_epochs'] - self.n_adversarial_start)
            elif isinstance(discriminator_penalty_weight, list):
                if len(discriminator_penalty_weight) != (self.hyper_params['max_epochs'] - self.n_adversarial_start):
                    raise ValueError('Must specify a discriminator penalty weight for each epoch')
            else:
                raise ValueError("'discriminator_penalty_weight' must be a float or list")

            discriminator_penalty_weight = self.n_adversarial_start * [0] + discriminator_penalty_weight
        else:
            discriminator_penalty_weight = []
        return discriminator_penalty_weight
    
    def compute_loss_all(self, y_out_, Y_hat, X_in_, covariates_idx_):
        """Used for training. Calculates the loss on all samples at once.
        Note, that X_in_ and covariates_idx_ are unused, but simply for consistency with `compute_loss_per_prediction`
        """
        return self.hyper_params['prediction_loss_fn_scaler']*self.prediction_loss_fn(y_out_, Y_hat)

    def compute_loss_per_condition(self, y_out_, Y_hat, X_in_, covariates_idx_):
        """Used for training. Calculates the loss per condition (per unique categorical covariate + perturbation).
        
        For EMD, this is useful to prevent inaccurate transport between conditions/ 
        For MSE and EMD, the weighting allows rarer conditions to have more or less influence. 
        """

        # reconstruction loss - per-condition EMD
        batch_conds = torch.cat([covariates_idx_, X_in_], dim = 1)
        unique_conds, inverse_indices = torch.unique(batch_conds, dim=0, return_inverse=True)

        counts = torch.bincount(inverse_indices) # no. of cells per condition    
        cond_losses = torch.zeros_like(counts, dtype=self.mod.dtype)
        for cond_i in range(unique_conds.size(0)):
            cond_idx = inverse_indices == cond_i

            y_out_cond = y_out_[cond_idx, :]
            Y_hat_cond = Y_hat[cond_idx, :]
            cond_losses[cond_i] = self.hyper_params['prediction_loss_fn_scaler']*self.prediction_loss_fn(y_out_cond, Y_hat_cond)

        # for MSE and EMD, allows rarer conditions to have more influence
        # for EMD, helps account for sample size bias
        cond_weights = self.make_weights(counts, mode = 'sqrt') 

        return torch.dot(cond_weights, cond_losses) # weighted average 
    
    def evaluate_mse_loss(self, y_out_, X_in_, covariates_idx_, expr_, e, batch):
        """Used for evaluation with MSE loss."""
        cur_mode_train = self.mod.training
        self.mod.eval()
        with torch.inference_mode(): 
            if self.n_adversarial_start <= e:
                y_pred_eval, _, _ = self.mod(X_in_, covariates_idx_,  expr_) 
            else:
                y_pred_eval, _, _ = self.mod.forward_novar(X_in_, covariates_idx_,  expr_) 
            # below makes it condition specific or not
            eval_loss = self.compute_loss(y_out_, y_pred_eval, X_in_, covariates_idx_).item()
        if cur_mode_train:
            self.mod.train()
            
        return np.array([e, batch, eval_loss])

    def evaluate_emd_loss_all(self, y_out_, X_in_, covariates_idx_, expr_, e, batch):
        """
        Used for evaluation with EMD loss. Downsamples to n_eval_cells to ensure sample size bias between 
        test and train comparisons are not present. 
        For training, recomputed without addition of noise, etc. 
        """
        raise ValueError('INTERNAL: Need to adjust for self.n_adversarial_start param, as done in evaluate_mse_loss')
        # comparable tracking of train data with test data
        cur_mode_train = self.mod.training
        self.mod.eval()
        with torch.inference_mode(): 
            n_cells = X_in_.shape[0]

            # get the full prediction for comparison with subsetting
            # TODO: move this just to the eval section and don't track the full
            y_pred_eval, _, _ = self.mod(X_in_, covariates_idx_,  expr_) 
            eval_loss_full = self.hyper_params['prediction_loss_fn_scaler']*self.prediction_loss_fn(y_out_,  y_pred_eval).item()
            print(eval_loss_full)

            if n_cells > self.n_eval_cells:
            
                eval_sv = np.empty((0, len(self._stats_cols_eval)))
                for eval_bootstrap_i in range(self.n_eval_bootstrap):
                    utils.set_seeds(self.mod.seed + self._bootstrap_counter)
                    n_eval_idx = torch.randperm(n_cells)[:self.n_eval_cells]#torch.randint(0, n_cells, (self.n_eval_cells, ))
                    eval_loss = self.hyper_params['prediction_loss_fn_scaler']*self.prediction_loss_fn(y_out_[n_eval_idx, :],
                                        y_pred_eval[n_eval_idx, :]).item()
                    eval_sv_ = np.array([e, batch, eval_loss_full, n_cells, eval_bootstrap_i, eval_loss, self.n_eval_cells])
                    eval_sv = np.vstack((eval_sv, eval_sv_))
                    self._bootstrap_counter += 1
            else:
                eval_sv = np.array([e, batch, eval_loss_full, n_cells, 0, eval_loss_full, n_cells])
            del y_pred_eval, _, eval_loss_full
            utils.clear_memory()
        if cur_mode_train:
            self.mod.train()

        return eval_sv
    
    def evaluate_emd_loss_per_condition(self, y_out_, X_in_, covariates_idx_, expr_, e, batch):
        """
        Used for evaluation with EMD loss per condition. Downsamples to n_eval_cells in EACH condition to 
        ensure sample size bias between test and train comparisons are not present. 
        For training, recomputed without addition of noise, etc. 
        Tracking of downsamples cells is on all cells in the batch, so it doesn't indicate which conditions
        were downsampled. 
        """
        raise ValueError('INTERNAL: Need to adjust for self.n_adversarial_start param, as done in evaluate_mse_loss')
        cur_mode_train = self.mod.training
        self.mod.eval()
        with torch.inference_mode(): 
            batch_conds = torch.cat([covariates_idx_, X_in_], dim = 1)
            unique_conds, inverse_indices = torch.unique(batch_conds, dim=0, return_inverse=True)

            counts = torch.bincount(inverse_indices) # no. of cells per condition  
            cond_losses = torch.empty_like(counts, dtype=self.mod.dtype)

            y_pred_eval, _, _ = self.mod(X_in_, covariates_idx_,  expr_) 

            track_losses = torch.zeros(len(unique_conds), self.n_eval_bootstrap, device = self.mod.device)
            track_losses_full = torch.zeros(len(unique_conds), device = self.mod.device)
            for cond_i in range(unique_conds.size(0)):
                cond_idx = inverse_indices == cond_i

                y_out_cond = y_out_[cond_idx, :]
                y_pred_eval_cond = y_pred_eval[cond_idx, :]
                eval_loss_full = self.hyper_params['prediction_loss_fn_scaler']*self.prediction_loss_fn(y_out_cond, y_pred_eval_cond).item()

                n_cells = y_out_cond.shape[0]
                if n_cells > self.n_eval_cells:
                    for eval_bootstrap_i in range(self.n_eval_bootstrap):
                        utils.set_seeds(self.mod.seed + self._bootstrap_counter)
                        n_eval_idx = torch.randperm(n_cells)[:self.n_eval_cells]
                        track_losses[cond_i, eval_bootstrap_i] = self.hyper_params['prediction_loss_fn_scaler']*self.prediction_loss_fn(y_out_[n_eval_idx, :], y_pred_eval[n_eval_idx, :]).item()
                        self._bootstrap_counter += 1
                else:
                    track_losses[cond_i, :] = eval_loss_full
                track_losses_full[cond_i] = eval_loss_full

            cond_weights = self.make_weights(counts, mode = 'sqrt')
            eval_loss_full = torch.dot(cond_weights, track_losses_full).item()
            n_cells_full = counts.sum().item()

            counts[counts > self.n_eval_cells] = self.n_eval_cells
            n_cells_bootstrap = counts.sum().item()

            if n_cells_bootstrap != n_cells_full: # if bootstrapping occured
                cond_weights = self.make_weights(counts, mode = 'sqrt') # control for EMD loss row-count bias
                eval_loss = (track_losses * cond_weights.unsqueeze(1)).sum(axis = 0).cpu().numpy()

                eval_sv = np.concatenate(
                    [np.tile(np.array([e, batch, eval_loss_full, n_cells_full]), (self.n_eval_bootstrap, 1)), 
                    np.vstack([range(self.n_eval_bootstrap), eval_loss, [n_cells_bootstrap]*self.n_eval_bootstrap]).T], 
                axis = 1)
            else:
                eval_sv = np.array([e, batch, eval_loss_full, n_cells_full, 0, eval_loss_full, n_cells_full])

            del y_pred_eval, _, eval_loss_full
            utils.clear_memory()
        if cur_mode_train:
            self.mod.train()
        return eval_sv

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
        self.mod.signaling_network.implement_mask() # shouldn't be necessary bc called in signaling_network init

        torch.autograd.set_detect_anomaly(True)

        for e in trange(self.hyper_params['max_epochs']):
            self._no_vae = (self.n_adversarial_start > e) or (e % self.n_discriminator_train != 0) or (self.n_discriminator_train > 1 and e == 0)

            cur_lr = self.prediction_optimizer.param_groups[0]['lr']
            self.cat_discriminator['_cur_lr'] = self.cat_discriminator['optimizer'].param_groups[0]['lr']
            self.pert_discriminator['_cur_lr'] = self.pert_discriminator['optimizer'].param_groups[0]['lr']
            self.vae_learning['_cur_lr'] = self.vae_learning['optimizer'].param_groups[0]['lr']
            if self.hyper_params['max_epochs'] > self.n_adversarial_start:
                cur_catdisc_lambda = self.cat_discriminator['params']['discriminator_penalty_weight'][e]
                cur_pertdisc_lambda = self.pert_discriminator['params']['discriminator_penalty_weight'][e]
            else:
                cur_catdisc_lambda = torch.nan
                cur_pertdisc_lambda = torch.nan

            # iterate through batches
            if self.mod.seed:
                utils.set_seeds(self.mod.seed + e)
            for batch, (X_in_, y_out_, covariates_idx_, expr_) in enumerate(self.train_dataloader):
                # set train mode or not (discriminator done below)
                self.mod.train()
                if self._no_vae:
                    freeze_model(self.mod.signaling_network.vae)
                else:
                    unfreeze_model(self.mod.signaling_network.vae)


                self.prediction_optimizer.zero_grad()
                self.cat_discriminator['optimizer'].zero_grad()
                self.pert_discriminator['optimizer'].zero_grad()
                self.vae_learning['optimizer'].zero_grad()

                X_in_, y_out_, covariates_idx_, expr_ = X_in_.to(self.mod.device), y_out_.to(self.mod.device), covariates_idx_.to(self.mod.device), expr_.to(self.mod.device)

                ######################## Forward Pass ########################
                X_full = self.mod.input_layer(X_in_) # transform to full network with ligand input concentrations

                # add noise to ninput
                utils.set_seeds(self.mod.seed + self.mod._gradient_seed_counter)
                network_noise = torch.randn(X_full.shape, device = X_full.device)
                noise_scale_factor = self.hyper_params['network_noise_scale'] * (cur_lr/self.lr_scheduler.max_lr)
                noise_scale_factor = max(noise_scale_factor, self.hyper_params['min_network_noise'])
                X_full = X_full + (noise_scale_factor * network_noise) # randomly add noise to signaling network input, makes model more robust                Y_full, bias_terms = self.mod.signaling_network(X_full = X_full, 

                # NEW
                if self.n_adversarial_start <= e:
                    Y_full, bias_terms = self.mod.signaling_network(X_full = X_full, 
                                                                    covariates_idx = covariates_idx_, 
                                                                    expr = expr_) # train signaling network weights
                    bias_global, bias_mu, bias_log_sigma_squared = bias_terms
                else:
                    Y_full, _ = self.mod.signaling_network.forward_novar(X_full = X_full, 
                                                                    covariates_idx = covariates_idx_, 
                                                                    expr = expr_) # train signaling network weights

                Y_hat = self.mod.output_layer(Y_full)

                if self.n_adversarial_start <= e:
                    unfreeze_model(model = self.pert_discriminator['discriminator'])
                    for discriminator in self.cat_discriminator['discriminators'].values():
                        unfreeze_model(model = discriminator)

                    ######################## Categorical DISCRIMINATOR ########################
                    # discriminator prediction and loss
                    cat_discriminator_loss_accuracy = torch.tensor(0, device = self.mod.device, dtype = self.mod.dtype)
                    for cat_group_idx, (cat, discriminator) in enumerate(self.cat_discriminator['discriminators'].items()):
                        bias_global_prediction = discriminator(bias_global.detach()) # predicted logits
                        # if don't use retain_graph = True, then use bias_global.detach() here

                        target = covariates_idx_[:, cat_group_idx]
                        if discriminator.n_labels == 2:
                            target = target.to(self.mod.dtype).unsqueeze(1)

                        cat_discriminator_loss_accuracy += discriminator.loss_fn(bias_global_prediction, target)   

                    # discriminator regularization
                    cat_discriminator_reg = torch.tensor(0, device = self.mod.device, dtype = self.mod.dtype)
                    for discriminator in self.cat_discriminator['discriminators'].values():
                        cat_discriminator_reg += discriminator.L2_reg(self.cat_discriminator['params']['discriminator_lambda_L2'])
                    cat_discriminator_loss = cat_discriminator_loss_accuracy + cat_discriminator_reg

                    # discriminator optimization
                    # NOTE: discriminator is optimized prior to adverserial training (and loss re-calculated)
                    cat_discriminator_loss.backward() # if bias global is not detached, need to set retain_graph = True here
                    cat_grad_l2s = np.array([self.get_global_l2_norm(mod_discriminator) for mod_discriminator in self.cat_discriminator['discriminators'].values()])
                    self.cat_discriminator['optimizer'].step()

                    # freeze discriminator (to prevent updating discriminator gradients when calling discriminator while 
                    # training generator adverserially below)
                    for discriminator in self.cat_discriminator['discriminators'].values():
                        freeze_model(model = discriminator)

                    # NOTE: 
                    # a good adverserial check here is to see if the vae (and all self.mod) param gradients are still 0, 
                    # as the backward pass for prediction has not yet been called; when using the retain_graph = True
                    # and not calling bias_global.detach() above, the gradients from calculating the discriminator loss
                    # on bias global were leaking into the generator portion

                    ######################## Perturbation DISCRIMINATOR ########################
                    # same implementation as categorical discriminator currently
                    # discriminator prediction and loss
                    bias_global_prediction = self.pert_discriminator['discriminator'](bias_global.detach()) # predicted logits

                    if self.pert_discriminator['discriminator'].n_labels != 2:
                        target = X_in_.argmax(dim=1)
                        # differentiate between rows with no perturbation and rows with perturbation at column 1 (index 0)
                        no_pert = X_in_.sum(dim=1) == 0  
                        target[no_pert] = self.pert_discriminator['discriminator'].n_labels - 1 # -1 for indexing
                    else:
                        target = X_in_#.long().reshape(-1)


                    pert_discriminator_loss_accuracy = self.pert_discriminator['discriminator'].loss_fn(bias_global_prediction, target)   

                    # discriminator regularization
                    pert_discriminator_reg = self.pert_discriminator['discriminator'].L2_reg(self.pert_discriminator['params']['discriminator_lambda_L2'])
                    pert_discriminator_loss = pert_discriminator_loss_accuracy + pert_discriminator_reg

                    # discriminator optimization
                    pert_discriminator_loss.backward() # if bias global is not detached, need to set retain_graph = True here
                    pert_grad_l2 = self.get_global_l2_norm(self.pert_discriminator['discriminator']) # tracking
                    self.pert_discriminator['optimizer'].step()

                    # freeze discriminator
                    freeze_model(model = self.pert_discriminator['discriminator'])
                else:
                    cat_discriminator_loss = torch.tensor(0.0)
                    pert_discriminator_loss = torch.tensor(0.0)
                    cat_discriminator_loss_accuracy = torch.tensor(0.0)
                    pert_discriminator_loss_accuracy = torch.tensor(0.0)
                    cat_discriminator_reg = torch.tensor(0.0)
                    pert_discriminator_reg = torch.tensor(0.0)
                    cat_grad_l2s = np.array([0]*len(self.cat_discriminator['discriminators']))
                    pert_grad_l2 = 0


                ######################## LEMBAS and generator ########################
                # reconstruction loss
                prediction_loss = self.compute_loss(y_out_, Y_hat, X_in_, covariates_idx_)

                # lembas regularization
                sign_reg = self.mod.signaling_network.sign_regularization(lambda_L1 = self.hyper_params['moa_lambda_L1']) # incorrect MoA
        #             ligand_reg = self.mod.ligand_regularization(lambda_L2 = self.hyper_params['ligand_lambda_L2']) # ligand biases
                stability_loss, spectral_radius = self.mod.signaling_network.get_SS_loss(Y_full = Y_full.detach(), spectral_loss_factor = self.hyper_params['spectral_loss_factor'],
                                                                                    subset_n = self.hyper_params['subset_n_spectral'], n_probes = self.hyper_params['n_probes_spectral'], 
                                                                                    power_steps = self.hyper_params['power_steps_spectral'])
                uniform_reg = self.mod.uniform_regularization(lambda_L2 = self.hyper_params['uniform_lambda_L2']*cur_lr, Y_full = Y_full, 
                                                        target_min = 0, target_max = self.hyper_params['uniform_max']) # uniform distribution

                input_param_reg, sn_param_reg, output_param_reg = self.mod.L2_reg(input_lambda_L2=self.hyper_params['input_lambda_L2'],
                                            bn_weights_lambda_L2=self.hyper_params['bn_weights_lambda_L2'], 
                                            global_bias_lambda_L2=self.hyper_params['global_bias_lambda_L2'], 
                                            bias_global = torch.tensor(0) if self._no_vae else bias_global,
                                            cat_bias_lambda_L2=self.hyper_params['cat_bias_lambda_L2'],
                                            output_weights_lambda_L2=self.hyper_params['output_weights_lambda_L2'],
                                            output_bias_lambda_L2=self.hyper_params['output_bias_lambda_L2'])
                sn_bias_l1_reg = self.mod.signaling_network.L1_reg_bias(bias_global = torch.tensor(0) if self._no_vae else bias_global, 
                                                                        global_bias_lambda_L1 = self.hyper_params['global_bias_lambda_L1'], 
                                                                        cat_bias_lambda_L1 = self.hyper_params['cat_bias_lambda_L1'])
    #                 from collections import OrderedDict
    #                 sn_cat_bias_orthogonality_reg = OrderedDict({'cat_bias_orthogonality_loss': 0})
                sn_cat_bias_orthogonality_reg = self.mod.signaling_network.cat_orthogonality_regularization(covariates_idx = covariates_idx_,
                                                                                                            X_in = X_in_,
                                                                                                            regularization_scaler = self.hyper_params['cat_bias_orthogonality_scaler'])
                sn_param_reg = {**sn_param_reg, **sn_bias_l1_reg, **sn_cat_bias_orthogonality_reg}
                param_reg = input_param_reg + sum(sn_param_reg.values()) + sum(output_param_reg.values())
                vae_reg = torch.tensor(0.0)
                if not self._no_vae:
                    vae_reg = self.mod.signaling_network.vae.L2_reg(lambda_L2=self.vae_learning['params']['lambda_l2']) 
                param_reg += vae_reg

                # NOTE: KL divergence is scaled to match loss magnitudes; no bias regularization given KL regularization
                # can use MMD in the future if KL unstable

                # for adj matrix 
                if self.hyper_params['adj_scaling_KL'] == 0:
                    kl_divergence_adj = torch.tensor(0.0, device=self.mod.device, dtype=self.mod.dtype)
                else:
                    unmasked_weights = self.mod.signaling_network.weights[~self.mod.signaling_network.mask]
                    kl_divergence_adj = self.hyper_params['adj_scaling_KL'] *kl_divergence_normal(empirical_values = unmasked_weights, 
                                                                mu=self.hyper_params['adj_prior_mu'], 
                                                                sigma=self.hyper_params['adj_prior_sigma'], 
                                                                eps=1e-8)

                kl_divergence_gb = torch.tensor(0.0)
                if not self._no_vae:
                    # for global bias
                    kl_divergence_gb = self.mod.signaling_network.vae.KL_divergence(z_mu = bias_mu, 
                                                                                z_log_sigma_squared = bias_log_sigma_squared, 
                                                                                scaling_factor = self.vae_learning['params']['scaling_KL'], 
                                                                                prior_mu = self.vae_learning['params']['prior_mu'], 
                                                                                prior_sigma = self.vae_learning['params']['prior_sigma'])
                tot_pred_loss = prediction_loss + sign_reg + param_reg + stability_loss + uniform_reg + kl_divergence_gb + kl_divergence_adj

                pert_adverserial_loss, cat_adverserial_loss = torch.tensor(0.0), torch.tensor(0.0)
                if not self._no_vae:

                    # adverserial portion -- same as discriminator, but recalculating on trained model
                    # adverserial portion -- same as discriminator, but recalculating on trained model
                    # categorical adversary
                    cat_adverserial_loss = torch.tensor(0, device = self.mod.device, dtype = self.mod.dtype)
                    for cat_group_idx, (covariate_cat, discriminator) in enumerate(self.cat_discriminator['discriminators'].items()):
                        bias_global_prediction = discriminator(bias_global) 

                        target = covariates_idx_[:, cat_group_idx]
                        if discriminator.n_labels == 2:
                            target = target.to(self.mod.dtype).unsqueeze(1) if not self.gradient_ascent else 1 - target.to(self.mod.dtype).unsqueeze(1)
                        else:
                            if self.gradient_ascent:
                                utils.set_seeds(self.mod.seed + e + batch)
                                target = self.flip_labels_proportional(true_labels = target, 
                                                                            classes = self.cat_class_probs[covariate_cat]['classes'], 
                                                                            probs = self.cat_class_probs[covariate_cat]['probs'], 
                                                                            n_labels = discriminator.n_labels)

                        cat_adverserial_loss += discriminator.loss_fn(bias_global_prediction, target)  

                    # perturbation adversary
                    bias_global_prediction = self.pert_discriminator['discriminator'](bias_global) 
                    if self.pert_discriminator['discriminator'].n_labels == 2:
                        target = X_in_ if not self.gradient_ascent else 1 - X_in_
                    else:
                        target = X_in_.argmax(dim=1)
                        no_pert = X_in_.sum(dim=1) == 0  
                        target[no_pert] = self.pert_discriminator['discriminator'].n_labels - 1

                        if self.gradient_ascent:
                            utils.set_seeds(self.mod.seed + e + batch)
                            target = self.flip_labels_proportional(true_labels = target, 
                                                                        classes = self.pert_class_probs['classes'], 
                                                                        probs = self.pert_class_probs['probs'], 
                                                                        n_labels = self.pert_discriminator['discriminator'].n_labels)
                    pert_adverserial_loss = self.pert_discriminator['discriminator'].loss_fn(bias_global_prediction, target) 

                    if self.gradient_ascent:
                        # goal is worse accuracy of discriminator
                        # without label flipping trick, we are maximizing the loss of the actual labels
                        # conversely, with label flipping, we are minimizing the loss with flipped labels
                        pert_adverserial_loss = -pert_adverserial_loss
                        cat_adverserial_loss = -cat_adverserial_loss

                tot_pred_loss = tot_pred_loss - (cur_catdisc_lambda*cat_adverserial_loss) - (cur_pertdisc_lambda*pert_adverserial_loss)


                # model gradient
                tot_pred_loss.backward()

                # tracking
                vae_grad_l2 = 0
                if self.n_adversarial_start <= e:
                    vae_grad_l2 = self.get_global_l2_norm(self.mod.signaling_network.vae) 

                self.add_gradient_noise(cur_lr, e, batch)
    #                 self.mod.add_gradient_noise(noise_level = self.hyper_params['gradient_noise_scale'])
                self.prediction_optimizer.step()
                if not self._no_vae:
                    self.vae_learning['optimizer'].step()
                self.mod.signaling_network.implement_mask() # moved out of forward pass to ensure after last backpass these are 0

                # NOTE: 
                # a good adverserial check here is to see if the discriminator parameter gradients have changed since 
                # calling cat_discriminator_loss.backward(); they should not have, but i've found calling the discriminator forward pass 
                # during the adversarial training and the tot_pred_loss.backward() does update them unless I manually freeze them


                # bias global masking can stay in forward pass because it is generated during the forward pass and 
                # won't be updated in the back pass

                sv = np.array([e + 1, batch, cur_lr, self.cat_discriminator['_cur_lr'], self.pert_discriminator['_cur_lr'], self.vae_learning['_cur_lr'],
                            time.time() - start_time, spectral_radius, 
                    self.mod.signaling_network.count_sign_mismatch(), 
                    tot_pred_loss.detach().item(), prediction_loss.detach().item(), #train_pearson_r, 
                    sign_reg.detach().item(), stability_loss.detach().item(), uniform_reg.detach().item(), 
                    input_param_reg.detach().item(), kl_divergence_adj.detach().item()])
                sv = np.concatenate([sv,
                                    np.array([v.detach().item() for v in sn_param_reg.values()]),
                                    np.array([v.detach().item() for v in output_param_reg.values()]),
                                    np.array([vae_reg.detach().item(), vae_grad_l2, kl_divergence_gb.detach().item(),
                                            cur_catdisc_lambda*cat_adverserial_loss.detach().item(), cat_discriminator_loss.detach().item(), 
                                            cat_discriminator_loss_accuracy.detach().item(), cat_discriminator_reg.detach().item()]),
                                    cat_grad_l2s,
                                    np.array([cur_pertdisc_lambda*pert_adverserial_loss.detach().item(), pert_discriminator_loss.detach().item(), 
                                            pert_discriminator_loss_accuracy.detach().item(), pert_discriminator_reg.detach().item(), pert_grad_l2
                                            ])
                                    ])


                self.stats['train'] = np.vstack((self.stats['train'], sv))

                # comparable tracking of train data with test data
                if self.track_test or self.track_validation:
                    eval_sv = self.evaluate_loss(y_out_, X_in_, covariates_idx_, expr_, e, batch)
                    self.stats['train_eval'] = np.vstack((self.stats['train_eval'], eval_sv))

                # free up CUDA mem
                del sign_reg, stability_loss, uniform_reg, param_reg, prediction_loss
                del input_param_reg, sn_param_reg, output_param_reg
                del vae_reg, kl_divergence_gb, kl_divergence_adj
                del cat_discriminator_loss, cat_discriminator_loss_accuracy, cat_discriminator_reg
                del pert_discriminator_loss, pert_discriminator_loss_accuracy, pert_discriminator_reg
                del X_in_, y_out_, covariates_idx_, X_full, Y_full, Y_hat
                utils.clear_memory()

            self.lr_scheduler.step()
            if self.n_adversarial_start <= e:
                self.cat_discriminator['lr_scheduler'].step()
                self.pert_discriminator['lr_scheduler'].step()
            if not self._no_vae:
                self.vae_learning['lr_scheduler'].step()

            # test/validation
            data_types = []
            if self.track_test:
                data_types.append('test')
            if self.track_validation:
                data_types.append('validation')
            for data_type in data_types:
                for batch, (X_in_, y_out_, covariates_idx_, expr_) in enumerate(self.__dict__[data_type + '_dataloader']):
                    X_in_, y_out_, covariates_idx_, expr_ = X_in_.to(self.mod.device), y_out_.to(self.mod.device), covariates_idx_.to(self.mod.device), expr_.to(self.mod.device)            
                    eval_sv = self.evaluate_loss(y_out_, X_in_, covariates_idx_, expr_, e, batch)
                    self.stats[data_type] = np.vstack((self.stats[data_type], eval_sv))


            if e % (self.hyper_params['max_epochs']/100) == 0 or e == self.hyper_params['max_epochs']:
                # vanishing/exploding gradients
                param_names = []
                for name, param in self.mod.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        param_names.append(name)
                if len(param_names) > 0:
                    log_error = 'NaN/inf values found in model parameters at epoch {}'.format(e)
                    log_error += ' for layers ' + ', '.join(param_names)
                    logging.error(log_error)
                    raise ValueError(log_error)

                # masks implemented correctly
                correct_masking = True
                for idx, cat_group in enumerate(self.mod.signaling_network.cat_embeddings.keys()):
                    mask_coords = torch.nonzero(self.mod.signaling_network.cat_embeddings_mask[cat_group].detach())
                    embedding_vals = self.mod.signaling_network.cat_embeddings[cat_group].weight.detach()
                    if not (embedding_vals[mask_coords[:, 0], mask_coords[:, 1]] == 0).all():
                        correct_masking = False
                mask_coords = torch.nonzero(self.mod.signaling_network.mask.detach())
                if not (self.mod.signaling_network.weights.detach()[mask_coords[:, 0], mask_coords[:, 1]] == 0).all():
                    correct_masking = False
                # mask_coords = torch.nonzero(self.mod.signaling_network.bias_mask.T.expand(bias_global.shape[0], -1))
                # if not (bias_global.detach()[mask_coords[:, 0], mask_coords[:, 1]] == 0).all():
                #     correct_masking = False
                if not correct_masking:
                    log_error = 'Masking is not being implemented correctly'
                    logging.error(log_error)
                    raise ValueError(log_error)

                if verbose:
                    self.print_stats(e)

            # deprresetting state
            # if np.logical_and(e % self.hyper_params['reset_optimizer_epoch'] == 0, e>0):
            #     self.prediction_optimizer.state = self.reset_state.copy()
            # if ((e - self.n_adversarial_start) % self.cat_discriminator['params']['reset_optimizer_epoch'] == 0) and e > (self.n_adversarial_start + 1):
            #     self.cat_discriminator['optimizer'].state = self.cat_discriminator['reset_state'].copy()
            # if ((e - self.n_adversarial_start) % self.pert_discriminator['params']['reset_optimizer_epoch'] == 0) and e > (self.n_adversarial_start + 1):
            #     self.pert_discriminator['optimizer'].state = self.pert_discriminator['reset_state'].copy()
            # if ((e - self.n_adversarial_start) % self.vae_learning['params']['reset_optimizer_epoch'] == 0) and e > (self.n_adversarial_start + 1):
            #     self.vae_learning['optimizer'].state = self.vae_learning['reset_state'].copy()

        # format the tracking metrics
        self.stats['train'] = pd.DataFrame(data = self.stats['train'], columns = self._stats_cols)
        reg_idx = [col_idx for col_idx, col_name in enumerate(self.stats['train'].columns.tolist()) if col_name.startswith('sn_param_reg')]
        self.stats['train'].insert(max(reg_idx) + 1, 'sn_param_reg_tot_loss', self.stats['train'].iloc[:, reg_idx].sum(axis = 1))

        reg_idx = [col_idx for col_idx, col_name in enumerate(self.stats['train'].columns.tolist()) if col_name.startswith('output_param_reg')]
        self.stats['train'].insert(max(reg_idx) + 1, 'output_param_reg_tot_loss', self.stats['train'].iloc[:, reg_idx].sum(axis = 1))

        # sanity check
        loss_cols = [
                'train_loss_prediction', 'sign_reg_loss',
                'stability_reg_loss', 'uniform_reg_loss', 'input_param_reg_loss',
                'sn_param_reg_tot_loss', 'output_param_reg_tot_loss', 'vae_param_reg_loss', 'global_bias_kl_divergence']

        reg_tot = self.stats['train'][loss_cols].sum(axis = 1)
        if not np.allclose(reg_tot - self.stats['train']['cat_adverserial_loss'] - self.stats['train']['pert_adverserial_loss'], self.stats['train']['train_loss_total']):
            warnings.warn('Training loss tracking is incorrect')

        if not np.allclose(self.stats['train']['cat_discriminator_loss_total'], 
                            self.stats['train'][['cat_discriminator_loss_prediction', 
                                                'cat_discriminator_param_reg_loss']].sum(axis = 1)):
            raise ValueError('Categorical discriminator loss tracking is incorrect')
        if not np.allclose(self.stats['train']['pert_discriminator_loss_total'], 
                            self.stats['train'][['pert_discriminator_loss_prediction', 
                                                'pert_discriminator_param_reg_loss']].sum(axis = 1)):
            raise ValueError('Perturbation discriminator loss tracking is incorrect')


        if self.track_test or self.track_validation:
            self.stats['train_eval'] = pd.DataFrame(data = self.stats['train_eval'], columns = self._stats_cols_eval)
            if self.track_test:
                self.stats['test'] = pd.DataFrame(data = self.stats['test'], columns = self._stats_cols_eval) 
            if self.track_validation:
                self.stats['validation'] = pd.DataFrame(data = self.stats['validation'], columns = self._stats_cols_eval) 

        self.stats['gradient_noise'] = pd.DataFrame(data = self.stats['gradient_noise'], 
                                                    columns = self._noise_cols)


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
            test_df = pd.DataFrame(data = self.stats['test'], columns = self._stats_cols_eval) 
            test_df = test_df[test_df.epoch == e + 1].mean()
            msg += ', l(te)={:.5f}'.format(test_df[self._verbose_loss])
        if self.track_validation:
            val_df = pd.DataFrame(data = self.stats['validation'], columns = self._stats_cols_eval) 
            val_df = val_df[val_df.epoch == e + 1].mean()
            msg += ', l(v)={:.5f}'.format(val_df[self._verbose_loss])
        msg += ', s={:.5f}'.format(temp_df.spectral_radius)
        msg += ', r={:.5f}'.format(temp_df.learning_rate)
        msg += ', v={:.5f}'.format(temp_df.n_moa_violations)
        print(msg)
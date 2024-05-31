"""
Train the signaling model.
"""
from typing import Dict, List, Union, Optional
import time
from tqdm import trange
import warnings
import logging

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import scLEMBAS.utilities as utils
from .model_utilities import update_with_defaults
from .bionetwork import BioNetSimple, BioNetCat
from .model_components import CatDiscriminator

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

    LR_PARAMS = {'max_epochs': 5000, 'learning_rate': 2e-3, 'reset_optimizer_epoch': 200}
    OTHER_PARAMS = {'batch_size': 32, 'network_noise_scale': 10, 'gradient_noise_scale': 1e-9}
    REGULARIZATION_PARAMS = {'param_lambda_L2': 1e-6, 'moa_lambda_L1': 0.1, #'ligand_lambda_L2': 1e-5, 
                            'uniform_lambda_L2': 1e-4, 'uniform_max': (1/1.2), 'spectral_loss_factor': 1e-5}
    SPECTRAL_RADIUS_PARAMS = {'n_probes_spectral': 5, 'power_steps_spectral': 50, 'subset_n_spectral': 10}
    HYPER_PARAMS = {**LR_PARAMS, **OTHER_PARAMS, **REGULARIZATION_PARAMS, **SPECTRAL_RADIUS_PARAMS}

    def __init__(self, 
                 mod, 
                 prediction_optimizer: torch.optim, 
                 prediction_loss_fn: torch.nn.modules.loss,
                 hyper_params: Dict[str, Union[int, float]] = None,
                 train_split_frac: Dict = {'train': 0.8, 'test': 0.2, 'validation': None},
                 train_seed: int = None):
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
                - 'learning_rate' : the starting learning rate, by default 2e-3
                - 'reset_optimizer_epoch' : number of epochs upon which to reset the optimizer state, by default 200
                - 'batch_size' : number of samples per batch, by default 8
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
        train_split_frac : Dict, optional
            fraction of samples to be assigned to each of train, test and split, by default 0.8, 0.2, and 0 respectively
        train_seed : int, optional
            seed value, by default mod.seed. By explicitly making this an argument, it allows different train-test splits even 
            with the same mod.seed, e.g., for cross-validation
        verbose : bool, optional
            whether to print various progress stats across training epochs


        Returns
        -------
        split_data_dict : Dict[str, pd.DataFrame]
            key value pairs represent the output of the `TrainBase.split_data` function
        """
        
        if not mod.signaling_network._prescaled_weights:
            warnings.warn('Recommended to run `self.mod.signaling_network.prescale_weights()` prior to training')
        
        self.mod = mod
        self.prediction_loss_fn = prediction_loss_fn
        self.prediction_optimizer = prediction_optimizer(self.mod.parameters(), lr=1, weight_decay=0)
        self.reset_state = self.prediction_optimizer.state.copy()

        if not hyper_params:
            self.hyper_params = self.HYPER_PARAMS.copy()
        else:
            self.hyper_params = {k: v for k,v in {**self.HYPER_PARAMS, **hyper_params}.items() if k in self.HYPER_PARAMS} # give user input 

        self.stats = utils.initialize_progress(hyper_params['max_epochs'])

        # set up data objects
        if not train_seed:
            self.train_seed = self.mod.seed
        else:
             self.train_seed = train_seed

        self.X_train, X_test, X_val, self.y_train, y_test, y_val = self.split_data(self.mod.X_in, self.mod.y_out, train_split_frac, train_seed)
        self.split_data_dict = {'X_train': self.X_train, 'X_test': X_test, 'X_val': X_val, 
                        'y_train': self.y_train, 'y_test': y_test, 'y_val': y_val}
        
    def create_data_loader(self, train_data):
            self.train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=self.hyper_params['batch_size'],
                                  # num_workers=n_cores_train,
                                  drop_last = False,
                                  pin_memory = False,#pin_memory,
                                  shuffle=True) 
 
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


class TrainSimple(TrainBase):
    """Training the signaling model for bulk data with no categorical covariates."""

    def __init__(self, 
                  mod, 
                 prediction_optimizer: torch.optim, 
                 prediction_loss_fn: torch.nn.modules.loss,
                 hyper_params: Dict[str, Union[int, float]] = None,
                 train_split_frac: Dict = {'train': 0.8, 'test': 0.2, 'validation': None},
                 train_seed: int = None):
        """See `TrainBase` for parameters."""
        
        if not (type(mod.signaling_network) is BioNetSimple):
            msg = 'You must use the correct training class to match the BioNet class.'
            msg += ' Do you have categorical covariates?'
            raise ValueError(msg)
        super().__init__(mod = mod, 
                           prediction_optimizer = prediction_optimizer, 
                           prediction_loss_fn = prediction_loss_fn, 
                           hyper_params=hyper_params, 
                           train_split_frac=train_split_frac, 
                           train_seed = train_seed)
        
        train_data = ModelData(X_in = self.mod.df_to_tensor(self.X_train).to('cpu'), 
                           y_out = self.mod.df_to_tensor(self.y_train).to('cpu'),
                           covariates_idx = None)
        self.create_data_loader(train_data)
  
    def train_model(self, verbose: bool = True):
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
        for e in trange(self.hyper_params['max_epochs']):
            # set learning rate
            cur_lr = utils.get_lr(e, self.hyper_params['max_epochs'], max_height = self.hyper_params['learning_rate'],
                                start_height=self.hyper_params['learning_rate']/10, end_height=1e-6, peak = 1000)
            self.prediction_optimizer.param_groups[0]['lr'] = cur_lr
            
            cur_loss = []
            cur_eig = []
            
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
                
                # get regularization losses
                sign_reg = self.mod.signaling_network.sign_regularization(lambda_L1 = self.hyper_params['moa_lambda_L1']) # incorrect MoA
        #             ligand_reg = self.mod.ligand_regularization(lambda_L2 = self.hyper_params['ligand_lambda_L2']) # ligand biases
                stability_loss, spectral_radius = self.mod.signaling_network.get_SS_loss(Y_full = Y_full.detach(), spectral_loss_factor = self.hyper_params['spectral_loss_factor'],
                                                                                    subset_n = self.hyper_params['subset_n_spectral'], n_probes = self.hyper_params['n_probes_spectral'], 
                                                                                    power_steps = self.hyper_params['power_steps_spectral'])
                uniform_reg = self.mod.uniform_regularization(lambda_L2 = self.hyper_params['uniform_lambda_L2']*cur_lr, Y_full = Y_full, 
                                                        target_min = 0, target_max = self.hyper_params['uniform_max']) # uniform distribution
                param_reg = self.mod.L2_reg(self.hyper_params['param_lambda_L2']) # all model weights and signaling network biases
                
        #             total_loss = fit_loss + sign_reg + ligand_reg + param_reg + stability_loss + uniform_reg
                total_loss = fit_loss + sign_reg + param_reg + stability_loss + uniform_reg

                # gradient
                total_loss.backward()
                self.mod.add_gradient_noise(noise_level = self.hyper_params['gradient_noise_scale'])
                self.prediction_optimizer.step()

                # store
                cur_eig.append(spectral_radius)
                cur_loss.append(fit_loss.item())

            self.stats = utils.update_progress(self.stats, iter = e, loss = cur_loss, eig = cur_eig, learning_rate = cur_lr, 
                                        n_sign_mismatches = self.mod.signaling_network.count_sign_mismatch())
            
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
                    utils.print_stats(self.stats, iter = e)

                if np.logical_and(e % self.hyper_params['reset_optimizer_epoch'] == 0, e>0):
                    self.prediction_optimizer.state = self.reset_state.copy()

        if verbose:
            mins, secs = divmod(time.time() - start_time, 60)
            print("Training ran in: {:.0f} min {:.2f} sec".format(mins, secs))

        return self.mod, cur_loss, cur_eig, self.split_data_dict, self.stats

class TrainCat(TrainBase):
    """Training the signaling model for bulk data, accounting for categorical covariates of the samples (e.g. cell line, genetic background, etc.)."""
    
    HYPER_PARAMS = TrainBase.HYPER_PARAMS
    
    def __init__(self,
                 mod, 
                 prediction_optimizer: torch.optim, 
                 prediction_loss_fn: torch.nn.modules.loss,
                 hyper_params: Dict[str, Union[int, float]] = None,
                 train_split_frac: Dict = {'train': 0.8, 'test': 0.2, 'validation': None},
                 train_seed: int = None
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
                           train_split_frac=train_split_frac, 
                           train_seed = train_seed)

        train_data = ModelData(X_in = self.mod.df_to_tensor(self.X_train).to('cpu'), 
                           y_out = self.mod.df_to_tensor(self.y_train).to('cpu'),
                           covariates_idx = self.mod.signaling_network.covariates_to_tensor(sample_ids = self.X_train.index))
        self.create_data_loader(train_data)

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
            cur_lr = utils.get_lr(e, self.hyper_params['max_epochs'], max_height = self.hyper_params['learning_rate'],
                                start_height=self.hyper_params['learning_rate']/10, end_height=1e-6, peak = 1000)
            self.prediction_optimizer.param_groups[0]['lr'] = cur_lr

            cur_loss = []
            cur_eig = []

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

                # regularization
                sign_reg = self.mod.signaling_network.sign_regularization(lambda_L1 = self.hyper_params['moa_lambda_L1']) # incorrect MoA
        #             ligand_reg = self.mod.ligand_regularization(lambda_L2 = self.hyper_params['ligand_lambda_L2']) # ligand biases
                stability_loss, spectral_radius = self.mod.signaling_network.get_SS_loss(Y_full = Y_full.detach(), spectral_loss_factor = self.hyper_params['spectral_loss_factor'],
                                                                                    subset_n = self.hyper_params['subset_n_spectral'], n_probes = self.hyper_params['n_probes_spectral'], 
                                                                                    power_steps = self.hyper_params['power_steps_spectral'])
                uniform_reg = self.mod.uniform_regularization(lambda_L2 = self.hyper_params['uniform_lambda_L2']*cur_lr, Y_full = Y_full, 
                                                        target_min = 0, target_max = self.hyper_params['uniform_max']) # uniform distribution
                param_reg = self.mod.L2_reg(self.hyper_params['param_lambda_L2']) # all model weights and signaling network biases

                tot_pred_loss = prediction_loss + sign_reg + param_reg + stability_loss + uniform_reg
                
                # gradient
                tot_pred_loss.backward()
                self.mod.add_gradient_noise(noise_level = self.hyper_params['gradient_noise_scale'])
                self.prediction_optimizer.step()

                # store
                cur_eig.append(spectral_radius)
                cur_loss.append(prediction_loss.item())

            self.stats = utils.update_progress(self.stats, iter = e, loss = cur_loss, eig = cur_eig, learning_rate = cur_lr, 
                                        n_sign_mismatches = self.mod.signaling_network.count_sign_mismatch())

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
                    utils.print_stats(self.stats, iter = e)

                if np.logical_and(e % self.hyper_params['reset_optimizer_epoch'] == 0, e>0):
                    self.prediction_optimizer.state = self.reset_state.copy()

        if verbose:
            mins, secs = divmod(time.time() - start_time, 60)
            print("Training ran in: {:.0f} min {:.2f} sec".format(mins, secs))

        return self.mod, cur_loss, cur_eig, self.split_data_dict, self.stats    
    
    

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
                 train_split_frac: Dict = {'train': 0.8, 'test': 0.2, 'validation': None},
                 train_seed: int = None
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
                           train_split_frac=train_split_frac, 
                           train_seed = train_seed)

        train_data = ModelData(X_in = self.mod.df_to_tensor(self.X_train).to('cpu'), 
                           y_out = self.mod.df_to_tensor(self.y_train).to('cpu'),
                           covariates_idx = self.mod.signaling_network.covariates_to_tensor(sample_ids = self.X_train.index))
        self.create_data_loader(train_data)
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
            # set learning rate
            cur_lr = utils.get_lr(e, self.hyper_params['max_epochs'], max_height = self.hyper_params['learning_rate'],
                                start_height=self.hyper_params['learning_rate']/10, end_height=1e-6, peak = 1000)
            self.prediction_optimizer.param_groups[0]['lr'] = cur_lr
            self.discriminator_optimizer.param_groups[0]['lr'] = cur_lr

            cur_loss = []
            cur_eig = []

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
                param_reg = self.mod.L2_reg(self.hyper_params['param_lambda_L2']) # all model weights and signaling network biases

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

            self.stats = utils.update_progress(self.stats, iter = e, loss = cur_loss, eig = cur_eig, learning_rate = cur_lr, 
                                        n_sign_mismatches = self.mod.signaling_network.count_sign_mismatch())

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
                    utils.print_stats(self.stats, iter = e)

            if np.logical_and(e % self.hyper_params['reset_optimizer_epoch'] == 0, e>0):
                self.prediction_optimizer.state = self.reset_state.copy()
                self.discriminator_optimizer.state = self.reset_state.copy()

        if verbose:
            mins, secs = divmod(time.time() - start_time, 60)
            print("Training ran in: {:.0f} min {:.2f} sec".format(mins, secs))

        return self.mod, cur_loss, cur_eig, self.split_data_dict, self.stats
    

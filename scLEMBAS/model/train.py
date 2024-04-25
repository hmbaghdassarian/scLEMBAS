"""
Train the signaling model.
"""
from collections import OrderedDict
from typing import Dict, List, Union
import time
from tqdm import trange

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import lightning as L
from lightning.pytorch.callbacks import Callback

class SignalDataset(Dataset):
    def __init__(self, X_in, y_out):
        self.X_in = X_in
        self.y_out = y_out
    def __len__(self) -> int:
        "Returns the total number of samples."
        return self.X_in.shape[0]
    def __getitem__(self, idx: int):
        "Returns one sample of data, data and label (X, y)."
        return self.X_in[idx, :], self.y_out[idx, :]

class SignalingDataModule(L.LightningDataModule):
    def __init__(self, X_in, y_out, 
                 batch_size: int, 
                 train_split_frac: Dict = {'train': 0.8, 'test': 0.2, 'validation': None}, 
                seed: int = 888):
        """Lightning Data Module for running the Signaling Model.

        Parameters
        ----------
        X_in : torch.tensor
            input data (should be  = ` SignalingModel.df_to_tensor(SignalingModel.X_in)`)
        y_out : torch.tensor
            output data (should be  = ` SignalingModel.df_to_tensor(SignalingModel.y_out)`)
        batch_size : int
            number of samples per batch
        train_split_frac : Dict, optional
            fraction of samples to be assigned to each of train, test, and validation, by default 0.8, 0.2, and 0 respectively
        seed : int
            random seed for torch and numpy operations, by default 888
        """
        super().__init__()
        self.seed = seed
        self.batch_size = batch_size
        self.data = SignalDataset(X_in.to('cpu'), y_out.to('cpu'))

        self.train_split_frac = OrderedDict({})
        key_order = ['train', 'test', 'validation']
        for key in key_order:
            if key in train_split_frac:
                self.train_split_frac[key] = train_split_frac[key] if train_split_frac[key] else 0
            else:
                self.train_split_frac[key] = 0
        if sum(self.train_split_frac.values())!= 1:
            raise ValueError('Must specify a train-test-val split that sums to 1')

    def setup(self, stage=None):
        self.train_data, self.test_data, self.val_data = random_split(self.data, list(self.train_split_frac.values()), 
                                                                     generator=torch.Generator().manual_seed(self.seed))

    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, batch_size=self.batch_size, drop_last = False, shuffle=True) # pin_memory = pin_memory,

    def val_dataloader(self):
        return DataLoader(dataset=self.val_data, batch_size=self.batch_size, drop_last = False, shuffle=False) # pin_memory = pin_memory,

    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, drop_last = False, shuffle=False) # pin_memory = pin_memory,

class TimerCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_end(self, trainer, pl_module):
        mins, secs = divmod(time.time() - self.start_time, 60)
        print("Training ran in: {:.0f} min {:.2f} sec".format(mins, secs))
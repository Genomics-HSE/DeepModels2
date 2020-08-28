from os.path import join
from typing import Any, Union, List, Optional

import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader

from .dataset_iterator import batchify, load_with_padding_X_y


class DatasetPL(pl.LightningDataModule):
    def __init__(self, path: str,
                 tr_file_first: int,
                 tr_file_last: int,
                 te_file_first: int,
                 te_file_last: int,
                 one_side_padding: int,
                 seq_len: int,
                 batch_size: int,
                 shuffle: bool,
                 num_workers: int):
        super(DatasetPL, self).__init__()
        self.path = path
        
        self.tr_file_first = tr_file_first
        self.tr_file_last = tr_file_last
        self.te_file_first = te_file_first
        self.te_file_last = te_file_last
        
        self.one_side_padding = one_side_padding
        self.seq_len = seq_len
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = DatasetTorch(path=self.path,
                                              file_first=self.tr_file_first,
                                              file_last=self.tr_file_last,
                                              one_side_padding=self.one_side_padding,
                                              seq_len=self.seq_len)
        
        if stage == 'test' or stage is None:
            self.test_dataset = DatasetTorch(path=self.path,
                                             file_first=self.te_file_first,
                                             file_last=self.te_file_last,
                                             one_side_padding=self.one_side_padding,
                                             seq_len=self.seq_len)
    
    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          )
    
    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          )
    

class DatasetTorch(data.Dataset):
    def __init__(self, path, file_first, file_last, one_side_padding, seq_len):
        """
        :param path: a common path to `X` and `y` folder.
        :param file_first:
        :param file_last:
        :param one_side_padding:
        :param seq_len:
        """
        super().__init__()
        
        X_path = join(path, "x")
        y_path = join(path, "y")
        
        data_filenames = [str(i) + ".npy" for i in range(file_first, file_last + 1)]
        
        X_data_res = []
        y_data_res = []
        
        for filename in data_filenames:
            X_file_path = join(X_path, filename)
            y_file_path = join(y_path, filename)
            X_seq_padded_tr, y_seq_tr = load_with_padding_X_y(X_file_path, y_file_path, one_side_padding)
            X_data_i, y_data_i = batchify(X_seq_padded_tr, y_seq_tr, seq_len, one_side_padding)
            X_data_res.append(X_data_i)
            y_data_res.append(y_data_i)
        
        self.X_data = np.vstack(X_data_res)
        self.y_data = np.vstack(y_data_res)
    
    def __len__(self):
        """
        :return: the amount of available samples
        """
        return len(self.X_data)
    
    def __getitem__(self, item):
        return self.X_data[item], self.y_data[item]

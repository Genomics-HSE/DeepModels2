import os
from typing import Union, List, Optional

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
                 sqz_seq_len: int,
                 batch_size: int,
                 n_output: int,
                 shuffle: bool,
                 num_workers: int,
                 use_distance: bool):
        super(DatasetPL, self).__init__()
        self.path = path
        
        self.tr_file_first = tr_file_first
        self.tr_file_last = tr_file_last
        self.te_file_first = te_file_first
        self.te_file_last = te_file_last
        self.use_distance = use_distance  # use distance instead of snp
        
        self.one_side_padding = one_side_padding
        self.seq_len = seq_len
        self.sqz_seq_len = sqz_seq_len
        self.n_output = n_output
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            # if os.environ.get("FAST_RUN") is not None:
            #     self.train_dataset = MockDataset(batch_size=self.batch_size,
            #                                      seq_len=self.seq_len,
            #                                      one_side_padding=self.one_side_padding,
            #                                      n_classes=self.n_output,
            #                                      use_distance=self.use_distance)
            # else:
            self.train_dataset = DatasetTorch(path=self.path,
                                              file_first=self.tr_file_first,
                                              file_last=self.tr_file_last,
                                              one_side_padding=self.one_side_padding,
                                              seq_len=self.seq_len,
                                              sqz_seq_len=self.sqz_seq_len,
                                              use_distance=self.use_distance)
        
        if stage == 'test' or stage is None:
            # if os.environ.get("FAST_RUN") is not None:
            #     self.test_dataset = MockDataset(batch_size=self.batch_size,
            #                                     seq_len=self.seq_len,
            #                                     one_side_padding=self.one_side_padding,
            #                                     n_classes=self.n_output,
            #                                     use_distance=self.use_distance)
            # else:
            self.test_dataset = DatasetTorch(path=self.path,
                                             file_first=self.te_file_first,
                                             file_last=self.te_file_last,
                                             one_side_padding=self.one_side_padding,
                                             seq_len=self.seq_len,
                                             sqz_seq_len=self.sqz_seq_len,
                                             use_distance=self.use_distance)
    
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
    def __init__(self, path, file_first, file_last, one_side_padding, seq_len, use_distance,
                 sqz_seq_len, split_to_batch=False):
        """
        :param path: a common path to `X` and `y` folder.
        :param file_first:
        :param file_last:
        :param one_side_padding:
        :param seq_len:
        """
        super().__init__()
        
        X_path = os.path.join(path, "x")
        y_path = os.path.join(path, "PD")
        
        data_filenames = [str(i) + ".npy" for i in range(file_first, file_last + 1)]
        
        X_data_res = []
        y_data_res = []
        
        for filename in data_filenames:
            X_file_path = os.path.join(X_path, filename)
            y_file_path = os.path.join(y_path, filename)
            X_seq_padded_full, y_seq_full = load_with_padding_X_y(X_file_path, y_file_path, one_side_padding)
            if split_to_batch:
                X_seq_padded_full, y_seq_full = batchify(X_seq_padded_full, y_seq_full, seq_len, one_side_padding)
            # y_data_i_one_hot = one_hot_encoding_numpy(y_data_i, 20)
            if use_distance:
                X_seq_padded_full = convert_snp_to_distances(X_seq_padded_full)
                if len(X_seq_padded_full) >= sqz_seq_len:
                    X_seq_padded_full = X_seq_padded_full[:sqz_seq_len]
                else:
                    X_seq_padded_full = add_zeros_at_end(X_seq_padded_full, sqz_seq_len)
            
            X_data_res.append(X_seq_padded_full)
            y_data_res.append(y_seq_full)
        
        self.X_data = np.vstack(X_data_res)
        self.y_data = np.vstack(y_data_res)
    
    def __len__(self):
        """
        :return: the amount of available samples
        """
        return len(self.X_data)
    
    def __getitem__(self, item):
        return self.X_data[item], self.y_data[item]


class MockDataset(data.Dataset):
    def __init__(self, batch_size: int, seq_len: int, one_side_padding: int,
                 n_classes: int):
        super(MockDataset, self).__init__()
        n_batches = 40
        n_samples = n_batches * batch_size
        self.X_data = torch.LongTensor(n_samples, seq_len + 2 * one_side_padding).random_(0, 2)
        self.y_data = np.random.randint(low=0, high=n_classes, size=(n_samples, seq_len))
    
    def __len__(self):
        """
        :return: the amount of available samples
        """
        return len(self.X_data)
    
    def __getitem__(self, item):
        return self.X_data[item], self.y_data[item]


def one_hot_encoding_numpy(y_data, num_class):
    """
    
    :param batch_data: (batch_size, seq_len)
    :return:
    """
    return (np.arange(num_class) == y_data[..., None]).astype(np.float32)


def convert_snp_to_distances(single_genome):
    """
    
    :param single_genome:
    :return:
    """
    indices = np.where(single_genome == 1)[0]
    distances = np.diff(indices) - 1
    
    return distances.astype('float32')


def add_zeros_at_end(X_seq_distances, desired_length):
    """
    :param X_seq_distances: np array of int
    :return:
    """
    add_len = desired_length - len(X_seq_distances)
    X_seq_dist_padded = np.pad(
        X_seq_distances,
        pad_width=(0, add_len),
        mode='constant',
        constant_values=-1
    )
    return X_seq_dist_padded

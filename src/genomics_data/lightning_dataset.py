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
                 seq2seq: bool,
                 seq_len: int,
                 squeeze: bool,
                 sqz_seq_len: int,
                 split_sample: bool,
                 split_seq_len: int,
                 n_class: int,
                 batch_size: int,
                 shuffle: bool,
                 num_workers: int
                 ):
        super(DatasetPL, self).__init__()
        self.path = path
        
        self.tr_file_first = tr_file_first
        self.tr_file_last = tr_file_last
        self.te_file_first = te_file_first
        self.te_file_last = te_file_last
        
        self.seq2seq = seq2seq
        self.seq_len = seq_len
        self.squeeze = squeeze  # use distance instead of snp
        self.sqz_seq_len = sqz_seq_len
        self.split_sample = split_sample
        self.split_seq_len = split_seq_len
        
        self.n_class = n_class
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        self.collate_fn = collate_distances_fn if squeeze else None
    
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = DatasetTorch(path=self.path,
                                              file_first=self.tr_file_first,
                                              file_last=self.tr_file_last,
                                              seq2seq=self.seq2seq,
                                              squeeze=self.squeeze,
                                              sqz_seq_len=self.sqz_seq_len,
                                              split_sample=self.split_sample,
                                              split_seq_len=self.split_seq_len
                                              )
        
        if stage == 'test' or stage is None:
            self.test_dataset = DatasetTorch(path=self.path,
                                             file_first=self.te_file_first,
                                             file_last=self.te_file_last,
                                             seq2seq=self.seq2seq,
                                             squeeze=self.squeeze,
                                             sqz_seq_len=self.sqz_seq_len,
                                             split_sample=False,
                                             split_seq_len=None
                                             )
    
    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn
                          )
    
    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_dataset,
                          batch_size=1,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn
                          )


def collate_distances_fn(batch):
    X = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [X, target]


class DatasetTorch(data.Dataset):
    def __init__(self, path, file_first, file_last, seq2seq, squeeze, sqz_seq_len,
                 split_sample, split_seq_len):
        """
        :param path: a common path to `X` and `y` folder.
        :param file_first:
        :param file_last:
        :param seq_len:
        """
        super().__init__()
        
        X_path = os.path.join(path, "x")
        y = "y" if seq2seq else "PD"
        y_path = os.path.join(path, y)
        
        data_filenames = [str(i) + ".npy" for i in range(file_first, file_last + 1)]
        
        X_data_res = []
        y_data_res = []
        
        self.ix_to_filename = {}
        
        for ix, filename in enumerate(data_filenames):
            X_file_path = os.path.join(X_path, filename)
            y_file_path = os.path.join(y_path, filename)
            
            X_seq_full = np.load(X_file_path, mmap_mode=None)
            y_seq_full = np.load(y_file_path, mmap_mode=None)
            
            if squeeze:
                X_seq_full = convert_snp_to_distances(X_seq_full)
            
            if split_sample:
                X_seq_full, y_seq_full = batchify(X_seq_full, y_seq_full, split_seq_len)
            # y_data_i_one_hot = one_hot_encoding_numpy(y_data_i, 20)
            
            X_data_res.append(X_seq_full)
            y_data_res.append(y_seq_full)
            self.ix_to_filename[ix] = filename
        
        # X_data batch_size, seq_len
        self.X_data = X_data_res
        # print(self.X_data.shape)
        self.y_data = y_data_res
    
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
    distances = np.diff(indices)
    
    # distances = np.insert(distances, [0, len(distances)], indices[0])
    distances = np.insert(distances, [0, len(distances)], [indices[0], len(single_genome) - indices[-1]])
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

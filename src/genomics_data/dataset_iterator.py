import os
from os.path import isfile, join

from abc import ABCMeta, abstractmethod

import torch
import numpy as np
from torch.utils import data


class Dataset(metaclass=ABCMeta):
    def __init__(self, path, train_size, batch_size,
                 seq_len):
        self.path = path
        self.batch_size = batch_size
        self.train_size = train_size
        self.seq_len = seq_len
    
        self.X_path = join(path, "x")
        self.y_path = join(path, "y")
    
        self.data_file_names = [f for f in os.listdir(self.X_path) if isfile(join(self.X_path, f))]
        
        self.X_data = None
        self.y_data = None
        self.X_data_test = None
        self.y_data_test = None
        
    def __len__(self):
        return self.X_data.shape[1] // self.seq_len
    
    def get_batch(self, X_data, y_data, batch_ix, seq_i):
        batch_len = min(self.batch_size, X_data.shape[0] - batch_ix)
        batch_beg_ix = batch_ix
        batch_end_ix = batch_ix + batch_len
        
        seq_len = min(self.seq_len, X_data.shape[1] - seq_i)
        seq_beg_ix = seq_i
        seq_end_ix = seq_i + seq_len
        
        return X_data[batch_beg_ix:batch_end_ix, seq_beg_ix:seq_end_ix], \
               y_data[batch_beg_ix:batch_end_ix, seq_beg_ix:seq_end_ix]
    
    def get_fixlen_iter(self, start=0, train=True):
        if train:
            X_data = self.X_data
            y_data = self.y_data
        else:
            X_data = self.X_data_test
            y_data = self.y_data_test
        for batch_ix in range(start, X_data.shape[0] - 1, self.batch_size):
            for seq_ix in range(start, X_data.shape[1] - 1, self.seq_len):
                yield self.get_batch(X_data, y_data, batch_ix, seq_ix)


class DataIterator(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        batch_filenames = self.data_file_names[:self.train_size]
        
        X_batch = []
        y_batch = []
        for filename in batch_filenames:
            X = np.load(join(self.X_path, filename), mmap_mode='r')
            y = np.load(join(self.y_path, filename), mmap_mode='r')
            X_batch.append(X)
            y_batch.append(y)
        
        self.X_data = np.array(X_batch)
        self.y_data = np.array(y_batch)
        
        test_file = self.data_file_names[self.train_size]
        self.X_data_test = np.array([np.load(join(self.X_path, test_file), mmap_mode='r')])
        self.y_data_test = np.array([np.load(join(self.y_path, test_file), mmap_mode='r')])
    
    


if __name__ == '__main__':
    path = "../../data/m_data_numpy"
    dataset = DataIterator(path, train_size=10, batch_size=2, seq_len=78)
    data_iter = dataset.get_fixlen_iter()
    
    for i in data_iter:
        print(i[0].shape)
    print("ko")

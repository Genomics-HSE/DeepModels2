import os
from os.path import isfile, join

import torch
import numpy as np
from torch.utils import data


class DataIterator(data.Dataset):
    def __init__(self, path, train_size, batch_size,
                 seq_len):
        self.path = path
        self.batch_size = batch_size
        self.train_size = train_size
        self.seq_len = seq_len
        
        self.X_path = join(path, "x")
        self.y_path = join(path, "y")
        
        self.data_file_names = [f for f in os.listdir(self.X_path) if isfile(join(self.X_path, f))]

        batch_filenames = self.data_file_names[:self.batch_size]
        
        X_batch = []
        y_batch = []
        for filename in batch_filenames:
            X = np.load(join(self.X_path, filename), mmap_mode='r')
            y = np.load(join(self.y_path, filename), mmap_mode='r')
            X_batch.append(X)
            y_batch.append(y)
        
        self.X_data = np.array(X_batch)
        self.y_data = np.array(y_batch)

        test_file = self.data_file_names[self.batch_size]
        self.X_data_test = np.array([np.load(join(self.X_path, test_file), mmap_mode='r')])
        self.y_data_test = np.array([np.load(join(self.y_path, test_file), mmap_mode='r')])
    
    def __len__(self):
        return self.X_data.shape[1] // self.seq_len
    
    def get_batch(self, X_data, y_data, i):
        seq_len = min(self.seq_len, X_data.shape[1] - i)
        beg_ix = i
        end_ix = i + seq_len
        
        return X_data[:, beg_ix:end_ix], y_data[:, beg_ix:end_ix]

    def get_fixlen_iter(self, start=0, train=True):
        if train:
            X_data = self.X_data
            y_data = self.y_data
        else:
            X_data = self.X_data_test
            y_data = self.y_data_test
        for i in range(start, X_data.shape[1]-1, self.seq_len):
            yield self.get_batch(X_data, y_data, i)


if __name__ == '__main__':
    path = "../../data/m_data_numpy"
    dataset = DataIterator(path, train_size=10, batch_size=2, seq_len=78)
    data_iter = dataset.get_fixlen_iter()
    
    for i in data_iter:
        print(i[0].shape)
    print("ko")
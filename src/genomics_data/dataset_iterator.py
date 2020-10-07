import os
from os.path import isfile, join
from abc import ABCMeta, abstractmethod

import torch
import numpy as np
from torch.utils import data
import torch.nn.functional as F

from tqdm import tqdm


class CustomDataset(metaclass=ABCMeta):
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


class SequentialDataIterator(CustomDataset):
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


class RandomDataIteratorOneSeq(CustomDataset):
    def __init__(self, one_side_padding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        train_file = self.data_file_names[0]
        test_file = self.data_file_names[1]
        
        X_seq_padded_tr, y_seq_tr = self.load_with_padding_X_y(train_file, one_side_padding)
        X_seq_padded_test, y_seq_test = self.load_with_padding_X_y(test_file, one_side_padding)
        self.X_data, self.y_data = batchify(X_seq_padded_tr, y_seq_tr, self.seq_len, one_side_padding)
        self.X_data_test, self.y_data_test = batchify(X_seq_padded_test, y_seq_test, self.seq_len, one_side_padding)
    
    def get_batch(self, X_data, y_data, batch_ix):
        batch_len = min(self.batch_size, X_data.shape[0] - batch_ix)
        batch_beg_ix = batch_ix
        batch_end_ix = batch_ix + batch_len
        
        return X_data[batch_beg_ix:batch_end_ix, :], \
               y_data[batch_beg_ix:batch_end_ix, :]
    
    @property
    def n_batches(self):
        return 1 + self.X_data.shape[0] // self.batch_size
    
    def get_fixlen_iter(self, start=0, train=True):
        if train:
            X_data = self.X_data
            y_data = self.y_data
        else:
            X_data = self.X_data_test
            y_data = self.y_data_test
        for batch_ix in range(start, X_data.shape[0], self.batch_size):
            yield self.get_batch(X_data, y_data, batch_ix)


def batchify(padded_x_seq, y_seq, seq_len, one_side_padding):
    X_batch = []
    y_batch = []
    for i in tqdm(range(one_side_padding, len(padded_x_seq) - seq_len - one_side_padding + 1, seq_len)):
        X_batch.append(padded_x_seq[i - one_side_padding:i + seq_len + one_side_padding])
        y_batch.append(y_seq[i - one_side_padding:i - one_side_padding + seq_len])
    X_batch = np.vstack(X_batch)
    y_batch = np.vstack(y_batch)
    print("Shape of datasets: X is {} and y {} ".format(X_batch.shape, y_batch.shape))
    return X_batch, y_batch


def load_with_padding_X_y(X_file_path: str, y_file_path: str, one_side_padding: int, mmap=None):
    """
    Load file depending on its type (torch or numpy)
    :param X_file_path: must by "*.npy" or "*.pt"
    :param y_file_path: must by "*.npy" or "*.pt"
    :param one_side_padding: padding length
    :param mmap: the option to use swap when using numpy load
    :return: data
    """
    
    if X_file_path.endswith(".npy") and y_file_path.endswith(".npy"):
        X_seq = np.pad(
            np.load(X_file_path, mmap_mode=mmap),
            pad_width=(one_side_padding,),
            mode='constant',
            constant_values=(-1,)
        )
        y_seq = np.load(y_file_path, mmap_mode=mmap)
    elif X_file_path.endswith(".pt") and y_file_path.endswith(".pt"):
        X_seq = F.pad(
            input=torch.load(X_file_path),
            pad=[one_side_padding, one_side_padding],
            value=-1
        )
        y_seq = torch.load(y_file_path)
    else:
        raise NameError("This shouldn't happen")
    return X_seq, y_seq


if __name__ == '__main__':
    path = "../../data/m_data_numpy"
    dataset = RandomDataIteratorOneSeq(path=path, train_size=1, batch_size=3, seq_len=10,
                                       one_side_padding=5)
    data_iter = dataset.get_fixlen_iter()
    for i in data_iter:
        print(i[0].shape)

import os
import numpy as np
import torch

__all__ = [
    'ensure_directories', 'one_hot_encoding', 'boolean_string'
]


def ensure_directories(root, *args):
    os.makedirs(root, exist_ok=True)
    
    def ensure(relpath):
        path = os.path.join(root, relpath)
        os.makedirs(path, exist_ok=True)
        return path
    
    return tuple(
        ensure(arg) for arg in args
    )


def one_hot_encoding_numpy(y_data, num_class):
    """
    
    :param batch_data: (batch_size, seq_len)
    :return:
    """
    return (np.arange(num_class) == y_data[..., None]).astype(np.float32)


def one_hot_encoding(y_data, num_class):
    batch_size, seq_len = y_data.shape
    y_one_hot = torch.FloatTensor(batch_size, seq_len, num_class)
    
    y_one_hot.zero_()
    y_one_hot.scatter_(2, y_data.unsqueeze(2), 1)
    return y_one_hot


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

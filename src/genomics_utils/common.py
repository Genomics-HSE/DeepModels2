import os
import numpy as np

__all__ = [
    'ensure_directories', 'boolean_string'
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


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

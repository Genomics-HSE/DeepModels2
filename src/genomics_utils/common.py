import os
import numpy as np

__all__ = [
    'ensure_directories', 'boolean_string', 'float_to_int'
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


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def float_to_int(f):
    return int(float(f))
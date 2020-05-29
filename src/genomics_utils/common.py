import os

__all__ = [
    'ensure_directories'
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
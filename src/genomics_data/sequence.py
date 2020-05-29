import os
from tqdm import tqdm
import torch
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, X, Y):
        """
        
        :param X: (batch_size, seq len)
        :param Y: (batch_size, seq len)
        """
        self.X = X
        self.Y = Y
    
    def __len__(self):
        assert len(self.X) == len(self.Y)
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


class OneSequenceDataset:
    
    def __init__(self, data_path, target_width, one_side_width, train_part=0.8):
        self.data_path = data_path
        self.target_width = target_width
        self.one_side_width = one_side_width
        self.train_part = train_part
        
        X_seq, Y_seq = get_one_sequence_data(data_path)
        X_batch, Y_batch = get_batches(X_seq, Y_seq, target_width, one_side_width)

        assert len(X_batch) == len(Y_batch)
        train_border = int(train_part * len(X_batch))
        self._train_set = Dataset(X_batch[:train_border], Y_batch[:train_border])
        self._test_set = Dataset(X_batch[train_border:], Y_batch[train_border:])

    @property
    def train_set(self):
        return self._train_set
    
    @property
    def test_set(self):
        return self._test_set


def get_one_sequence_data(data_path):
    x_path = os.path.join(data_path, "x")
    y_path = os.path.join(data_path, "y")
    
    x = torch.load(os.path.join(x_path, "0.pt"))
    y = torch.load(os.path.join(y_path, "0.pt"))
    
    return x, y


def get_batches(x_sequence, y_sequence, target_width, one_side_width):
    x_batch = []
    y_batch = []
    print("Generating data")
    for i in tqdm(range(one_side_width, len(x_sequence) - one_side_width - target_width, target_width)):
        x_batch.append(x_sequence[i - one_side_width:i + target_width + one_side_width])
        y_batch.append(y_sequence[i:i + target_width])
    x_batch = torch.stack(x_batch)
    y_batch = torch.stack(y_batch)
    return x_batch, y_batch


def get_data_iterators(data_path, target_width, one_side_width, bsize):
    params = {'batch_size': bsize,
              'shuffle': False,
              'num_workers': 0
              }
    
    X_seq, Y_seq = get_one_sequence_data(data_path)
    X_batch, Y_batch = get_batches(X_seq, Y_seq, target_width, one_side_width)
    
    train_border = int(0.8 * len(X_batch))
    
    training_set = BatchDataset(X_batch[:train_border], Y_batch[:train_border])
    training_generator = data.DataLoader(training_set, **params)
    
    validation_set = BatchDataset(X_batch[train_border:], Y_batch[train_border:])
    validation_generator = data.DataLoader(validation_set, **params)
    
    return training_generator, validation_generator




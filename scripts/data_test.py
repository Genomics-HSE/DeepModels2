import sys
from genomics_utils import available
from genomics_data.sequence import OneSequenceDataset

import math

def kotok():
    W = int(float(3e6))
    F = 25
    S = 2
    num_layers = 4
    
    PF = 5
    PS = PF
    conv_pad = (F - 1) / 2
    for i in range(num_layers):
        W = math.floor(1 + ((W - F + 2*conv_pad) / S))
        W = math.floor(1 + (W - PF) / PS)
    
    return W


if __name__ == '__main__':
    print(kotok())
    
    pass
    # data_path = sys.argv[1]
    # dataset = OneSequenceDataset(data_path, target_width=5, one_side_width=5)
    # print("Len of train: {}".format(len(dataset.train_set)))
    # print("Len of test: {}".format(len(dataset.test_set)))
    # print("")
    # print("Size of one train sample: {} {}".format(dataset.train_set[0][0].size(), dataset.train_set[0][1].size()))

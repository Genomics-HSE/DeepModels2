import sys
from genomics_utils import available
from genomics_data.sequence import OneSequenceDataset

if __name__ == '__main__':
    data_path = sys.argv[1]
    dataset = OneSequenceDataset(data_path, target_width=5, one_side_width=5)
    print("Len of train: {}".format(len(dataset.train_set)))
    print("Len of test: {}".format(len(dataset.test_set)))
    print("")
    print("Size of one train sample: {} {}".format(dataset.train_set[0][0].size(), dataset.train_set[0][1].size()))

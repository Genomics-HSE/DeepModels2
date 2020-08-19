import os
import numpy as np
from tqdm import tqdmv

import pytorch_lightning as pl
import comet_ml
import torch
from torch.utils import data

from genomics import Classifier
from genomics_data import RandomDataIteratorOneSeq, SequentialDataIterator
from genomics_utils import available, get_logger, ensure_directories


model = Model()

# most basic trainer, uses good defaults (1 gpu)
trainer = pl.Trainer(gpus=1)
trainer.fit(model)

"""
gru: gru-train gru-test

gru-vars = hidden_size=256 \
		num_layers=2 \
		batch_first=true \
		bidirectional=true \
		dropout=0.1 \
		conv_n_layers=4 \
		kernel_size=5
"""

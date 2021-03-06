import torch.nn as nn
import pytorch_lightning as pl


class PrepareFloat(pl.LightningModule):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        """
        :param input: (batch_size, seq_len)
        :return: # (batch_size, 1, seq_len)
        """
        input = input.float()
        input = input.unsqueeze(1)
        return input

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .training_process import LightningModuleExtended


class CNNSqueeze(LightningModuleExtended):
    def __init__(self):
        self.convs = nn.ModuleList([ConvBlock(in_channels=hidden_size,
                                              out_channels=hidden_size,
                                              kernel_size=kernel_size,
                                              stride=1,
                                              padding=(kernel_size - 1) // 2,
                                              dropout=dropout)
                                    for _ in range(n_layers)])
        
    
    def forward(self, input):
        pass


class ConvBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels, conv_kernel_size, conv_stride, dropout_p,
                 pool_kernel_size):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=conv_kernel_size,
                                stride=conv_stride
                                )
        
        self.batch_norm = nn.BatchNorm1d(num_features=out_channels)
        self.dropout = nn.Dropout(dropout_p)
        self.pooling = nn.MaxPool1d(
            kernel_size=pool_kernel_size
        )
    
    def forward(self, input):
        # (batch_size, input_channels, seq_len)
        output = self.conv1d(input)
        # (batch_size, out_channels, seq_len)
        output = self.batch_norm(output)
        output = F.relu(output)
        output = self.dropout(output)
        
        output = self.pooling(output)
        
        return output

import math
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from genomics_utils import LightningModuleExtended


class CNNSqueeze(LightningModuleExtended):
    def __init__(self, hidden_size, conv_kernel_size, conv_stride, num_layers, dropout_p,
                 pool_kernel_size, n_output, seq_len, conv_pad=0):
        super().__init__()
        
        self.n_output = n_output
        self.seq_len = seq_len
        
        in_out_channels = [(1, hidden_size)]
        in_out_channels = in_out_channels + [(hidden_size, hidden_size) for _ in range(num_layers - 2)]
        in_out_channels = in_out_channels + [(hidden_size, 1)]
        
        for _ in range(num_layers):
            conv_out_size = math.floor(((seq_len - conv_kernel_size + 2 * conv_pad) / conv_stride) + 1)
            pool_out_size = math.floor(((conv_out_size - pool_kernel_size) / pool_kernel_size) + 1)
            seq_len = pool_out_size
        
        self.convs = nn.ModuleList([ConvBlock(in_channels=in_hs,
                                              out_channels=out_hs,
                                              conv_kernel_size=conv_kernel_size,
                                              conv_stride=conv_stride,
                                              dropout_p=dropout_p,
                                              pool_kernel_size=pool_kernel_size)
                                    for in_hs, out_hs in in_out_channels])
        
        self.dense1 = nn.Linear((seq_len, seq_len))
        self.dropout1 = nn.Dropout(dropout_p)
        self.dense2 = nn.Linear(seq_len, seq_len)
        self.dropout2 = nn.Dropout(dropout_p)
        self.dense3 = nn.Linear(seq_len, n_output)
    
    def forward(self, input, hiddens):
        """
        :param input: (batch_size, seq_len)
        :return:
        """
        input = input.float()
        input = input.unsqueeze(1)
        # (batch_size, 1, seq_len)

        # begin convolutional blocks...
        for i, conv in enumerate(self.convs):
            input = conv(input)

        # input = (batch_size, 1, res_len)
        input = input.squeeze(1)
        pred = F.relu(self.dropout(self.dense1(input)))
        pred = F.relu(self.dropout(self.dense2(pred)))
        output = F.log_softmax(self.dense3(pred), dim=-1)
        return output, None


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

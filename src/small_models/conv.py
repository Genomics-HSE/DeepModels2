import math
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .training import SmallModelTraining


class Model:
    def __new__(cls, args):
        return CNN(channel_size=args.channel_size,
                   conv_kernel_size=args.conv_kernel_size,
                   conv_stride=args.conv_stride,
                   num_layers=args.num_layers,
                   dropout_p=args.dropout_p,
                   pool_kernel_size=args.pool_kernel_size,
                   n_output=args.n_class,
                   seq_len=args.seq_len,
                   sqz_seq_len=args.sqz_seq_len
                   )


class CNN(SmallModelTraining):
    def __init__(self, channel_size, conv_kernel_size, conv_stride, num_layers, dropout_p,
                 pool_kernel_size, n_output, seq_len, sqz_seq_len):
        super().__init__()
        self.save_hyperparameters()
        
        self.channel_size = channel_size
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.num_layers = num_layers
        self.pool_kernel_size = pool_kernel_size
        self.n_output = n_output
        self.seq_len = seq_len
        self.sqz_seq_len = sqz_seq_len
        
        in_out_channels = [(1, channel_size)]
        in_out_channels = in_out_channels + [(channel_size, channel_size) for _ in range(num_layers - 2)]
        in_out_channels = in_out_channels + [(channel_size, 1)]
        
        conv_pad = 0
        for _ in range(num_layers):
            conv_out_size = math.floor(((sqz_seq_len - conv_kernel_size + 2 * conv_pad) / conv_stride) + 1)
            pool_out_size = math.floor(((conv_out_size - pool_kernel_size) / pool_kernel_size) + 1)
            sqz_seq_len = pool_out_size
        
        self.convs = nn.ModuleList([ConvBlock(in_channels=in_hs,
                                              out_channels=out_hs,
                                              conv_kernel_size=conv_kernel_size,
                                              conv_stride=conv_stride,
                                              dropout_p=dropout_p,
                                              pool_kernel_size=pool_kernel_size)
                                    for in_hs, out_hs in in_out_channels])
        
        self.dense1 = nn.Linear(sqz_seq_len, sqz_seq_len)
        self.dropout = nn.Dropout(dropout_p)
        self.dense2 = nn.Linear(sqz_seq_len, sqz_seq_len)
        self.dense3 = nn.Linear(sqz_seq_len, self.n_output)
    
    def forward(self, input):
        """
        :param input: (batch_size, seq_len, 1)
        :return:
        """
        input = input.permute(0, 2, 1)
        # (batch_size, 1, seq_len)
        # begin convolutional blocks...
        for i, conv in enumerate(self.convs):
            input = conv(input)
        
        # input = (batch_size, 1, res_len)
        input = input.squeeze(1)
        pred = F.relu(self.dropout(self.dense1(input)))
        pred = F.relu(self.dropout(self.dense2(pred)))
        output = F.log_softmax(self.dense3(pred), dim=-1)
        return output
    
    @property
    def name(self):
        return 'Small-len{}-CONV-chan{}-krs{}-str{}-nl{}-pkrs{}'.format(
            self.seq_len,
            self.channel_size,
            self.conv_kernel_size,
            self.conv_stride,
            self.num_layers,
            self.pool_kernel_size
        )


class ConvBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels, conv_kernel_size, conv_stride, dropout_p,
                 pool_kernel_size):
        super().__init__()
        self.kernel_size = conv_kernel_size
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
        # pad = int((self.kernel_size - 1) / 2)
        # input = F.pad(input, pad=(pad, pad), value=-1, mode='constant')
        output = self.conv1d(input)
        # (batch_size, out_channels, seq_len)
        output = self.batch_norm(output)
        output = F.relu(output)
        output = self.dropout(output)
        output = self.pooling(output)
        return output

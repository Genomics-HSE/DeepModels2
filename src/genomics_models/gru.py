import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .conv_layer import ConvLayer


class Model:
    def __new__(cls, args):
        return EncoderGRU(seq_len=args.seq_len,
                          input_size=args.input_size,
                          out_channels=args.out_channels,
                          kernel_size=args.kernel_size,
                          hidden_size=args.hidden_size,
                          num_layers=args.num_layers,
                          batch_first=args.batch_first,
                          bidirectional=args.bidirectional,
                          dropout=args.dropout,
                          n_output=args.n_output,
                          conv_n_layers=args.conv_n_layers).to(args.device)
    
    # @staticmethod
    # @property
    # def name(args):
    #     return 'GRU-sl{}-tl{}-hs{}-nl{}'.format(args.seq_len,
    #                                             args.tgt_len,
    #                                             args.hidden_size,
    #                                             args.num_layers)


class EncoderGRU(pl.LightningModule):
    def __init__(self, seq_len, input_size, out_channels, kernel_size,
                 hidden_size, num_layers, batch_first, bidirectional,
                 dropout, n_output, conv_n_layers):
        super().__init__()
        # conv
        self.conv1d = torch.nn.Conv1d(in_channels=input_size,
                                      out_channels=hidden_size,
                                      kernel_size=kernel_size,
                                      stride=1)
        
        self.batch_norm = torch.nn.BatchNorm1d(num_features=hidden_size)
        self.dropout0 = torch.nn.Dropout(dropout)
        
        self.kernel_size = kernel_size
        self.inp_seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_output = n_output
        
        self.conv_n_layers = conv_n_layers
        self.convs = nn.ModuleList([ConvLayer(in_channels=hidden_size,
                                              out_channels=hidden_size,
                                              kernel_size=kernel_size,
                                              stride=1,
                                              padding=(kernel_size - 1) // 2,
                                              dropout=dropout)
                                    for _ in range(conv_n_layers)])
        
        self.gru = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=batch_first,
                          bidirectional=bidirectional,
                          dropout=dropout
                          )
        self.dense1 = nn.Linear((1 + bidirectional) * hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.dense3 = nn.Linear(hidden_size // 2, n_output)
    
    def get_init_state(self, batch_size):
        return torch.Tensor([])
    
    def forward(self, input):
        """
        :param input: (batch_size, pad_seq_len)
        :return:
        """
        input = input.float()
        input = input.unsqueeze(1)
        # (batch_size, input_channels, pad_seq_len)
        output = self.conv1d(input)
        # (batch_size, out_channels, seq_len)
        output = self.batch_norm(output)
        output = F.relu(output)
        output = self.dropout0(output)
        
        # convolutional block
        for conv in self.convs:
            output = conv(output)
        
        output = output.permute(0, 2, 1)
        # (batch_size, seq_len, out_channels)
        output, _ = self.gru(output)
        
        output = self.dropout1(F.relu(self.dense1(output)))
        output = self.dropout2(F.relu(self.dense2(output)))
        output = F.log_softmax(self.dense3(output), dim=-1)
        
        return output,
    
    @property
    def name(self):
        return 'CONV{}-GRU-sl{}-ker{}-hs{}-nl{}'.format(self.conv_n_layers,
                                                        self.inp_seq_len,
                                                        self.kernel_size,
                                                        self.hidden_size,
                                                        self.num_layers
                                                        )

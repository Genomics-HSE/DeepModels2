import torch
import torch.nn as nn
import torch.nn.functional as F

from genomics_utils import LightningModuleExtended


class Model:
    def __new__(cls, args):
        return EncoderGRU(seq_len=args.seq_len,
                          input_size=args.input_size,
                          hidden_size=args.hidden_size,
                          num_layers=args.num_layers,
                          batch_first=args.batch_first,
                          bidirectional=args.bidirectional,
                          dropout=args.dropout,
                          n_output=args.n_output
                          )


class EncoderGRU(LightningModuleExtended):
    def __init__(self, seq_len, input_size, hidden_size, num_layers,
                 batch_first, bidirectional, dropout, n_output):
        super().__init__()
        self.save_hyperparameters()
        
        self.inp_seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.n_output = n_output
        
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=batch_first,
                          bidirectional=bidirectional,
                          dropout=dropout
                          )
        
        self.dense1 = nn.Linear((1 + bidirectional) * hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.dense3 = nn.Linear(hidden_size, n_output)
    
    def forward(self, input):
        """
        :param input: (batch_size, seq_len, 1), the last dim is input dim
        :return:
        """
        # (batch_size, seq_len, 1)
        print("kotok")
        print("kotok")
        print("kotok")
        _, hidden = self.gru(input)
        num_layers_and_bidir, batch, hidden_size = hidden.shape
        hidden = hidden.view(self.num_layers, 1 + self.bidirectional, batch, hidden_size)

        # take only output from the last layers
        hidden = hidden[-1]
        
        # hidden has size [bidir, batch, hidden_size]
        hidden = hidden.permute(1, 0, 2)
        # hidden has size [batch, bidir, hidden_size]
        hidden = hidden.contiguous()
        hidden = hidden.view(hidden.shape[0], -1)
        # hidden has size [batch, bidir * hidden_size]
        
        hidden = self.dropout1(F.relu(self.dense1(hidden)))
        hidden = self.dropout2(F.relu(self.dense2(hidden)))
        hidden = F.log_softmax(self.dense3(hidden), dim=-1)
        return hidden
    
    @property
    def name(self):
        return 'GRU-Gen{}-hs{}-nl{}-bidir{}'.format(self.n_output,
                                                    self.hidden_size,
                                                    self.num_layers,
                                                    self.bidirectional
                                                    )

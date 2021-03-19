import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from genomics_utils import LightningModuleExtended


class Model:
    def __new__(cls, args):
        return Encoder(seq_len=args.seq_len,
                       input_size=args.input_size,
                       hidden_size=args.hidden_size,
                       num_layers=args.num_layers,
                       batch_first=args.batch_first,
                       bidirectional=args.bidirectional,
                       dropout=args.dropout,
                       n_output=args.n_output,
                       t_bptt=args.truncated_bptt_steps
                       )


class GruDummyInput(nn.Module):
    def __init__(self, gru_module):
        super().__init__()
        self.gru_module = gru_module
    
    def forward(self, input, hiddens, dummy_input=None):
        assert dummy_input is not None
        output, hiddens = self.gru_module(input, hiddens)
        return output, hiddens


class Encoder(LightningModuleExtended):
    def __init__(self, seq_len, input_size, hidden_size, num_layers,
                 batch_first, bidirectional, dropout, n_output, t_bptt):
        super().__init__()
        self.save_hyperparameters()
        
        self.inp_seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.n_output = n_output
        self.t_bptt = t_bptt
        
        gru_module = nn.GRU(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=batch_first,
                            bidirectional=bidirectional,
                            dropout=dropout
                            )
        
        self.gru = GruDummyInput(gru_module)
        
        self.dense1 = nn.Linear((1 + bidirectional) * hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(hidden_size, n_output)
        
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
    
    def forward(self, input, hiddens):
        """
            :param input: (batch_size, seq_len, 1)
            :return:
            """
        input = input.float()
        output, hiddens = cp.checkpoint(self.gru, *(input, hiddens, self.dummy_tensor))
        
        output = self.dropout1(F.relu(self.dense1(output)))
        output = F.log_softmax(self.dense2(output), dim=-1)
        return output, hiddens
    
    @property
    def name(self):
        return 'Gen{}-tbptt{}-GRU-sl{}-hs{}-nl{}-dir{}'.format(self.n_output,
                                                               self.t_bptt,
                                                               self.inp_seq_len,
                                                               self.hidden_size,
                                                               self.num_layers,
                                                               1 + int(self.bidirectional)
                                                               )

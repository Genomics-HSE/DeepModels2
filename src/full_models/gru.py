import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from genomics_utils import LightningModuleExtended


class Model:
    def __new__(cls, args):
        return Encoder(seq2seq=args.seq2seq,
                       seq_len=args.seq_len,
                       squeeze=args.squeeze,
                       input_size=args.input_size,
                       hidden_size=args.hidden_size,
                       num_layers=args.num_layers,
                       batch_first=args.batch_first,
                       bidirectional=args.bidirectional,
                       dropout=args.dropout,
                       n_output=args.n_class,
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
    def __init__(self, seq2seq, seq_len, squeeze, input_size, hidden_size, num_layers,
                 batch_first, bidirectional, dropout, n_output, t_bptt):
        super().__init__()
        self.save_hyperparameters()
        
        self.seq2seq = seq2seq
        self.seq_len = seq_len
        self.squeeze = squeeze
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
        #print(len(input), type(input))
        #print(len(input[0]), type(input[0]))
        lengths = [len(x) for x in input]
        input = [torch.from_numpy(x) for x in input]
        input = torch.nn.utils.rnn.pad_sequence(input, batch_first=True)
        input = input.unsqueeze(2)
        input = torch.nn.utils.rnn.pack_padded_sequence(input,
                                                        batch_first=True,
                                                        lengths=lengths)
        
        # print(input.shape)
        
        input = input.float()
        output, hiddens = cp.checkpoint(self.gru, *(input, hiddens, self.dummy_tensor))
        # output, hiddens = self.gru(input, hiddens, self.dummy_tensor)
        
        output = self.dropout1(F.relu(self.dense1(output)))
        output = F.log_softmax(self.dense2(output), dim=-1)
        return output, hiddens
    
    @property
    def name(self):
        return 'Gen{}-sqz{}-tbptt{}-GRU-sl{}-hs{}-nl{}-dir{}'.format(self.n_output,
                                                                     int(self.squeeze),
                                                                     self.t_bptt,
                                                                     self.seq_len,
                                                                     self.hidden_size,
                                                                     self.num_layers,
                                                                     1 + int(self.bidirectional)
                                                                     )

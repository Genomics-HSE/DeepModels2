import torch
import torch.nn as nn
import torch.nn.functional as F


class Model:
    def __new__(cls, args):
        return EncoderGRUOneDirectional(seq_len=args.seq_len,
                                        input_size=args.input_size,
                                        hidden_size=args.hidden_size,
                                        num_layers=args.num_layers,
                                        batch_first=args.batch_first,
                                        dropout=args.dropout,
                                        n_output=args.n_output).to(args.device)


class EncoderGRUOneDirectional(nn.Module):
    def __init__(self, seq_len, input_size, hidden_size, num_layers,
                 batch_first, dropout, n_output):
        super().__init__()
        self.inp_seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_output = n_output
        
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=batch_first,
                          dropout=dropout
                          )
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dense3 = nn.Linear(hidden_size // 2, n_output)
    
    def forward(self, input, hidden_state):
        """
        :param input: (batch_size, seq_len, input_size)
        :param hidden_state: (num_layers, batch, hidden_size)
        :return:
        """
        input = input.float()
        input = input.unsqueeze(-1)
        outputs, hidden_state = self.gru(input, hidden_state)
        
        pred = F.relu(self.dense1(outputs))
        pred = F.relu(self.dense2(pred))
        pred = F.log_softmax(self.dense3(pred), dim=-1)
        
        return pred, hidden_state
    
    def get_init_state(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
    
    @property
    def name(self):
        return 'GRU-OneDir-sl{}-hs{}-nl{}'.format(self.inp_seq_len,
                                                  self.hidden_size,
                                                  self.num_layers)

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model:
    def __new__(cls, args):
        return EncoderGRU(seq_len=args.seq_len,
                          tgt_len=args.tgt_len,
                          input_size=args.input_size,
                          hidden_size=args.hidden_size,
                          num_layers=args.num_layers,
                          batch_first=args.batch_first,
                          bidirectional=args.bidirectional,
                          dropout=args.dropout,
                          n_output=args.n_output).to(args.device)


# @staticmethod
# @property
# def name(args):
# 	return 'GRU-sl{}-tl{}-hs{}-nl{}'.format(args.seq_len,
# 											args.tgt_len,
# 											args.hidden_size,
# 											args.num_layers)


class EncoderGRU(nn.Module):
    def __init__(self, seq_len, tgt_len, input_size, hidden_size, num_layers,
                 batch_first, bidirectional, dropout, n_output):
        super().__init__()
        self.inp_seq_len = seq_len
        self.tgt_len = tgt_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=batch_first,
                          bidirectional=bidirectional,
                          dropout=dropout
                          )
        self.dense1 = nn.Linear((1 + bidirectional) * hidden_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dense3 = nn.Linear(hidden_size // 2, n_output)
    
    def forward(self, input):
        """
        :param input: (batch_size, seq_len, input_size)
        :return:
        """
        input = input.float()
        input = input.unsqueeze(-1)
        outputs, _ = self.gru(input)
        
        start_pos = int((self.inp_seq_len - self.tgt_len) / 2)
        reduced_outputs = outputs.narrow(1, start_pos, self.tgt_len)
        
        pred = F.relu(self.dense1(reduced_outputs))
        pred = F.relu(self.dense2(pred))
        pred = self.dense3(pred)
        
        pred = pred.permute(0, 2, 1)
        return pred
    
    @property
    def name(self):
        return 'GRU-sl{}-tl{}-hs{}-nl{}'.format(self.inp_seq_len,
                                                self.tgt_len,
                                                self.hidden_size,
                                                self.num_layers)

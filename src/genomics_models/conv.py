import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_layer import ConvLayer


class Model:
    def __new__(cls, args):
        return EncoderConv(seq_len=args.seq_len,
                           n_output=args.n_output,
                           emb_size=args.emb_size_conv,
                           hidden_size=args.hidden_size_conv,
                           n_layers=args.n_layers_conv,
                           kernel_size=args.kernel_size,
                           dropout=args.dropout_conv,
                           scale=args.scale_conv,
                           device=args.device).to(args.device)


class EncoderConv(nn.Module):
    def __init__(self, seq_len, n_output, emb_size, hidden_size, n_layers,
                 kernel_size, dropout, scale, device):
        super().__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd!"
        _ = emb_size
        self.device = device
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.scale = scale
        
        self.scale = torch.sqrt(torch.FloatTensor([self.scale])).to(device)
        
        self.embedding = ConvLayer(in_channels=1,
                                   out_channels=hidden_size,
                                   kernel_size=kernel_size,
                                   stride=1,
                                   padding=0,
                                   dropout=dropout
                                   )
        
        self.convs = nn.ModuleList([ConvLayer(in_channels=hidden_size,
                                              out_channels=hidden_size,
                                              kernel_size=kernel_size,
                                              stride=1,
                                              padding=(kernel_size - 1) // 2,
                                              dropout=dropout)
                                    for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.dense1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dense2 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.dense3 = nn.Linear(hidden_size // 2, n_output)
    
    def get_init_state(self, batch_size):
        return torch.Tensor([])
    
    def forward(self, input):
        """
        :param input: (batch_size, seq_len)
        :return:
        """
        
        input = input.float()
        input = input.unsqueeze(1)
        # (batch_size, 1, pad_seq_len)
        input = self.embedding(input)
        # input = (batch_size, out_channels, seq_len)
        
        # begin convolutional blocks...
        for i, conv in enumerate(self.convs):
            input = conv(input)
        
        # input = (batch_size, out_channels, seq_len)
        
        pred = F.relu(self.dropout(self.dense1(input)))
        pred = F.relu(self.dropout(self.dense2(pred)))
        output = F.log_softmax(self.dense3(pred), dim=-1)
        return output,
    
    @property
    def name(self):
        return 'CONV-sl{}-hs{}-kl{}-nl{}'.format(self.seq_len,
                                                      self.hidden_size,
                                                      self.kernel_size,
                                                      self.n_layers)

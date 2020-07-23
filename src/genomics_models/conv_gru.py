import torch
import torch.nn as nn

from .gru import EncoderGRU
from .conv_layer import ConvLayer

## TODO
class Model:
    def __new__(cls, args):
        return EncoderConvGRU(
            seq_len=args.seq_len,
            vocab_size=args.n_token_in,
            hidden_size=args.hidden_size_bert,
            num_hidden_layers=args.num_layers_bert,
            num_attention_heads=args.num_attention_heads,
            intermediate_size=args.intermediate_size,
            num_labels=args.n_output
        ).to(args.device)


class EncoderConvGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = ConvLayer(in_channels=1,
                                   out_channels=hidden_size,
                                   kernel_size=kernel_size,
                                   stride=1,
                                   padding=0,
                                   dropout=dropout
                                   )
    
    def forward(self):
    
    
    
    

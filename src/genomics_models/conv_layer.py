import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                )
        
        self.batch_norm = nn.BatchNorm1d(num_features=out_channels)
        self.dropout0 = nn.Dropout(dropout)
    
    def forward(self, input):
        # (batch_size, input_channels, pad_seq_len)
        output = self.conv1d(input)
        # (batch_size, out_channels, seq_len)
        output = self.batch_norm(output)
        output = F.relu(output)
        output = self.dropout0(output)
        
        return output

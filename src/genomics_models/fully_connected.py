import torch.nn as nn
import torch.nn.functional as F


class FullConnected(nn.Module):
    def __init__(self, hidden_size, n_output, dropout):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dense2 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.dense3 = nn.Linear(hidden_size // 2, n_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        """
        
        :param input: # (batch_size, seq_len, hidden_size)
        :return:
        """
        output = F.relu(self.dropout(self.dense1(input)))
        output = F.relu(self.dropout(self.dense2(output)))
        output = F.log_softmax(self.dense3(output), dim=-1)
        return output

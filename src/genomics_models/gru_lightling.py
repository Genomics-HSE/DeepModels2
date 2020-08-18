import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from .conv_layer import ConvLayer


class Model:
    def __new__(cls, args):
        return LightEncoderGRU(seq_len=args.seq_len,
                               input_size=args.input_size,
                               out_channels=args.out_channels,
                               kernel_size=args.kernel_size,
                               hidden_size=args.hidden_size,
                               num_layers=args.num_layers,
                               batch_first=args.batch_first,
                               bidirectional=args.bidirectional,
                               dropout=args.dropout,
                               n_output=args.n_output,
                               conv_n_layers=args.conv_n_layers,
                               lr=0.2)
        # .to(args.device)


class LightEncoderGRU(pl.LightningModule):
    def __init__(self, seq_len, input_size, out_channels, kernel_size,
                 hidden_size, num_layers, batch_first, bidirectional,
                 dropout, n_output, conv_n_layers, lr):
        super(LightEncoderGRU, self).__init__()
        ###
        self.lr = lr

        self.kernel_size = kernel_size
        self.inp_seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_output = n_output
        self.conv_n_layers = conv_n_layers

        self.model = nn.Sequential(
            torch.nn.Conv1d(in_channels=input_size,
                            out_channels=hidden_size,
                            kernel_size=kernel_size,
                            stride=1),
            torch.nn.BatchNorm1d(num_features=hidden_size),
            torch.nn.Dropout(dropout),
            *[ConvLayer(in_channels=hidden_size,
                        out_channels=hidden_size,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=(kernel_size - 1) // 2,
                        dropout=dropout)
                for _ in range(conv_n_layers)],
            nn.GRU(input_size=hidden_size,
                   hidden_size=hidden_size,
                   num_layers=num_layers,
                   batch_first=batch_first,
                   bidirectional=bidirectional,
                   dropout=dropout
                   ),  # maybe need to stop there
            nn.Linear((1 + bidirectional) * hidden_size, hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, n_output)
        )

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

        # output=self.dropout1(F.relu(self.dense1(output)))  # Важно, надо переделать
        # output=self.dropout2(F.relu(self.dense2(output)))
        output = F.log_softmax(self.model(input), dim=-1)

        return output,

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.kl_div(self(x), y)  # KLDivLoss
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.classifier.parameters(),
            lr=self.lr
        )

    @property
    def name(self):
        return 'CONV{}-GRU-sl{}-ker{}-hs{}-nl{}'.format(self.conv_n_layers,
                                                        self.inp_seq_len,
                                                        self.kernel_size,
                                                        self.hidden_size,
                                                        self.num_layers
                                                        )

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model:
    def __new__(cls, args):
        return EncoderConv(seq_len=args.seq_len,
                           tgt_len=args.tgt_len,
                           n_token_in=args.n_token_in,
                           n_output=args.n_output,
                           emb_size=args.emb_size,
                           hidden_size=args.hidden_size,
                           n_layers=args.n_layers,
                           kernel_size=args.kernel_size,
                           dropout=args.dropout,
                           device=args.device).to(args.device)


class EncoderConv(nn.Module):
    
    def __init__(self, seq_len, tgt_len, n_token_in, n_output, emb_size, hidden_size, n_layers,
                 kernel_size, dropout, device):
        super().__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd!"
        self.device = device
        self.seq_len = seq_len
        self.tgt_len = tgt_len
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        
        self.tok_embedding = nn.Embedding(n_token_in, emb_size)
        self.pos_embedding = nn.Embedding(seq_len, emb_size)
        
        self.emb2hid = nn.Linear(emb_size, hidden_size)
        self.hid2emb = nn.Linear(hidden_size, emb_size)
        
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=hidden_size,
                                              out_channels=2 * hidden_size,
                                              kernel_size=kernel_size,
                                              padding=(kernel_size - 1) // 2)
                                    for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.dense1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dense2 = nn.Linear(hidden_size // 2, n_output)
    
    def forward(self, src):
        # src = [batch size, src len]
        
        batch_size = src.shape[0]
        seq_len = src.shape[1]
        
        # create position tensor
        pos = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        # embed tokens and positions
        tok_embedded = self.tok_embedding(src)
        pos_embedded = self.pos_embedding(pos)
        
        embedded = self.dropout(tok_embedded + pos_embedded)
        
        # embedded = [batch size, src len, emb dim]
        
        # pass embedded through linear layer to convert from emb dim to hid dim
        conv_input = self.emb2hid(embedded)
        
        # conv_input = [batch size, src len, hid dim]
        
        # permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)
        
        # conv_input = [batch size, hid dim, src len]
        
        # begin convolutional blocks...
        for i, conv in enumerate(self.convs):
            # pass through convolutional layer
            conved = conv(self.dropout(conv_input))
            
            # conved = [batch size, 2 * hid dim, src len]
            
            # pass through GLU activation function
            conved = F.glu(conved, dim=1)
            
            # conved = [batch size, hid dim, src len]
            
            # apply residual connection
            conved = (conved + conv_input) * self.scale
            
            # conved = [batch size, hid dim, src len]
            
            # set conv_input to conved for next loop iteration
            conv_input = conved
        
        # ...end convolutional blocks
        pred = conved.permute(0, 2, 1)
        # conved = [batch size, src len, hid dim]

        start_pos = int((self.seq_len - self.tgt_len) / 2)
        pred = pred.narrow(1, start_pos, self.tgt_len)
        
        pred = F.relu(self.dense1(pred))
        pred = self.dense2(pred)
        
        return pred.permute(0, 2, 1)

    @property
    def name(self):
        return 'CONV-sl{}-tl{}-hs{}-kl{}-nl{}'.format(self.seq_len,
                                                      self.tgt_len,
                                                      self.hidden_size,
                                                      self.kernel_size,
                                                      self.n_layers)
    

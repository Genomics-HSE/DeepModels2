import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertForTokenClassification
import pytorch_lightning as pl

from .conv_layer import ConvLayer


class Model:
    def __new__(cls, args):
        return EncoderBert(
            seq_len=args.seq_len,
            vocab_size=args.n_token_in,
            hidden_size=args.hidden_size_bert,
            num_hidden_layers=args.num_layers_bert,
            num_attention_heads=args.num_attention_heads,
            intermediate_size=args.intermediate_size,
            num_labels=args.n_output,
            kernel_size=args.kernel_size_b,
        ).to(args.device)


class EncoderBert(pl.LightningModule):
    def __init__(self, seq_len=10000,
                 vocab_size=2, hidden_size=196, num_hidden_layers=12,
                 num_attention_heads=12, intermediate_size=1024, hidden_act='relu',
                 hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
                 type_vocab_size=1, initializer_range=0.02, layer_norm_eps=1e-12,
                 pad_token_id=0, gradient_checkpointing=False, num_labels=20,
                 kernel_size=51, **kwargs):
        super().__init__()
        
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_output = num_labels
        self.kernel_size = kernel_size
        
        config = BertConfig(
            vocab_size=vocab_size, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads, intermediate_size=intermediate_size,
            hidden_act=hidden_act, hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=seq_len, type_vocab_size=type_vocab_size,
            initializer_range=initializer_range, layer_norm_eps=layer_norm_eps, pad_token_id=pad_token_id,
            gradient_checkpointing=gradient_checkpointing, num_labels=num_labels, **kwargs
        )
        
        self.embedding = ConvLayer(in_channels=1,
                                   out_channels=hidden_size,
                                   kernel_size=kernel_size,
                                   stride=1,
                                   padding=0,
                                   dropout=0.1
                                   )
        
        self.bert_model = BertForTokenClassification(config)
    
    def get_init_state(self, batch_size):
        return torch.Tensor([])
    
    def forward(self, input):
        """
        :param input: (batch_size, seq_len)
        :return:
        """
        input = input.float()
        input = input.unsqueeze(1)
        # (batch_size, 1, seq_len)
        input = self.embedding(input)
        
        input = input.permute(0, 2, 1)
        output, *h = self.bert_model(inputs_embeds=input)
        output = F.log_softmax(output, dim=-1)
        return output,
    
    @property
    def name(self):
        return 'BERT-sl{}-ker{}-hs{}-nl{}-ah{}'.format(self.seq_len,
                                                       self.kernel_size,
                                                       self.hidden_size,
                                                       self.num_hidden_layers,
                                                       self.num_attention_heads)

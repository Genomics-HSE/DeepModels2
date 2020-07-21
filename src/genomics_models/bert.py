import torch
import torch.nn as nn
from transformers import BertConfig, BertForTokenClassification


class Model:
    def __new__(cls, args):
        return EncoderBert(
            seq_len=args.seq_len,
            vocab_size=args.n_token_in,
            hidden_size=args.hidden_size_bert,
            num_hidden_layers=args.num_layers_bert,
            num_attention_heads=args.num_attention_heads,
            intermediate_size=args.intermediate_size,
            num_labels=args.n_output
        ).to(args.device)


class EncoderBert(nn.Module):
    def __init__(self, seq_len=10000,
                 vocab_size=2, hidden_size=196, num_hidden_layers=12,
                 num_attention_heads=12, intermediate_size=1024, hidden_act='relu',
                 hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
                 type_vocab_size=1, initializer_range=0.02, layer_norm_eps=1e-12,
                 pad_token_id=0, gradient_checkpointing=False, num_labels=20, **kwargs):
        super().__init__()
        
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_output = num_labels
        
        config = BertConfig(
            vocab_size=vocab_size, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads, intermediate_size=intermediate_size,
            hidden_act=hidden_act, hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=seq_len, type_vocab_size=type_vocab_size,
            initializer_range=initializer_range, layer_norm_eps=layer_norm_eps, pad_token_id=pad_token_id,
            gradient_checkpointing=gradient_checkpointing, num_labels=num_labels, **kwargs
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
        input = input.unsqueeze(2)
        # (batch_size, seq_len, 1)
        output, *h = self.bert_model(inputs_embeds=input)
        return output,
    
    @property
    def name(self):
        return 'BERT-sl{}-hs{}-nl{}-ah{}'.format(self.seq_len,
                                                 self.hidden_size,
                                                 self.num_hidden_layers,
                                                 self.num_attention_heads)

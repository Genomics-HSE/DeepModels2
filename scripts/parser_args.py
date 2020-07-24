
def gru_add_arguments(parser):
    parser.add_argument('--out_channels', type=int, default=128)
    parser.add_argument('--kernel_size', type=int, default=1001)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--batch_first', type=bool, default=True)
    parser.add_argument('--bidirectional', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--conv_n_layers', type=int)
    

def conv_add_arguments(parser):
    parser.add_argument('--hidden_size_conv', type=int)
    parser.add_argument('--emb_size_conv', type=int)
    parser.add_argument('--kernel_size', type=int)
    parser.add_argument('--n_layers_conv', type=int)
    parser.add_argument('--dropout_conv', type=float)
    parser.add_argument('--scale_conv', type=int)


def bert_add_arguments(parser):
    parser.add_argument("--n_token_in", type=int)
    parser.add_argument("--hidden_size_bert", type=int)
    parser.add_argument("--num_layers_bert", type=int)
    parser.add_argument("--num_attention_heads", type=int)
    parser.add_argument("--intermediate_size", type=int)
    parser.add_argument("--hidden_dropout_prob", type=float)
    parser.add_argument("--attention_probs_dropout_prob", type=float)
    parser.add_argument("--type_vocab_size", type=int)
    parser.add_argument("--initializer_range", type=float)
    parser.add_argument("--layer_norm_eps", type=float)
    parser.add_argument("--pad_token_id", type=int)
    parser.add_argument("--gradient_checkpointing", type=bool)
    parser.add_argument("--kernel_size_b", type=int)


def conv_bert_add_arguments(parser):
    conv_add_arguments(parser)
    bert_add_arguments(parser)


def gru_one_dir_add_arguments(parser):
    parser.add_argument('--input_size', type=int, default=1)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--batch_first', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.1)
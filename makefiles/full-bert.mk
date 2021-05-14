.PHONY:	bert bert-train bert-test

########################
#       BERT MODEL     #
########################

bert: bert-train bert-test

bert-vars = hidden_size_bert=504 \
			num_layers_bert=12 \
			num_attention_heads=12 \
			intermediate_size=1024 \
			hidden_dropout_prob=0.1 \
			attention_probs_dropout_prob=0.1 \
			type_vocab_size=1 \
			initializer_range=0.02 \
			layer_norm_eps=1e-12 \
			pad_token_id=0 \
			gradient_checkpointing=False \
			kernel_size=101

$(call assign-vars, bert-train bert-test, $(bert-vars))

bert-args = --n_token_in=$(n_token_in) --hidden_size_bert=$(hidden_size_bert) \
            --num_layers_bert=$(num_layers_bert) --num_attention_heads=$(num_attention_heads) \
            --intermediate_size=$(intermediate_size) --hidden_dropout_prob=$(hidden_dropout_prob) \
		    --attention_probs_dropout_prob=$(attention_probs_dropout_prob) --type_vocab_size=$(type_vocab_size) \
		    --initializer_range=$(initializer_range) --layer_norm_eps=$(layer_norm_eps) \
		    --pad_token_id=$(pad_token_id) --gradient_checkpointing=$(gradient_checkpointing) \
		    --kernel_size_b=$(kernel_size)

bert-train:
	$(train) bert $(bert-args)

bert-test:
	$(test) bert $(bert-args)
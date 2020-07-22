SHELL := bash
assign-vars = $(foreach A,$2,$(eval $1: $A))
.PHONY:	test hse-run test_data clean-output
.PHONY:	gru gru-test gru-train
.PHONY:	bert bert-train bert-test

DATA = data/micro_data
OUTPUT = output/
SEED = 42
BATCH_SIZE = 8
LOGGER = local
PROJECT = population-genomics
WORKSPACE = kenenbek
OFFLINE = true

padding=0
n_token_in=2
N_CLASS=20
lr=0.001


ifdef FAST_RUN
	SEQ_LEN=10
	N_EPOCHS=1
	DEVICE=cpu
	NUM_WORKERS=1
else
	SEQ_LEN=5000
	N_EPOCHS=50
	DEVICE=cuda
	NUM_WORKERS=8
endif

launcher = @python scripts/main.py \
		  --device=$(DEVICE) --data=$(DATA) --output=$(OUTPUT) --logger=$(LOGGER) --offline=$(OFFLINE) \
		  --project $(PROJECT) --workspace $(WORKSPACE) --batch_size=$(BATCH_SIZE) \
		  --seq_len=$(SEQ_LEN) --padding=$(PAD) --n_output=$(N_CLASS) --input_size=$(input_size) \
		  --n_token_in=$(n_token_in)

train = $(launcher) \
        --action=train --seed=$(SEED) --epochs=$(N_EPOCHS) --lr=$(lr)


test = $(launcher) \
		--action=test

########################
#       RNN MODEL      #
########################

gru: gru-train gru-test

gru-vars = hidden_size=256 \
		num_layers=4 \
		batch_first=true \
		bidirectional=true \
		dropout=0.1

$(call assign-vars, gru-train gru-test, $(gru-vars))

gru-args = --hidden_size=$(hidden_size) --num_layers=$(num_layers) \
           --batch_first=$(batch_first) --bidirectional=$(bidirectional) --dropout=$(dropout)

gru-train:
	$(train) gru $(gru-args)

gru-test:
	$(test) gru $(gru-args)

########################
# CONVOLUTIONAL MODEL  #
########################

conv: conv-train conv-test

conv-vars = hidden_size_conv=32 \
			emb_size_conv=16 \
			kernel_size=4097 \
			n_layers_conv=6 \
			dropout_conv=0.1

conv-args = --hidden_size_conv=$(hidden_size_conv) --emb_size_conv=$(emb_size_conv) \
            	  --kernel_size=$(kernel_size) --n_layers=$(n_layers_conv) --dropout=$(dropout_conv)

$(call assign-vars, conv-train, $(conv-vars))

conv-train:
	$(train) conv $(conv-args)

conv-test:
	$(test) conv $(conv-args)

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
			gradient_checkpointing=False

$(call assign-vars, bert-train bert-test, $(bert-vars))

bert-args = --input_size=$(input_size) --n_token_in=$(n_token_in) --hidden_size_bert=$(hidden_size_bert) \
            --num_layers_bert=$(num_layers_bert) --num_attention_heads=$(num_attention_heads) \
            --intermediate_size=$(intermediate_size) --hidden_dropout_prob=$(hidden_dropout_prob) \
		    --attention_probs_dropout_prob=$(attention_probs_dropout_prob) --type_vocab_size=$(type_vocab_size) \
		    --initializer_range=$(initializer_range) --layer_norm_eps=$(layer_norm_eps) \
		    --pad_token_id=$(pad_token_id) --gradient_checkpointing=$(gradient_checkpointing)

bert-train:
	$(train) bert $(bert-args)

bert-test:
	$(test) bert $(bert-args)



test_data:
	@python scripts/data_test.py $(DATA_PATH)

clean-output:
	rm -rf output/*


TARGET=gru
GPU=1
CPU=2
T=600
hse-run:
	echo "#!/bin/bash" > tmp_script.sh; \
	make $(TARGET) --just-print --dry-run -s >> tmp_script.sh;
	sbatch --gpus=$(GPU) -c $(CPU) -t $(T) tmp_script.sh; \
	rm tmp_script.sh

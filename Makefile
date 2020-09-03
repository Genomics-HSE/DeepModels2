SHELL=/bin/bash
assign-vars = $(foreach A,$2,$(eval $1: $A))
.PHONY:	test hse-run test_data clean-output
.PHONY:	gru gru-test gru-train
.PHONY: conv conv-train conv-train
.PHONY:	bert bert-train bert-test

DATA = data/micro_data
OUTPUT = output/
SEED = 42
BATCH_SIZE = 8
LOGGER = local
PROJECT = population-genomics-new
WORKSPACE = kenenbek
OFFLINE = True
RESUME = False
exp_key = ""

padding=0
n_token_in=2
input_size=1
N_CLASS=20
lr=0.001
auto_lr_find=True
shuffle=False

tr_file_first=0
tr_file_last=0
te_file_first=1
te_file_last=1


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

launcher = python scripts/main.py \
		  --device=$(DEVICE) --data=$(DATA) --output=$(OUTPUT) --logger=$(LOGGER) --offline=$(OFFLINE) \
		  --project $(PROJECT) --workspace $(WORKSPACE) \
		  --seq_len=$(SEQ_LEN) --padding=$(padding) --n_output=$(N_CLASS) --input_size=$(input_size) \
		  --n_token_in=$(n_token_in) \
		  --tr_file_first=$(tr_file_first) --tr_file_last=$(tr_file_last) --te_file_first=$(te_file_first) \
		  --te_file_last=$(te_file_last) \
		  --batch_size=$(BATCH_SIZE) --shuffle=$(shuffle) --num_workers=$(NUM_WORKERS)


train = $(launcher) \
        --action=train --seed=$(SEED) --epochs=$(N_EPOCHS) --lr=$(lr) --auto_lr_find=$(auto_lr_find) \
        --resume=$(RESUME)


test = $(launcher) \
		--action=test --exp_key=$(exp_key)

########################
#    Recurrent MODEL   #
########################

gru: gru-train gru-test

gru-vars = hidden_size=256 \
		num_layers=2 \
		batch_first=true \
		bidirectional=true \
		dropout=0.1 \
		conv_n_layers=4 \
		kernel_size=21

$(call assign-vars, gru-train gru-test, $(gru-vars))

gru-args = --hidden_size=$(hidden_size) --num_layers=$(num_layers) \
           --batch_first=$(batch_first) --bidirectional=$(bidirectional) --dropout=$(dropout) \
           --conv_n_layers=$(conv_n_layers) --kernel_size=$(kernel_size)

gru-train:
	$(train) gru $(gru-args)

gru-test:
	$(test) gru $(gru-args)

########################
# CONVOLUTIONAL MODEL  #
########################

conv: conv-train conv-test

conv-vars = hidden_size_conv=256 \
			emb_size_conv=256 \
			kernel_size=51 \
			n_layers_conv=12 \
			dropout_conv=0.1 \
			scale_conv=1

conv-args = --hidden_size_conv=$(hidden_size_conv) --emb_size_conv=$(emb_size_conv) \
            --kernel_size=$(kernel_size) --n_layers_conv=$(n_layers_conv) --dropout=$(dropout_conv) \
            --scale_conv=$(scale_conv)

$(call assign-vars, conv-train conv-test, $(conv-vars))

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

SHELL := bash
assign-vars = $(foreach A,$2,$(eval $1: $A))
.PHONY:  gru gru-test gru-train bert bert-train bert-test test_data clean-output

DATA = data/micro_data
OUTPUT = output/
SEED = 42
BATCH_SIZE = 8
LOGGER = local
PROJECT = population-genomics
WORKSPACE = kenenbek
OFFLINE = true

padding=0

N_CLASS=20
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

gru: gru-train gru-test

$(call assign-vars, gru-train gru-test, input_size=1 \
								out_channels=128 \
								kernel_size=5 \
								hidden_size=256 \
								num_layers=4 \
								batch_first=true \
								bidirectional=true \
								dropout=0.1 \
								lr=0.001 \
								)

train = @python scripts/main.py \
        --device=$(DEVICE) --data=$(DATA) --output=$(OUTPUT) --logger=$(LOGGER) --offline=$(OFFLINE) \
        --project $(PROJECT) --workspace $(WORKSPACE) --batch_size=$(BATCH_SIZE) \
        --seq_len=$(SEQ_LEN) --padding=$(padding) --n_output=$(N_CLASS) \
        --action=train --seed=$(SEED) --epochs=$(N_EPOCHS) --lr=$(lr)


test = @python scripts/main.py \
       	  --device=$(DEVICE) --data=$(DATA) --output=$(OUTPUT) --logger=$(LOGGER) --offline=$(OFFLINE) \
       	  --project $(PROJECT) --workspace $(WORKSPACE) --batch_size=$(BATCH_SIZE) \
       	  --seq_len=$(SEQ_LEN) --padding=$(PAD) --n_output=$(N_CLASS) \
       	  --action=test

gru-train:
	@python scripts/main.py \
  --device=$(DEVICE) --data=$(DATA) --output=$(OUTPUT) --logger=$(LOGGER) --offline=$(OFFLINE) \
  --project $(PROJECT) --workspace $(WORKSPACE) --batch_size=$(BATCH_SIZE) \
  --seq_len=$(SEQ_LEN) --n_output=$(N_CLASS) \
  --action=train --seed=$(SEED) --epochs=$(N_EPOCHS) --lr=$(lr) \
  gru --input_size=$(input_size) --out_channels=$(out_channels) --kernel_size=$(kernel_size) \
  --hidden_size=$(hidden_size) --num_layers=$(num_layers) \
  --batch_first=$(batch_first) --bidirectional=$(bidirectional) --dropout=$(dropout)


gru-test:
	@python scripts/main.py \
	  --device=$(DEVICE) --data=$(DATA) --output=$(OUTPUT) --logger=$(LOGGER) --offline=$(OFFLINE) \
	  --project $(PROJECT) --workspace $(WORKSPACE) --batch_size=$(BATCH_SIZE) \
	  --seq_len=$(SEQ_LEN) --n_output=$(N_CLASS) \
	  --action=test \
	  gru --input_size=$(input_size)  --out_channels=$(out_channels) --kernel_size=$(kernel_size) \
	   --hidden_size=$(hidden_size) --num_layers=$(num_layers) \
	  --batch_first=$(batch_first) --bidirectional=$(bidirectional) --dropout=$(dropout)

gru-one-dir: gru-one-dir-train gru-one-dir-test

$(call assign-vars, gru-one-dir-train gru-one-dir-test, input_size=1 \
								hidden_size=256 \
								num_layers=4 \
								batch_first=true \
								dropout=0.1 \
								lr=0.001 \
								)

gru-one-dir-train:
	$(train) \
  gru_one_dir --input_size=$(input_size) --hidden_size=$(hidden_size) --num_layers=$(num_layers) \
  --batch_first=$(batch_first) --dropout=$(dropout)


gru-one-dir-test:
	$(test) \
	  gru_one_dir --input_size=$(input_size) --hidden_size=$(hidden_size) --num_layers=$(num_layers) \
	  --batch_first=$(batch_first) --dropout=$(dropout)

$(call assign-vars, conv-train, n_token_in=2 \
								hidden_size=32 \
								emb_size=16 \
								kernel_size=4097 \
								n_layers=6 \
								dropout=0.1 \
								lr=0.01 \
								)

conv-train:
	$(train) \
	  conv --n_token_in=$(n_token_in) --hidden_size=$(hidden_size) --emb_size=$(emb_size) \
	  --kernel_size=$(kernel_size) --n_layers=$(n_layers) --dropout=$(dropout)

bert: bert-train bert-test

$(call assign-vars, bert-train bert-test, input_size=1 \
										n_token_in=2 \
										hidden_size_bert=504 \
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
										lr=0.001 \
								)

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

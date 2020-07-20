SHELL := bash
assign-vars = $(foreach A,$2,$(eval $1: $A))
.PHONY:  gru gru-test gru-train test_data clean-output

DATA = data/micro_data
OUTPUT = output/
SEED = 42
BATCH_SIZE = 8
LOGGER = local
PROJECT = population-genomics
WORKSPACE = kenenbek
OFFLINE = true

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
	@python scripts/main.py \
  --device=$(DEVICE) --data=$(DATA) --output=$(OUTPUT) --logger=$(LOGGER) --offline=$(OFFLINE) \
  --project $(PROJECT) --workspace $(WORKSPACE) --batch_size=$(BATCH_SIZE) \
  --seq_len=$(SEQ_LEN) --n_output=$(N_CLASS) \
  --action=train --seed=$(SEED) --epochs=$(N_EPOCHS) --lr=$(lr) \
  gru_one_dir --input_size=$(input_size) --hidden_size=$(hidden_size) --num_layers=$(num_layers) \
  --batch_first=$(batch_first) --dropout=$(dropout)


gru-one-dir-test:
	@python scripts/main.py \
	  --device=$(DEVICE) --data=$(DATA) --output=$(OUTPUT) --logger=$(LOGGER) --offline=$(OFFLINE) \
	  --project $(PROJECT) --workspace $(WORKSPACE) --batch_size=$(BATCH_SIZE) \
	  --seq_len=$(SEQ_LEN) --n_output=$(N_CLASS) \
	  --action=test \
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
	@python scripts/main.py \
	  --device=$(DEVICE) --data=$(DATA) --output=$(OUTPUT) --logger=$(LOGGER) --offline=$(OFFLINE) \
	  --project $(PROJECT) --workspace $(WORKSPACE) --batch_size=$(BATCH_SIZE) \
	  --seq_len=$(SEQ_LEN) --tgt_len=$(TGT_LEN) --n_output=$(N_CLASS) \
	  --action=train --seed=$(SEED) --epochs=$(N_EPOCHS) --lr=$(lr) \
	  conv --n_token_in=$(n_token_in) --hidden_size=$(hidden_size) --emb_size=$(emb_size) \
	  --kernel_size=$(kernel_size) --n_layers=$(n_layers) --dropout=$(dropout)

bert-train:
	@python scripts/main.py \
	  --device=$(DEVICE) --data=$(DATA) --output=$(OUTPUT) --logger=$(LOGGER) --offline=$(OFFLINE) \
	  --project $(PROJECT) --workspace $(WORKSPACE) --batch_size=$(BATCH_SIZE) \
	  --seq_len=$(SEQ_LEN) --n_output=$(N_CLASS) \
	  --action=train --seed=$(SEED) --epochs=$(N_EPOCHS) --lr=$(lr) \
	  bert --input_size=$(input_size)  \
	  --hidden_size=$(hidden_size) --num_layers=$(num_layers) \
	  --batch_first=$(batch_first) --bidirectional=$(bidirectional) --dropout=$(dropout)




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

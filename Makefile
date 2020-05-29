SHELL := bash
assign-vars = $(foreach A,$2,$(eval $1: $A))
.PHONY:  test_data clean-output

DATA = data/micro_data
OUTPUT = output/
SEED = 42
BATCH_SIZE = 16
LOGGER = local
PROJECT = population-genomics
WORKSPACE = kenenbek

N_CLASS=20
ifdef FAST_RUN
	SEQ_LEN=10
    TGT_LEN=4
	N_EPOCHS=1
	DEVICE=cpu
	NUM_WORKERS=1
else
	SEQ_LEN=5000
    TGT_LEN=1000
	N_EPOCHS=50
	DEVICE=cuda
	NUM_WORKERS=8
endif

gru: gru-train gru-test

$(call assign-vars, gru-train gru-test, input_size=1 \
								hidden_size=64 \
								num_layers=4 \
								batch_first=true \
								bidirectional=true \
								dropout=0.1 \
								lr=0.001 \
								)

gru-train:
	@python scripts/main.py \
  --device=$(DEVICE) --data=$(DATA) --output=$(OUTPUT) --logger=$(LOGGER) \
  --project $(PROJECT) --workspace $(WORKSPACE) --batch_size=$(BATCH_SIZE) \
  --seq_len=$(SEQ_LEN) --tgt_len=$(TGT_LEN) --n_output=$(N_CLASS) \
  --action=train --seed=$(SEED) --epochs=$(N_EPOCHS) --lr=$(lr) \
  gru --input_size=$(input_size) --hidden_size=$(hidden_size) --num_layers=$(num_layers) \
  --batch_first=$(batch_first) --bidirectional=$(bidirectional) --dropout=$(dropout)


gru-test:
	@python scripts/main.py \
      --device=$(DEVICE) --data=$(DATA) --output=$(OUTPUT) --logger=$(LOGGER) \
      --project $(PROJECT) --workspace $(WORKSPACE) --batch_size=$(BATCH_SIZE) \
      --seq_len=$(SEQ_LEN) --tgt_len=$(TGT_LEN) --n_output=$(N_CLASS) \
      --action=test \
      gru --input_size=$(input_size) --hidden_size=$(hidden_size) --num_layers=$(num_layers) \
      --batch_first=$(batch_first) --bidirectional=$(bidirectional) --dropout=$(dropout)

test_data:
	@python scripts/data_test.py $(DATA_PATH)

clean-output:
	rm -rf output/*

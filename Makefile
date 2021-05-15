SHELL=/bin/bash
assign-vars = $(foreach A,$2,$(eval $1: $A))
.PHONY: hse-run test_data clean-output

seq2seq = True
squeeze = True
split_sample = False
split_seq_len = 0
n_token_in=2
input_size=1
n_class=20

output = output/
checkpoint_path =
logger = local
cmt_project = population-genomics-new
cmt_workspace = kenenbek
cmt_offline = True
exp_key = ""

seed = 42
resume = False
lr=0.001
auto_lr_find=False
shuffle=False

ifdef FAST_RUN
	seq_len=1000
	sqz_seq_len=100
	n_epochs=1
	device=cpu
	num_workers=1
	cmt_disabled=True
	data = data/len1000-gen20-np
	batch_size=4
	tr_file_first=0
    tr_file_last=4
    te_file_first=5
    te_file_last=8
else
	seq_len=3e5
	sqz_seq_len=3e4
	n_epochs=50
	device=cuda
	num_workers=8
	cmt_disabled=False
	data = ../len3e5-gen32-dem100-np
	batch_size=16
	tr_file_first=0
    tr_file_last=399
    te_file_first=400
    te_file_last=499
endif

launcher = python scripts/main.py \
		  --device=$(device) --data=$(data) --output=$(output) --logger=$(logger) --cmt_offline=$(cmt_offline) \
		  --cmt_project=$(cmt_project) --cmt_workspace=$(cmt_workspace) --cmt_disabled=$(cmt_disabled) \
		  --seq2seq=$(seq2seq) --seq_len=$(seq_len) --squeeze=$(squeeze) --sqz_seq_len=$(sqz_seq_len) \
		  --split_sample=$(split_sample) --split_seq_len=$(split_seq_len) --n_class=$(n_class) \
		  --tr_file_first=$(tr_file_first) --tr_file_last=$(tr_file_last) --te_file_first=$(te_file_first) \
		  --te_file_last=$(te_file_last) \
		  --batch_size=$(batch_size) --shuffle=$(shuffle) --num_workers=$(num_workers)


train = $(launcher) \
        --action=train --seed=$(seed) --epochs=$(n_epochs) --lr=$(lr) --auto_lr_find=$(auto_lr_find) \
        --resume=$(resume)


test = $(launcher) \
		--action=test --exp_key=$(exp_key) --checkpoint_path=$(checkpoint_path)

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
	echo $(model-path)
	make -f $(model-path) $(TARGET) --just-print --dry-run -s >> tmp_script.sh;
	sbatch --gpus=$(GPU) -c $(CPU) -t $(T) tmp_script.sh; \
	rm tmp_script.sh

include Makefile

##################################################################
#    						GRU                                  #
##################################################################

gru: gru-train gru-test

ifdef FAST_RUN
gru-vars = hidden_size=10 \
			   num_layers=2 \
			   batch_first=True \
			   bidirectional=True \
			   dropout=0.1 \
			   truncated_bptt_steps=10
else
gru-vars = hidden_size=256 \
			   num_layers=2 \
			   batch_first=True \
			   bidirectional=True \
			   dropout=0.1 \
			   truncated_bptt_steps=10000
endif

$(call assign-vars, gru-train gru-test\
 					gru-print-args, $(gru-vars))

gru-args = --hidden_size=$(hidden_size) --num_layers=$(num_layers) \
           --batch_first=$(batch_first) --bidirectional=$(bidirectional) --dropout=$(dropout) \
           --truncated_bptt_steps=$(truncated_bptt_steps)

gru-train:
	$(train) gru $(gru-args)

gru-test:
	$(test) gru $(gru-args)

gru-print-args:
	echo $(gru-args)
########################
#    Recurrent MODEL   #
#    CNN + GRU         #
########################

conv-gru: conv-gru-train conv-gru-test

conv-gru-vars = $(gru-vars) \
		        conv_n_layers=4 \
		        kernel_size=21

$(call assign-vars, conv-gru-train conv-gru-test, $(conv-gru-vars))

conv-gru-args = $(gru-args) \
           --conv_n_layers=$(conv_n_layers) --kernel_size=$(kernel_size)

conv-gru-train:
	$(train) conv-gru $(conv-gru-args)

conv-gru-test:
	$(test) conv-gru $(conv-gru-args)
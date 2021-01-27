.PHONY:	gru gru-test gru-train

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
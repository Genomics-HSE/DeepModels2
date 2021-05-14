.PHONY: conv-small conv-train-small conv-test-small conv-print-args
model-path := $(MAKEFILE_LIST)
include Makefile

#####################################################################
#                       CONVOLUTIONAL MODEL                         #
#####################################################################

conv-small: conv-train-small conv-test-small

ifdef FAST_RUN
conv-vars = channel_size=10 \
			conv_kernel_size=5 \
			conv_stride=1 \
			num_layers=2 \
			dropout_p=0.1 \
			pool_kernel_size=5
else
conv-vars = channel_size=32 \
			conv_kernel_size=25 \
			conv_stride=2 \
			num_layers=4 \
			dropout_p=0.1 \
			pool_kernel_size=5
endif

conv-args = --channel_size=$(channel_size) --conv_kernel_size=$(conv_kernel_size) \
            --conv_stride=$(conv_stride) --num_layers=$(num_layers) --dropout_p=$(dropout_p) \
            --pool_kernel_size=$(pool_kernel_size)

$(call assign-vars, conv-train-small conv-test-small conv-print-args, $(conv-vars))

conv-train-small:
	$(train) conv_small $(conv-args)

conv-test-small:
	$(test) conv_small $(conv-args)

conv-print-args:
	@echo $(conv-args)
	@echo $(seed)
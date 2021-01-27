##################################################################
#    			Truncated BPTT GRU                               #
##################################################################

gru-fg-train:
	$(train) gru_fg $(gru-args)

gru-fg-test:
	$(test) gru_fg $(gru-args)
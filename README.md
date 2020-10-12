# Population genomics

## Data generation 

`data_gen_numpy` file has four arguments (`sys.argv`): 

|Argument | Meaning |
| --- | --- |
| sys.argv[1] | Saving directory |
| sys.argv[2] | Number of examples to generate |
| sys.argv[3] |  Genome length |
| sys.argv[4] | Number of generations |

## Models 

- CNN :white_check_mark:
- GRU :white_check_mark:
- LSTM :negative_squared_cross_mark:
- Transformers :white_check_mark:
- CONV + GRU :white_check_mark:


### CNN 
| Parameter | Default value |
| --- | --- |
|hidden_size_conv | 256 | 
|emb_size_conv|256 |
|kernel_size |51 |
|n_layers_conv |12 |
|dropout_conv | 0.1 |
|scale_conv |1 |

### GRU
| Parameter | Default value |
| --- | --- |
|hidden_size | 1 |
|num_layers | 2 |
|batch_first | True |
|bidirectional | True |
|dropout | 0.1 | 

### CNN + GRU 
| Parameter | Default value |
| --- | --- |
|hidden_size|1 |
|num_layers|2 |
|batch_first|True |
|bidirectional|True |
|dropout| 0.1 |
|conv_n_layers| 4 |
|kernel_size| 5 |

### BERT 

| Parameter | Description | Default value |
| --- | --- | --- |
| | | |


## Launcher options  
| Parameter | Default value |
| --- | --- |
|DATA  |  data/micro_data |
|OUTPUT  |  output/ |
|SEED  |  42 |
|BATCH_SIZE  |  8|
|LOGGER  |  local|
|PROJECT  |  population-genomics-new |
|WORKSPACE  |  kenenbek |
|OFFLINE  |  True |
|RESUME  |  False |
|lr | 0.001 |
|auto_lr_find | False |

## Data generation options
| Parameter | Default value |
| --- | --- |
|padding | 0 |
|n_token_in | 2 |
|input_size | 1 |
|N_CLASS | 20 |
|shuffle | False |
|tr_file_first | 0 |
|tr_file_last | 0 |
|te_file_first | 1 |
|te_file_last | 1 |
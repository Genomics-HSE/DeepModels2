from typing import Any
import numpy as np
import torch

from genomics_utils import LightningModuleExtended


class SmallModelTraining(LightningModuleExtended):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.KLDivLoss(reduction='batchmean')
        # self.example_input_array = torch.LongTensor(1, 10).random_(0, 2)
        self.lr = 0.001
    
    def training_step(self, batch, batch_ix):
        X_batch, y_batch = batch
        X_batch = self.pad_input(X_batch, self.sqz_seq_len)
        X_batch = torch.from_numpy(X_batch.astype('float32')).to(self.device)
        
        logits = self.forward(X_batch)
        y_batch = torch.from_numpy(np.vstack(y_batch).astype('float32')).to(self.device)
        loss = self.loss(logits, y_batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return {'loss': loss
                }
    
    def pad_input(self, input, sqz_seq_len):
        # input is a list of numpy arrays
        data = []
        
        for single_genome_distances in input:
            if len(single_genome_distances) > sqz_seq_len:
                new_genome = single_genome_distances[:sqz_seq_len]
            else:
                new_genome = np.pad(single_genome_distances,
                                    mode='constant',
                                    constant_values=-1,
                                    pad_width=(0, sqz_seq_len - len(single_genome_distances)),
                                    )
            data.append(new_genome)
        
        data = np.expand_dims(np.vstack(data), axis=2)
        
        return data

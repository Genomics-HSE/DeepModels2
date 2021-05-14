import os
import collections
from typing import Any
import numpy as np
import torch
import pytorch_lightning as pl

__all__ = [
    'LightningModuleExtended'
]


class LightningModuleExtended(pl.LightningModule):
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def save(self, trainer: pl.Trainer, parameters_path: str):
        if os.environ.get("FAST_RUN") is None or not os.path.exists(parameters_path):
            print('saving to {parameters_path}'.format(parameters_path=parameters_path))
            trainer.save_checkpoint(filepath=parameters_path)
            
            trainer.logger.experiment.log_model(self.name, parameters_path)
    
    #
    # def test_step(self, batch, batch_idx) -> Any:
    #     X_batch, y_batch = batch
    #     logits = self.forward(X_batch)
    #     preds = torch.exp(logits)
    #     preds = torch.flatten(preds, start_dim=0, end_dim=1)
    #
    #     # y_batch = torch.argmax(y_batch, dim=-1)
    #     y = y_batch.flatten()
    #
    #     preds = preds.cpu().detach()
    #     self.logger.log_coalescent_heatmap(self.name, [preds.T, y.T], batch_idx)

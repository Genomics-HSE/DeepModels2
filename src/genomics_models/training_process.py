import os
from typing import Any
import torch
import pytorch_lightning as pl


class LightningModuleExtended(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.KLDivLoss(reduction='batchmean')
        # self.example_input_array = torch.LongTensor(1, 10).random_(0, 2)
        self.lr = 0.001

    def training_step(self, batch, batch_ix):
        X_batch, y_batch = batch
        y_one_hot = one_hot_encoding(y_batch, self.n_output)
        logits = self.forward(X_batch)
        loss = self.loss(logits, y_one_hot)
        result = pl.TrainResult(minimize=loss)
        result.log("train_loss", loss, on_step=True, on_epoch=True)
        return result
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def test_step(self, batch, batch_idx) -> Any:
        X_batch, y_batch = batch
        logits = self.forward(X_batch)
        preds = torch.exp(logits)
        preds = torch.flatten(preds, start_dim=0, end_dim=1)
        y = y_batch.flatten()
        self.logger.log_coalescent_heatmap(self.name, [preds.T, y.T], batch_idx)
        
    def save(self, trainer: pl.Trainer, parameters_path: str):
        if os.environ.get("FAST_RUN") is None or not os.path.exists(parameters_path):
            print('saving to {parameters_path}'.format(parameters_path=parameters_path))
            trainer.save_checkpoint(filepath=parameters_path)
            
            # save to comet-ml
            trainer.logger.experiment.log_model(self.name, parameters_path)


def one_hot_encoding(y_data, num_class):
    batch_size, seq_len = y_data.shape
    y_one_hot = torch.FloatTensor(batch_size, seq_len, num_class)
    
    y_one_hot.zero_()
    y_one_hot.scatter_(2, y_data.unsqueeze(2), 1)
    return y_one_hot
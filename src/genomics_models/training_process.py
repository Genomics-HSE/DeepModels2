import os
import abc
from typing import Any
import torch
import pytorch_lightning as pl
import numpy as np
import tqdm

from genomics_utils import one_hot_encoding


class LightningModuleExtended(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.KLDivLoss(reduction='batchmean')
        self.example_input_array = torch.LongTensor(1, 10).random_(0, 2)

    def training_step(self, batch, batch_ix):
        X_batch, y_batch = batch
        y_one_hot = one_hot_encoding(y_batch, self.n_output)
        logits = self.forward(X_batch)
        loss = self.loss(logits, y_one_hot)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def predict_proba(self, dataset: pl.LightningDataModule, logger: Any, mean=False):
        with torch.no_grad():
            heatmap_predictions = []
            ground_truth = []
            for ix, (X, y) in tqdm(enumerate(dataset.tes)):
                torch.cuda.empty_cache()
                X = torch.from_numpy(X).to(self.device)
                preds = self.forward(X)
                preds = torch.exp(preds)
                preds = torch.flatten(preds, start_dim=0, end_dim=1)
                y = y.flatten()
            
                if mean:
                    heatmap_predictions.append(np.mean(preds.cpu().detach().numpy(), axis=0))
                    ground_truth.append(np.mean(y, axis=0))
                else:
                    preds = preds.cpu().detach().numpy()
                
                    heatmap_predictions.extend(preds)
                    ground_truth.extend(y)
                
                    logger.log_coalescent_heatmap(self.name, [preds.T, y.T], ix)
    
        return np.array(heatmap_predictions).T, np.array(ground_truth).T

    def save(self, parameters_path, quiet):
        if os.environ.get("FAST_RUN") is None or not os.path.exists(parameters_path):
            torch.save(self.state_dict(), parameters_path)
        
            if not quiet:
                print('saving to {parameters_path}'.format(parameters_path=parameters_path))
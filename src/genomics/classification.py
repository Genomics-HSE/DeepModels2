import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class Classifier(object):
    def __init__(self, classifier, logger, device, lr=1e-3):
        self.classifier = classifier
        self.device = device
        self.logger = logger
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            params=self.classifier.parameters(),
            lr=lr
        )
    
    def fit(self, dataloader, n_epochs=32, progress=None):
        if progress is None:
            def progress(x):
                return x
        
        n_batches = len(dataloader)
        
        losses = np.ndarray(shape=(n_epochs, n_batches))
        
        for i in progress(range(n_epochs)):
            for j, (X_batch, y_batch) in enumerate(dataloader):
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()
                logits = self.classifier(X_batch)
                loss = self.loss(logits, y_batch)
                loss.backward()
                self.optimizer.step()
                
                losses[i, j] = loss.item()
            
            self.logger.log_metric("epoch_loss", losses[i].mean())
        return losses
    
    def predict(self, dataloader):
        with torch.no_grad():
            predictions = []
            ground_truth = []
            for j, (X_batch, y_batch) in enumerate(dataloader):
                X_batch = X_batch.to(self.device)
                predictions.append(self.classifier(X_batch))
                ground_truth.append(y_batch)
            return torch.cat(predictions, dim=0).cpu().numpy(), torch.cat(ground_truth, dim=0).cpu().numpy()
    
    def predict_proba(self, datapath, step=50):
        X = torch.load(os.path.join(datapath, "x", "0.pt"))
        y = torch.load(os.path.join(datapath, "y", "0.pt"))
        with torch.no_grad():
            heatmap_predictions = []
            heatmap_predictions_log = []
            ground_truth = []
            for i in tqdm(range(0, len(X), step)):
                X_i = X[i:i+step].unsqueeze(1)
                preds = self.classifier(X_i).squeeze(-1)
                
                preds_soft = F.softmax(preds, dim=-1).numpy()
                preds_log_soft = F.log_softmax(preds, dim=-1).numpy()
                
                heatmap_predictions.append(np.mean(preds_soft, axis=0))
                heatmap_predictions_log.append(np.mean(preds_log_soft, axis=0))
                ground_truth.append(np.mean(y[i:i+step].numpy(), axis=0))
        return np.array(heatmap_predictions).T, np.array(heatmap_predictions_log).T, np.array(ground_truth).T
    
    def save(self, parameters_path, quiet):
        if os.environ.get("FAST_RUN") is None or not os.path.exists(parameters_path):
            torch.save(self.classifier.state_dict(), parameters_path)
            
            if not quiet:
                print('  saving to {parameters_path}'.format(parameters_path=parameters_path))

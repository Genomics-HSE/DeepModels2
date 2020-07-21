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
        self.loss = torch.nn.KLDivLoss(reduction='batchmean')
        self.optimizer = torch.optim.Adam(
            params=self.classifier.parameters(),
            lr=lr
        )
    
    def fit(self, dataset, n_epochs=32, progress=None):
        if progress is None:
            def progress(x):
                return x
                
        losses = np.ndarray(shape=(n_epochs, dataset.n_batches))
        
        for i in progress(range(n_epochs)):
            hidden_state = self.classifier.get_init_state(dataset.batch_size).to(self.device)
            data_iterator = dataset.get_fixlen_iter(train=True)
            for j, (X_batch, y_batch) in enumerate(data_iterator):
                X_batch = torch.from_numpy(X_batch).to(self.device)
                y_one_hot = one_hot_encoding(y_batch, self.classifier.n_output)
                y_one_hot = torch.from_numpy(y_one_hot).to(self.device)
                
                self.optimizer.zero_grad()
                logits, *hidden_state = self.classifier(X_batch, *hidden_state)
                print(logits.shape, y_one_hot.shape)
                loss = self.loss(logits, y_one_hot)
                loss.backward()
                self.optimizer.step()
                
                losses[i, j] = loss.item()
                print("Ep. {} Loss {}".format(i, loss.item()))
                # hidden_state = hidden_state.detach()
            
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
    
    def predict_proba(self, dataset, logger, mean=False):
        name = self.classifier.name
        with torch.no_grad():
            heatmap_predictions = []
            ground_truth = []
            data_iterator = dataset.get_fixlen_iter(train=False)
            hidden_state = self.classifier.get_init_state(1).to(self.device)
            for ix, (X, y) in tqdm(enumerate(data_iterator)):
                torch.cuda.empty_cache()
                X = torch.from_numpy(X).to(self.device)
                preds, *hidden_state = self.classifier(X, *hidden_state)
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
                    
                    logger.log_coalescent_heatmap(name, [preds.T, y.T], ix)
        
        return np.array(heatmap_predictions).T, np.array(ground_truth).T
    
    def save(self, parameters_path, quiet):
        if os.environ.get("FAST_RUN") is None or not os.path.exists(parameters_path):
            torch.save(self.classifier.state_dict(), parameters_path)
            
            if not quiet:
                print('  saving to {parameters_path}'.format(parameters_path=parameters_path))


def one_hot_encoding(y_data, num_class):
    """
    
    :param batch_data: (batch_size, seq_len)
    :return:
    """
    return (np.arange(num_class) == y_data[..., None]).astype(np.float32)

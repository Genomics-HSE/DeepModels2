from genomics_utils import LightningModuleExtended


class FullModelTraining(LightningModuleExtended):
    def training_step(self, batch, batch_ix, hiddens):
        X_batch, y_batch = batch
        logits, hiddens = self.forward(X_batch, hiddens)
        y_one_hot = one_hot_encoding(y_batch, self.n_output, device=self.device)
        loss = self.loss(logits, y_one_hot)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return {'loss': loss,
                'hiddens': hiddens
                }

    def test_step(self, batch, batch_idx) -> Any:
        X_batch, y_batch = batch
        logits = self.forward(X_batch)
        preds = torch.exp(logits)
        preds = torch.flatten(preds, start_dim=0, end_dim=1)
    
        # y_batch = torch.argmax(y_batch, dim=-1)
        y = y_batch.flatten()
    
        preds = preds.cpu().detach()
        self.logger.log_coalescent_heatmap(self.name, [preds.T, y.T], batch_idx)
    
    
    
    def tbptt_split_batch(self, batch: torch.Tensor, split_size: int) -> list:
        X_batch = batch[0]
        Y_batch = batch[1]
        batch_len = len(X_batch)
        
        distances_dims = [len(X) for X in batch[0]]
        max_time_dim = np.max(distances_dims)
        
        splits = []
        x_split_size = split_size
        y_t_indexes = [0 for _ in range(batch_len)]
        
        for x_t in range(0, max_time_dim, x_split_size):
            batch_split = []
            
            split_x = [[] for _ in range(batch_len)]
            split_y = [[] for _ in range(batch_len)]
            
            for batch_idx in range(batch_len):
                if x_t > len(X_batch[batch_idx]):
                    continue
                elif x_t + x_split_size > len(X_batch[batch_idx]):
                    current_x_sample = X_batch[batch_idx][x_t: len(X_batch[batch_idx])]
                else:
                    current_x_sample = X_batch[batch_idx][x_t:x_t + x_split_size]
                
                y_t = y_t_indexes[batch_idx]
                y_split_size = int(np.sum(current_x_sample))
                split_x[batch_idx] = current_x_sample
                split_y[batch_idx] = Y_batch[batch_idx][y_t:y_t + y_split_size]
                
                y_t_indexes[batch_idx] = y_t + y_split_size
            
            batch_split.append(split_x)
            batch_split.append(split_y)
            
            splits.append(batch_split)
        
        return splits

def one_hot_encoding(y_data, num_class, device):
    batch_size, seq_len = y_data.shape
    y_one_hot = torch.FloatTensor(batch_size, seq_len, num_class).to(device)
    
    y_one_hot.zero_()
    y_one_hot.scatter_(2, y_data.unsqueeze(2), 1)
    return y_one_hot
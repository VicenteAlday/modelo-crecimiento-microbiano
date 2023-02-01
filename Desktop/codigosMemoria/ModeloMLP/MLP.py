
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
import torch
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from tqdm.notebook import tqdm


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

if __name__ == '__main__':

    class RMSLELoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.mse = nn.MSELoss()
            
        def forward(self, pred, actual):
            return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))
        
    def RMSLE(y_true, y_pred):
        return np.sqrt(mean_squared_log_error(y_true, y_pred))

    class MLP(pl.LightningModule):
    
        def __init__(self, X, y, X_test, learning_rate, y_scaler, seed):
            super().__init__()
            self.save_hyperparameters()
            
            self.layers = nn.Sequential(
                nn.Linear(3, 32),
                nn.ReLU(),
                nn.Linear(32, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 3),
                nn.ReLU())

            
            self.X = X
            self.y = y
            self.X_test = X_test
            self.learning_rate = learning_rate
            self.seed = seed
            self.y_scaler = y_scaler
            self.loss = RMSLELoss()


        def forward(self, x):
            return self.layers(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.layers(x)
            loss = self.loss(y_hat, y)
            self.log('train_loss', loss)
            return loss
        
        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.layers(x)
            y_true = self.y_scaler.inverse_transform(y.cpu().numpy())
            y_pred = self.y_scaler.inverse_transform(y_hat.cpu().numpy())
            loss = RMSLE(y_true, y_pred)
            return loss
        
        def validation_epoch_end(self, val_step_outputs):
            loss = sum(val_step_outputs) / len(val_step_outputs)
            self.log('val_loss', loss)

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.75, patience=6, verbose = 1,mode = 'min', cooldown = 0, min_lr = 10e-7)
            optimizer_dict = {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
            return optimizer_dict
        
        def setup(self, stage):
            X = self.X
            y = self.y
            X_test = self.X_test
            
            X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.85, random_state=self.seed)
            
            self.X_train_scaled = X_train
            self.X_val_scaled = X_val
            self.X_test_scaled = X_test

            self.y_train_scaled = y_train
            self.y_val_scaled = y_val

        def train_dataloader(self):
            dataset = TensorDataset(torch.FloatTensor(self.X_train_scaled), torch.FloatTensor(self.y_train_scaled))
            train_loader = DataLoader(dataset, batch_size=256, num_workers=8, shuffle=True)
            return train_loader
        
        def val_dataloader(self):
            val_dataset = TensorDataset(torch.FloatTensor(self.X_val_scaled), torch.FloatTensor(self.y_val_scaled))
            val_loader = DataLoader(val_dataset, batch_size=256, num_workers=8, shuffle=False)
            return val_loader
        
        def test_dataloader(self):
            test_dataset = TensorDataset(torch.FloatTensor(self.X_test_scaled))
            test_dataloader = DataLoader(test_dataset, batch_size=512, num_workers=8, shuffle=False)
            return test_dataloader

            #learningratefinder


    def run():
        torch.multiprocessing.freeze_support()
        print('loop')

    if __name__ == '__main__':
        run()

    train = pd.read_csv(r"C:\Users\vicen\OneDrive\Escritorio\Memoria\modeloAI\datasets\1_train.csv", sep=';', delimiter=None, header='infer')
    test = pd.read_csv(r"C:\Users\vicen\OneDrive\Escritorio\Memoria\modeloAI\datasets\2_test.csv", sep=';', delimiter=None, header='infer')

    
    X = train[['concBioinicial','concEtincial', 'concSusinicial']]
    y = train[['concBiofinal', 'concEtfinal', 'concSusfinal']]
    X_test = test[['concBioinicial','concEtincial', 'concSusinicial']]

    X_scaler = StandardScaler()
    y_scaler = MinMaxScaler()

    X = X_scaler.fit_transform(X)
    y = y_scaler.fit_transform(y)
    X_test = X_scaler.transform(X_test)

    N_FOLDS = 3

    scores=list()
    preds = list()
    for fold in tqdm(range(N_FOLDS)):

        early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=20,
        verbose=True,
        mode='min',
        )
        ckpt_callback = ModelCheckpoint(mode="min", 
                                        monitor="val_loss", 
                                        dirpath=r"C:\Users\vicen\OneDrive\Escritorio\Memoria\modeloAI\datasets", filename=f'fold_{N_FOLDS}_{fold}')

        
        model = MLP(X, y, X_test, 1e-3, y_scaler=y_scaler, seed=42 + fold)
        trainer = pl.Trainer(auto_lr_find=True)
        trainer.tune(model)
        print('Learning rate:', model.learning_rate)
        trainer = pl.Trainer(callbacks=[early_stop_callback, ckpt_callback])
        trainer.fit(model)
        test_loader = model.test_dataloader()
        
        print(f'FOLD #{fold}| best rmsle: {ckpt_callback.best_model_score.item():.5g}')
        
        model = model.load_from_checkpoint(str(list(Path(r"C:\Users\vicen\OneDrive\Escritorio\Memoria\modeloAI\datasets").glob(f'fold_{N_FOLDS}_{fold}*ckpt'))[0]))
        model.eval()
        y_test = list()
        for x, in test_loader:
            y_test.append(model.forward(x.to(model.device)).detach().cpu().numpy())
        y_test = y_scaler.inverse_transform(np.concatenate(y_test))
        
        preds.append(y_test)
        scores.append(ckpt_callback.best_model_score.item())

        np.mean(scores)

        for i, pred in enumerate(preds):
            if i == 0:
                y_test = pred
            else:
                y_test = y_test + pred
        y_test = y_test / len(preds)

        submission = pd.read_csv(r"C:\Users\vicen\OneDrive\Escritorio\Memoria\modeloAI\datasets\sample_submission.csv")
        submission['concBiofinal']=y_test[:,0]
        submission['concEtfinal']=y_test[:,1]
        submission['concSusfinal']=y_test[:,2]
        submission.to_csv('submission.csv', index=False)


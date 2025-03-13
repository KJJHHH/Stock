import sys 
sys.path.append('../')
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from accelerate import Accelerator

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from accelerate import Accelerator
from abc import abstractmethod
from numpy import inf
# from logger import TensorboardWriter

from transformer_based.datas import *
from base.trainer import BaseTrainer
from transformer_based.backtestor import Backtestor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""""
Future update:
- Add different model to fit this predictor
- Use differnet src data to train transformer
- Train from stored model
"""

class TransformerTrainer(BaseTrainer):
    def __init__(self, 
        stock_list, 
        data,
        model, 
        config,  
        dirs,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        ) -> None:
        super().__init__(stock_list, data, model, config, dirs, device)
        
        # Data  
        """datas
        - srcs_trained: list of stocks as src are trained
        """
        self.data_src = data.src.to(device)
        self.srcs_trained = [] 
        self.memory = None
        
    # Transformer function
    def _model_train(self):
        loss_train_mean = 0
        self.model.train()
        for x, y in tqdm(self.data.trainloader): 
            
            self.optimizer.zero_grad()       
            self.memory, outputs = self.model(src=self.data_src, tgt=x)    
            loss = self.criterion(outputs, y)
            self.accelerator.backward(loss)
            self.optimizer.step()
            loss_train_mean += loss.item()
            
        return loss_train_mean / len(self.data.trainloader)
    
    def _model_validate(self):
        loss_valid_mean = 0
        with torch.no_grad():
            self.model.eval()
            for x_val, y_val in self.data.validloader:
                _, outputs_val = self.model(memory=self.memory, tgt=x_val)
                loss = self.criterion(outputs_val, y_val)
                loss_valid_mean += loss.item()
        
        return loss_valid_mean / len(self.data.validloader)
    
    def _val_with_asset(self, epoch):
        # Validate with return
        val_return, hist = Backtestor.testModel(self.model, self.data.validloader, self.data.src, short=True)

        if val_return > self.best_val_result:
            print(f'New best model found with return {val_return}')
            self.not_improve_cnt = 0
            self.best_val_result = val_return
            self._save_checkpoint(epoch, save_best=True)
        else:
            self.not_improve_cnt += 1
            if self.not_improve_cnt >= 100:
                print(f'Early stopping at epoch {epoch}')
                return True
        
        return False
    
    def _val_with_loss(self, epoch):
        # Validate with loss
        loss_valid_mean = self._model_validate()

        if loss_valid_mean < self.best_val_result:
            print(f'New best model found in epoch {epoch} with val loss: {loss_valid_mean}')
            self.not_improve_cnt = 0
            self.best_val_result = loss_valid_mean
            self._save_checkpoint(epoch, save_best=True)
        else:
            self.not_improve_cnt += 1
            if self.not_improve_cnt >= 100:
                print(f'Early stopping at epoch {epoch}')
                return True
        
        return False
    
    def _update_src(self, src):
        """
        self.data_src = src.to(self.device)
        self.srcs_trained.append(self.stock)
        """

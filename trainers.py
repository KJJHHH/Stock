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

from datas import *
from base.base_trainer import BaseTrainer
from backtestors import *
from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Issues
- udpate data fail
- save checkpoint failed for last stock
"""
class TransformerTrainer(BaseTrainer):
    
    def __init__(self, 
        stock_list, 
        config,  
        dirs,
        ) -> None:       
        
        # Config
        self.config = config
        
        # Data for transformer
        self.memory = ...
        self.src_len = ...  
        
        # Init base trainer
        super().__init__(stock_list, config, dirs)
    
    # Model
    def _init_model(self):
        return Transformer(
            d_model=self.config["d_model"],
            dropout=self.config["dropout"],
            d_hid=self.config["d_hid"],
            nhead=self.config["nhead"],
            nlayers_e=self.config["nlayers_e"],
            nlayers_d=self.config["nlayers_d"],
            ntoken=self.config["ntoken"],
            ).to(device) 
    
    # Data
    def _data_obj(self, stock):
        return TransformerData(
            stock=stock,
            config=self.config,
            )
    
    def _init_data(self):
        data = self._data_obj(self.stock_target)
        self.src_len = data.src.shape[1]  
        return data
        
    def _update_data(self):
        # data object
        self.data = self._data_obj(self.stock)
        
        # padding data src
        if self.data.src.shape[1] > self.src_len:
            self.data.src = self.data.src[:, :self.src_len, :]
        if self.data.src.shape[1] < self.src_len:
            self.data.src = torch.cat(
                (
                    self.data.src, 
                    torch.zeros(
                        (self.data.src.shape[0], self.src_len - self.data.src.shape[1], self.data.src.shape[2]),
                        device=self.data.src.device)
                ), dim=1)
    
    # Train detailed function
    def _model_train(self):
        loss_train_mean = 0
        self.model.train()
        for x, y in tqdm(self.data.trainloader): 
            
            self.optimizer.zero_grad()       
            outputs = self.model(src=self.data.src, tgt=x)    
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
                outputs_val = self.model(src=self.data.src, tgt=x_val)
                loss = self.criterion(outputs_val, y_val)
                loss_valid_mean += loss.item()
        
        return loss_valid_mean / len(self.data.validloader)
    
    def _model_backtest(self):
        asset_hist = TransformerBacktestor.backtesting(
            TransformerBacktestor._test_method, 
            self.model, 
            (self.data.validloader, self.data.src), 
            short=True, 
            verbose=False)
        return asset_hist[-1]
    
    
    # ...
    def _update_(self, src):
        self.data.src = ...
        """
        self.data_src = src.to(self.device)
        self.srcs_trained.append(self.stock)
        """


class ResnetTrainer(BaseTrainer):
    def __init__(self, 
        stock_list, 
        data,
        model, 
        config,  
        dirs,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        ) -> None:
        super().__init__(stock_list, data, model, config, dirs, device)
        
    """Resnet input (Conv2d): https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    - input: (N, C_in, H, W)
    - output: (N, C_out, H_out, W_out)
    """
    
    # train function
    def _model_train(self):
        
        loss_train_mean = 0
        self.model.train()
        for x, y in tqdm(self.data.trainloader): 
            
            self.optimizer.zero_grad()       
            outputs = self.model(x)    
            loss = self.criterion(outputs, y)
            self.accelerator.backward(loss)
            self.optimizer.step()
            loss_train_mean += loss.item()
            
        return loss_train_mean / len(self.data.trainloader)
    
    # Let this function to backtestor: use in backtestor and trainer or testmodel to trainer
    def _model_validate(self):
        loss_valid_mean = 0
        with torch.no_grad():
            self.model.eval()
            for x_val, y_val in self.data.validloader:
                outputs_val = self.model(x_val)
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
    
    def _update_data(self, src):
        """
        self.data_src = src.to(self.device)
        self.srcs_trained.append(self.stock)
        """

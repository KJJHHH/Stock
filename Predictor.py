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

from transformer_based.datas import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""""
Future update:
- Use differnet src data to train transformer
- Add different model to fit this predictor
- Train from stored model
"""

class Seq2SeqPredictor:
    def __init__(self, stock, model, model_dir="./", num_epochs=200, lr=0.001):
        
        self.stock = stock
        self.num_epochs = num_epochs
        self.lr = lr
        self.model_dir = model_dir
        
        # Data Preparation
        self.data = TransformerData(stock)
        self.data.prepareData()
        
        # Accelerator setup
        self.accelerator = Accelerator(mixed_precision='fp16')
        self.device = self.accelerator.device
        
        # Model, loss, optimizer, scheduler
        self.model = model().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.00001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=len(self.data.trainloader) * 1, gamma=0.9)
        self.memory = None
        
        # Prepare with accelerator
        (
            self.model,
            self.optimizer,
            self.data.trainloader,
            self.data.validloader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model, self.optimizer, self.data.trainloader, self.data.validloader, self.scheduler
        )
        
    def checkDir(self):
        file_lists = [self.model_dir+f"{self.model.__name__}-result", self.model_dir+f"{self.model.__name__}-temp"]
        for file in file_lists:
            if not os.path.exists(file):
                os.makedirs(file)
                print(f"Created directory: {file}")
            else:
                print(f"Directory already exists: {file}")
    
    def train(self):
        
        self.checkDir()
        
        print(f"Start training stock {self.stock} model")
        
        last_epoch = 0
        not_improve_cnt = 0
        min_val_loss = float("inf")
        self.data.src = self.data.src.to(self.device)
        
        
        for epoch in range(last_epoch, self.num_epochs):
            # Train
            loss_train_mean = self.transformer_train()
            
            # Scheduler 
            if epoch > 50:
                self.scheduler.step()
                
            # Store ckpt
            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), self.model_dir+f'{self.model.__name__}-temp/{self.stock}_epoch_{epoch}.pt')
            
            # Validating
            loss_valid_mean = self.transformer_validate()
            if loss_valid_mean < min_val_loss:
                min_val_loss = loss_valid_mean
                not_improve_cnt = 0
                print(f'New best model found in epoch {epoch} with val loss: {min_val_loss}')
                torch.save(self.model.state_dict(), self.model_dir+f'{self.model.__name__}-result/{self.stock}__best.pt')
            else:
                not_improve_cnt += 1
                if not_improve_cnt >= 50:
                    print(f'Early stopping at epoch {epoch}')
                    break
                    
            print(f"Epoch {epoch} | training loss: {loss_train_mean:.3f} evaluating loss: {loss_valid_mean:.3f}")

    # Transformer function
    def transformer_train(self):
        loss_train_mean = 0
        self.model.train()
        for x, y in tqdm(self.data.trainloader): 
            
            self.optimizer.zero_grad()       
            self.memory, outputs = self.model(src=self.data.src, tgt=x, train=True)    
            loss = self.criterion(outputs, y)
            self.accelerator.backward(loss)
            self.optimizer.step()
            loss_train_mean += loss.item()
            
        return loss_train_mean / len(self.data.trainloader)
    
    def transformer_validate(self):
        loss_valid_mean = 0
        with torch.no_grad():
            self.model.eval()
            for x_val, y_val in self.data.validloader:
                _, outputs_val = self.model(tgt=x_val, train=False, memory=self.memory)
                loss = self.criterion(outputs_val, y_val)
                loss_valid_mean += loss.item()
        
        return loss_valid_mean / len(self.data.validloader)
    
    # more model function...


if __name__ == "__main__":
    # Usage example
    from transformer_based.models import Transformer
    stock = "2884.TW"
    predictor = Seq2SeqPredictor(stock, Transformer)
    predictor.train()

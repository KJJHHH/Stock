import sys 
sys.path.append('../')
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import *
from datas import *
from accelerate import Accelerator

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from accelerate import Accelerator

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
        self.data = Data(stock)
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
        file_lists = [self.model_dir+"result", self.model_dir+"temp"]
        for file in file_lists:
            if not os.path.exists(file):
                os.makedirs(file)
                print(f"Created directory: {file}")
            else:
                print(f"Directory already exists: {file}")
    
    def train(self):
        
        self.checkDir()
        
        last_epoch = 0
        not_improve_cnt = 0
        min_val_loss = float("inf")
        self.data.src = self.data.src.to(self.device)
        
        for epoch in range(last_epoch, self.num_epochs):
            self.model.train()
            loss_train_mean = 0
            
            # A function for this (for different model)
            # =======
            for x, y in tqdm(self.data.trainloader): 
                x = x.permute(0, 2, 1)    
                
                self.optimizer.zero_grad()       
                memory, outputs = self.model(src=self.data.src, tgt=x, train=True)    
                loss = self.criterion(outputs, y)
                self.accelerator.backward(loss)
                self.optimizer.step()
                loss_train_mean += loss.item()
            # =======
            
            # Train loss
            loss_train_mean /= len(self.data.trainloader)
            
            # Scheduler 
            if epoch > 50:
                self.scheduler.step()
                
            # Store ckpt
            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), self.model_dir+f'temp/{self.stock}_epoch_{epoch}.pt')
            
            # Validating
            loss_valid_mean = self.validate(memory)
            if loss_valid_mean < min_val_loss:
                min_val_loss = loss_valid_mean
                not_improve_cnt = 0
                print(f'New best model found in epoch {epoch} with val loss: {min_val_loss}')
                torch.save(self.model.state_dict(), self.model_dir+f'result/{self.stock}_best.pt')
            else:
                not_improve_cnt += 1
                if not_improve_cnt >= 50:
                    print(f'Early stopping at epoch {epoch}')
                    break
                    
            print(f"Epoch {epoch} | training loss: {loss_train_mean:.3f} evaluating loss: {loss_valid_mean:.3f}")

    
    def validate(self, memory):
        loss_valid_mean = 0
        with torch.no_grad():
            self.model.eval()
            for x_val, y_val in self.data.validloader:
                x_val = x_val.permute(0, 2, 1)
                _, outputs_val = self.model(tgt=x_val, train=False, memory=memory)
                loss = self.criterion(outputs_val, y_val)
                loss_valid_mean += loss.item()
            
        loss_valid_mean /= len(self.data.validloader)
        
        return loss_valid_mean

if __name__ == "__main__":
    # Usage
    from Transformers import Transformer
    stock = "2884.TW"
    predictor = Seq2SeqPredictor(stock, Transformer)
    predictor.train()

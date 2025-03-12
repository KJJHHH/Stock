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
from Backtestor import Backtestor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""""
Future update:
- Add different model to fit this predictor
- Use differnet src data to train transformer
- Train from stored model
"""


class Seq2SeqPredictor:
    def __init__(self, 
        stock_list, 
        model, 
        config,  
        dirs,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        ) -> None:
        
        self.stock_list = stock_list
        self.ckpt_dir = dirs[0]
        self.performance_dir = dirs[1]
        self.device = device
        self.checkDir()
        
        # target stock
        self.stock = stock[0]     
        
        # Config
        self.config = config
        self.start_date = config["start_date"]
        self.end_date = config["end_date"]
        self.epochs = config["epochs"]
        self.val_type = config["val_type"] # by loss or by asset
        
        # Logger
        # self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        
        # Init training state
        self.not_improve_cnt = 0
        if self.val_type == "loss":
            self.best_val_result = float("inf")
        else:
            self.best_val_result = float("-inf")
        
        # Data  
        """datas
        - srcs_trained: list of stocks as src are trained
        """
        self.data = self.getSrc(self.stock)
        self.data_src = None
        self.srcs_trained = []
        
        # Accelerator setup
        self.accelerator = Accelerator(mixed_precision='fp16')
        self.device = self.accelerator.device
        
        # Model, loss, optimizer, scheduler
        self.model = model
        self.model_best = self.model
        self.criterion = nn.MSELoss()
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        self.scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, self.optimizer)
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
        file_lists = [self.ckpt_dir, self.performance_dir]
        for file in file_lists:
            if not os.path.exists(file):
                os.makedirs(file)
                print(f"Created directory: {file}")
            else:
                print(f"Directory already exists: {file}")
    
    def getSrc(self, stock):     
        return TransformerData(
            stock=stock, 
            start_date=self.start_date,
            end_date=self.end_date,
            window=5, 
            batch_size=128, 
            percentage_test=0.05, 
            percentage_valid=0.1, 
            src_len=0
            ).src.to(device)
    
    def train(self):
        
        print(f"Start training stock {self.stock} model")
        
        last_epoch = 0
        
        for epoch in range(last_epoch, self.epochs):
            # Train
            loss_train_mean = self.transformer_train()
            
            # Scheduler 
            if epoch > 100:
                self.scheduler.step()
                
            # Save ckpt
            if epoch % 10 == 0:
                self._save_checkpoint(epoch)
            
            # Validating with return
            if self.val_type == "asset":
                stop = self.val_asset(epoch)
                if stop:
                    break
            else:
                stop = self.val_loss(epoch)
                if stop:
                    break
                    
            print(f"Epoch {epoch} | training loss: {loss_train_mean:.3f}")

    def trainMultipleSrc(self):
        """Train with src from different stock
        """
        resume = False
        if resume:
            self._resume_checkpoint()
        for s in self.stock_list:
            if s in self.srcs_trained:
                continue
            self.data_src = self.getSrc(s)
            self.train()
            self.srcs_trained.append(s)
    
    
    def val_asset(self, epoch):
        val_return, hist = Backtestor.testModel(self.model, self.data.validloader, self.data.src, short=True)

        if val_return > self.best_val_result:
            print(f'New best model found with return {val_return}')
            self.not_improve_cnt = 0
            self.best_val_result = val_return
            self.model_best = self.model
        else:
            self.not_improve_cnt += 1
            if self.not_improve_cnt >= 100:
                print(f'Early stopping at epoch {epoch}')
                return True
        
        return False
    
    def val_loss(self, epoch):
        loss_valid_mean = self.transformer_validate()

        if loss_valid_mean < self.best_val_result:
            print(f'New best model found in epoch {epoch} with val loss: {self.best_val_result}')
            self.not_improve_cnt = 0
            self.best_val_result = loss_valid_mean
            self.model_best = self.model
        else:
            self.not_improve_cnt += 1
            if self.not_improve_cnt >= 100:
                print(f'Early stopping at epoch {epoch}')
                return True
        
        return False
    
    # Transformer function
    def transformer_train(self):
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
    
    def transformer_validate(self):
        
        # val with loss
        loss_valid_mean = 0
        with torch.no_grad():
            self.model.eval()
            for x_val, y_val in self.data.validloader:
                _, outputs_val = self.model(memory=self.memory, tgt=x_val)
                loss = self.criterion(outputs_val, y_val)
                loss_valid_mean += loss.item()
        
        return loss_valid_mean / len(self.data.validloader)
    
    # Save: 'epoch-{}-{}.pth'.format(epoch, self.stock)
    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'not_improve_cnt': self.not_improve_cnt,
            'best_val_result': self.best_val_result,
            'srcs_trained': self.srcs_trained,
        }
        filename = str(self.ckpt_dir / 'epoch-{}-{}.pth'.format(epoch, self.stock))
        torch.save(state, filename)
        if save_best:
            best_path = str(self.ckpt_dir / f'{self.stock}_best.pth')
            torch.save(state, best_path)

    def _resume_checkpoint(self, load_best=False):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        
        def resumePath(load_best):
            if load_best:
                return str(self.ckpt_dir / f'{self.stock}_best.pth')
            
            ckpts = [f for f in os.listdir(self.ckpt_dir) if os.path.isfile(os.path.join(self.ckpt_dir, f))]
            ckpt_epochs = [int(file.split(".")[0].split("-")[2]) for file in ckpts if self.stock in file and "checkpoint-epoch" in file]
            return str(self.ckpt_dir / f'epoch-{max[ckpt_epochs]}-{self.stock}.pth')
        
        resume_path = resumePath(load_best)
        checkpoint = torch.load(resume_path)

        # load architecture params from checkpoint.
        assert checkpoint['config']['arch'] == self.config['arch'] , \
            "Architecture configuration given in config file is different from that of checkpoint. " \
            "This may yield an exception while state_dict is being loaded."
        self.model.load_state_dict(checkpoint['state_dict'])

        # load best used in backtesting, no need for load others
        if load_best:
            return None
        
        # load optimizer state from checkpoint only when optimizer type is not changed.
        assert checkpoint['config']['optimizer']['type'] == self.config['optimizer']['type'], \
            "Optimizer type given in config file is different from that of checkpoint. " \
            "Optimizer parameters from checkpoint are not being resumed."
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        # laod training state
        self.start_epoch = checkpoint['epoch'] + 1
        self.not_improve_cnt = checkpoint['not_improve_cnt']
        self.best_val_result = checkpoint['best_val_result']  
        self.srcs_trained = checkpoint['srcs_trained']
        

if __name__ == "__main__":
    # Usage example
    from transformer_based.models import Transformer
    stock = "2884.TW"
    predictor = Seq2SeqPredictor(stock, Transformer)
    predictor.train()

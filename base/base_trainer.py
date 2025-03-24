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

"""
- Delete val_with_asset, loss function, move to _train
"""

# Save: 'epoch-{}-{}.pt'.format(epoch, self.stock)
class BaseTrainer:
    def __init__(self, 
        stock_list, 
        config,  
        dirs,
        short = True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        ) -> None:
        
        # target stock
        self.stock = None
        self.stock_target = stock_list[0]
        self.stock_list = stock_list
        self.stock_trained = []
        self.file_prefix = "-".join(stock_list)
        self.ckpt_dir = dirs["ckpt_dir"]
        self.performance_dir = dirs["performance_dir"]
        self.device = device
        self._check_dir()
        
        # Config
        self.config = config
        self.epochs = config["epochs"]
        self.val_type = config["val_type"] # by loss or by asset
        self.lr = config["optimizer"]["args"]["lr"]
        self.weight_decay = config["optimizer"]["args"]["weight_decay"]
        self.amsgrad = config["optimizer"]["args"]["amsgrad"]
        self.scheduler_step = config["lr_scheduler"]["args"]["step_size"]
        self.scheduler_gamma = config["lr_scheduler"]["args"]["gamma"]
        
        # Logger
        # self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        
        # Init training control
        self.start_epoch = ...
        self.not_improve_cnt = ...
        self.best_val_result = ...
        self._init_training_control()
        
        # Data  
        self.data = self._init_data()  
        
        # Accelerator setup
        self.accelerator = Accelerator(mixed_precision='fp16')
        self.device = self.accelerator.device
        
        # Model, loss, optimizer, scheduler
        self.model = self._init_model()
        self.model_best = self.model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad
        )
        self.scheduler = optim.lr_scheduler.StepLR( 
            self.optimizer, 
            step_size=self.scheduler_step, 
            gamma=self.scheduler_gamma
        )        
    
        self.short = short
    
    # Model
    @abstractmethod
    def _init_model(self):
        raise NotImplementedError
    
    # Data
    @abstractmethod
    def _data_obj(self):
        raise NotImplementedError  
    @abstractmethod
    def _init_data(self):
        raise NotImplementedError    
    @abstractmethod
    def _update_data(self):
        raise NotImplementedError
    
    # Train
    def train(self):
        print(f"Validatin method: {self.val_type}")
        
        self._resume_checkpoint()
        
        for stock in self.stock_list:
            
            if stock in self.stock_trained:
                print(f"Stock {stock} already trained")
                continue
            
            self.stock = stock
            self._init_training_control()
            self._update_data()
            self._accelerate()
            self._train_stock()
    
    def _train_stock(self):
        
        print(f"Start training stock {self.stock}")
        
        for epoch in range(self.start_epoch, self.epochs):
            # Train
            loss_train_mean = self._model_train()
            
            # Scheduler 
            if epoch > 100:
                self.scheduler.step()
                
            # Save ckpt
            if epoch % 10 == 0:
                self._save_checkpoint(epoch)
            if epoch == self.epochs - 1:
                self.stock_trained.append(self.stock)
                self._save_checkpoint(epoch)
            
            # Validating with return
            stop = self._validate(epoch)
            if stop:
                break
                    
            torch.cuda.empty_cache()
            print(f"Epoch {epoch} | training loss: {loss_train_mean:.3f}")
        
    def _validate(self, epoch):
        if self.val_type == "asset":
            val_return, val_hold_return = self._model_backtest()

            if val_return > self.best_val_result:
                print(f'New best model found with return {val_return} | hold return {val_hold_return}')
                self.not_improve_cnt = 0
                self.best_val_result = val_return
                self._save_checkpoint(epoch, save_best=True)
            else:
                self.not_improve_cnt += 1
                if self.not_improve_cnt >= 100:
                    print(f'Early stopping at epoch {epoch}')
                    return True
            return False
        
        else:
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
        
    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints at: 'epoch-{}-{}.pth'.format(epoch, self.stock)

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'stock_trained': self.stock_trained,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(), # learning rate will change
            'config': self.config,
            'not_improve_cnt': self.not_improve_cnt,
            'best_val_result': self.best_val_result,
        }
        if save_best:
            file = f"{self.file_prefix}-best.pth"
            best_path = str(self.ckpt_dir + file)
            torch.save(state, best_path)
        else:
            file = '{}-{}.pth'.format(self.file_prefix, epoch) 
            filename = str(self.ckpt_dir + file)
            torch.save(state, filename)

    def _resume_checkpoint(self):
        """
        Resume from saved checkpoints at: 'epoch-{}-{}.pth'.format(epoch, self.stock)

        :param resume_path: Checkpoint path to be resumed
        """
        
        # Get resume path
        ckpts = [f for f in os.listdir(self.ckpt_dir) if os.path.isfile(os.path.join(self.ckpt_dir, f)) and self.file_prefix in f and "best" not in f]
        if ckpts == []:
            return None  
        ckpt_epochs = [int(file.split("-")[-1].split(".")[0]) for file in ckpts]
        file = '{}-{}.pth'.format(self.file_prefix, max(ckpt_epochs)) 
        resume_path = str(self.ckpt_dir + file)
        
        # Load checkpoint
        print("Loading checkpoint: {}".format(resume_path))
        checkpoint = torch.load(resume_path, weights_only=False)

        # load architecture params from checkpoint.
        assert checkpoint['arch'] == self.config['name'] , \
            "Architecture configuration given in config file is different from that of checkpoint. " \
            "This may yield an exception while state_dict is being loaded."
        try:
            self.model.load_state_dict(checkpoint['state_dict'])
        except:
            print("Model state_dict not loaded")
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
        self.stock_trained = checkpoint['stock_trained']
    
    # Train detailed function
    @abstractmethod
    def _model_train(self):
        raise NotImplementedError
    @abstractmethod
    def _model_validate(self):
        raise NotImplementedError
    @abstractmethod
    def _model_backtest(self):
        raise NotImplementedError
    
    # Utils
    def _check_dir(self):
        file_lists = [self.ckpt_dir, self.performance_dir]
        for file in file_lists:
            if not os.path.exists(file):
                os.makedirs(file)
                print(f"Created directory: {file}")
            else:
                print(f"Directory already exists: {file}")
    
    def _init_training_control(self):
        self.start_epoch = 0
        self.not_improve_cnt = 0
        if self.val_type == "loss":
            self.best_val_result = float("inf")
        else:
            self.best_val_result = float("-inf")
    
    def _accelerate(self):    
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
    
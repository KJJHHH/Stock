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
        self.ckpt_dir = dirs["ckpt_dir"]
        self.performance_dir = dirs["performance_dir"]
        self.device = device
        self._check_dir()
        
        # Config
        self.config = config
        self.start_date = config["start_date"]
        self.end_date = config["end_date"]
        self.epochs = config["epochs"]
        self.val_type = config["val_type"] # by loss or by asset
        self.lr = config["optimizer"]["args"]["lr"]
        self.weight_decay = config["optimizer"]["args"]["weight_decay"]
        self.amsgrad = config["optimizer"]["args"]["amsgrad"]
        self.scheduler_step = config["lr_scheduler"]["args"]["step_size"]
        self.scheduler_gamma = config["lr_scheduler"]["args"]["gamma"]
        
        # Logger
        # self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        
        # Init training state
        self.start_epoch = 0
        self.not_improve_cnt = 0
        if self.val_type == "loss":
            self.best_val_result = float("inf")
        else:
            self.best_val_result = float("-inf")
        
        # Data  
        self.data = None
        
        # Accelerator setup
        self.accelerator = Accelerator(mixed_precision='fp16')
        self.device = self.accelerator.device
        
        # Model, loss, optimizer, scheduler
        self.model = self._build_model()
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
        
    @abstractmethod
    def _update_data(self):
        raise NotImplementedError
    
    @abstractmethod
    def _build_model(self):
        raise NotImplementedError
    
    def _train(self):
        
        print(f"Start training stock {self.stock} model")
        print(f"Validatin method: {self.val_type}")
        
        self._resume_checkpoint()
        
        for epoch in range(self.start_epoch, self.epochs):
            # Train
            loss_train_mean = self._model_train()
            
            # Scheduler 
            if epoch > 100:
                self.scheduler.step()
                
            # Save ckpt
            if epoch % 10 == 0:
                self._save_checkpoint(epoch)
            
            # Validating with return
            if self.val_type == "asset":
                stop = self._val_with_asset(epoch)
                if stop:
                    break
            else:
                stop = self._val_with_loss(epoch)
                if stop:
                    break
                    
            print(f"Epoch {epoch} | training loss: {loss_train_mean:.3f}")
    
    def train(self):
        for stock in self.stock_list:
            self.stock = stock
            self._update_data()
            self._accelerate()
            self._train()
    
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
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(), # learning rate will change
            'config': self.config,
            'not_improve_cnt': self.not_improve_cnt,
            'best_val_result': self.best_val_result,
        }
        if save_best:
            best_path = str(self.ckpt_dir + f'{self.stock}-best.pth')
            torch.save(state, best_path)
            return None
        filename = str(self.ckpt_dir + '{}-{}.pth'.format(self.stock, epoch))
        torch.save(state, filename)

    def _resume_checkpoint(self):
        """
        Resume from saved checkpoints at: 'epoch-{}-{}.pth'.format(epoch, self.stock)

        :param resume_path: Checkpoint path to be resumed
        """
        
        # Get resume path
        ckpts = [f for f in os.listdir(self.ckpt_dir) if os.path.isfile(os.path.join(self.ckpt_dir, f)) and self.stock in f and "best" not in f]
        if ckpts == []:
            return None  
        ckpt_epochs = [int(file.split("-")[1].split(".")[0]) for file in ckpts]
        resume_path = str(self.ckpt_dir + f'{self.stock}-{max(ckpt_epochs)}.pth')
        
        # Load checkpoint
        print("Loading checkpoint: {}".format(resume_path))
        checkpoint = torch.load(resume_path)

        # load architecture params from checkpoint.
        assert checkpoint['arch'] == self.config['name'] , \
            "Architecture configuration given in config file is different from that of checkpoint. " \
            "This may yield an exception while state_dict is being loaded."
        self.model.load_state_dict(checkpoint['state_dict'])

        
        # load optimizer state from checkpoint only when optimizer type is not changed.
        assert checkpoint['config']['optimizer']['type'] == self.config['optimizer']['type'], \
            "Optimizer type given in config file is different from that of checkpoint. " \
            "Optimizer parameters from checkpoint are not being resumed."
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        # laod training state
        self.start_epoch = checkpoint['epoch'] + 1
        self.not_improve_cnt = checkpoint['not_improve_cnt']
        self.best_val_result = checkpoint['best_val_result']  
        
    def _val_with_asset(self, epoch):
        # Validate with return
        val_return = self._model_backtest()

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
    
    def _check_dir(self):
        file_lists = [self.ckpt_dir, self.performance_dir]
        for file in file_lists:
            if not os.path.exists(file):
                os.makedirs(file)
                print(f"Created directory: {file}")
            else:
                print(f"Directory already exists: {file}")
    

    @abstractmethod
    def _model_train(self):
        raise NotImplementedError
    @abstractmethod
    def _model_validate(self):
        raise NotImplementedError
    @abstractmethod
    def _model_backtest(self):
        raise NotImplementedError
from __future__ import annotations
import matplotlib.pyplot as plt
import torch
import pandas as pd
from abc import abstractmethod
import os


class Backtestor():
    def __init__(self, stock_list, config, dirs):
        
        self.stock = stock_list
        self.stock_target = stock_list[0]
        self.config = config
        self.ckpt_dir = dirs["ckpt_dir"]
        self.performance_dir = dirs["performance_dir"]
        self.file_prefix = dirs["file_prefix"]  
        
        self.model = self._init_model()
        
        self.data = self._init_data()       
        
        self.loader = self.data.testloader
        self.len = self.__get_datalen()
        self.dates = self.__get_time()
        
        print("Backtesting " + self.model.__name__ + " on " + self.stock_target)
    
    @abstractmethod
    def _init_model(self):
        raise NotImplementedError
    
    @abstractmethod
    def _data_obj(self, stock):
        raise NotImplementedError
    @abstractmethod
    def _init_data(self):
        raise NotImplementedError
    
    def __get_datalen(self):
        for x, y in self.loader:
            pass
        return x.shape[0]
    
    def __get_time(self):
        return self.data.data.index[-self.len:]

    def __load_model(self, epoch):
        """
        Resume from saved checkpoints at: 'epoch-{}-{}.pth'.format(epoch, self.stock)

        :param resume_path: Checkpoint path to be resumed
        """
        # Load checkpoint
        if epoch == "best":
            path = self.ckpt_dir + f"{self.file_prefix}-best.pth"
        else:
            path = self.ckpt_dir + f"{self.file_prefix}-{epoch}.pth"
        
        if not os.path.exists(path):
            print("Checkpoint not found: {}".format(path))
            return False
        
        print("Loading checkpoint: {}".format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        
        return True
    
    # strategies
    def single_stock_training(self):
        ckpts = [0, 40, "best"]
        # update prefix for checkpoints file
        self.file_prefix = self.stock_target        
        self._plot(ckpts, "single")
        
    def multi_stock_training(self):
        ckpts = ["best"]
        self._plot(ckpts, "multiple")
    
    def buy_hold(self):
        self._plot()
    
    # plot
    def plot(self):
        plt.figure(figsize=(10, 6)) 
        self.buy_hold()
        self.multi_stock_training()
        self.single_stock_training()
        
        plt.xlabel("Time")
        plt.ylabel("Portfolio Value")
        plt.title(f"Model vs Buy & Hold Strategy for {self.stock_target}")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.performance_dir + f"{self.stock_target}: {self.stock}.png", dpi=300, bbox_inches="tight")
        plt.show()
        
    def _plot(self, ckpts: list = [], train_method: str = None):
        
        # the plot labels
        if train_method is None:
            label = f"BuyHold"
        if train_method == "multiple":
            label = f"Model multiple"
        if train_method == "single":
            label = f"Model single"
        
        print("Predicting " + label + "...")
        if ckpts != []:
            for epoch in ckpts:                         
                if self.__load_model(epoch):        
                    asset_hist = self.backtesting(self._test_method, self.model, (self.loader, self.data.src), verbose=False)
                    plt.plot(pd.DataFrame(asset_hist).set_index(self.dates), label=label+str(epoch))   # Model performance
                    
                    if train_method == "multiple":
                        print(asset_hist)
                    
                torch.cuda.empty_cache()
            
            return None
        
        asset_hist = self.backtesting(self._test_method, self.model, (self.loader, self.data.src), use_model=False, verbose=False)
        plt.plot(pd.DataFrame(asset_hist).set_index(self.dates), linestyle="dashed", label=label)  # Buy & Hold strategy
        torch.cuda.empty_cache()
        
    
    @staticmethod
    def backtesting(_test_method, model, data, use_model=True, short=True, verbose=True):
        """_summary_

        Args:
            _test_method (_type_): _description_
            model (_type_): _description_
            data (_type_): _description_
            use_model (bool, optional): _description_. Defaults to True.
            short (bool, optional): _description_. Defaults to True.
            verbose (bool, optional): _description_. Defaults to True.

        Returns:
            asset_hist: history of asset with model / buyhold
        """
        
        result, truth = _test_method(model, data)
        
        if not use_model:
            # Hold return
            asset_hold_hist = 1 + truth
            asset_hold_hist = torch.cumprod(asset_hold_hist, dim=0).cpu().numpy()
            return asset_hold_hist
        
        # Model return
        truth[result >= 0] = 1 + truth[result >= 0]
        if short:
            truth[result < 0] = 1 - truth[result < 0]
        else:
            truth[result < 0] = 1
        asset_hist = torch.cumprod(truth, dim=0).cpu().numpy()        
        
        if verbose:
            print(result)
            plt.plot(asset_hist)
            plt.show()
            
        return asset_hist
    
    @abstractmethod
    def _test_method(model, data):
        """
        - input: model, data, short
        - output: predictions of doc, shape (test_len, )
        """
        raise NotImplementedError
        
    
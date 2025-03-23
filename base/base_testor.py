from __future__ import annotations
import matplotlib.pyplot as plt
import torch
import pandas as pd
from abc import abstractmethod
import os


class Backtestor():
    def __init__(self, stock, model, data, dirs):
        
        self.stock = stock
        self.data = data
        self.model = model
        self.ckpt_dir = dirs["ckpt_dir"]
        self.performance_dir = dirs["performance_dir"]
        
        self.loader = self.data.testloader
        self.len = self.__get_datalen()
        self.dates = self.__get_time()
        
        print("Backtesting " + self.model.__name__ + " on " + self.stock)
    
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
            path = self.ckpt_dir + f"{self.stock}-best.pth"
        else:
            path = self.ckpt_dir + f"{self.stock}-{epoch}.pth"
        
        if not os.path.exists(path):
            print("Checkpoint not found: {}".format(path))
            return False
        
        print("Loading checkpoint: {}".format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        
        return True
        
    def __test_buyhold(self):
        
        test_data = self.data.data_origin.loc[self.dates]["Adj Close"]
        test_data /= test_data.iloc[0]
            
        return test_data
        
    def plot_result(self, ckpts: list = [], short: bool = True):
        """
        Backtest with test set
        """
        
        plt.figure(figsize=(10, 6)) 
        
        # Backtest buyhold
        plt.plot(self.__test_buyhold(), linestyle="dashed", label=f"BuyHold")  # Buy & Hold strategy
        
        # Backtest model
        ckpts.append("best")
        for epoch in ckpts:
            
            
            assert epoch is not int or epoch % 10 == 0, "Epoch must be multiple of 10"
            
            print(f"Predicting epoch {epoch}...")
            if self.__load_model(epoch):        
                asset_hist = self.test_model(self._test_method, self.model, (self.loader, self.data.src), short, verbose=False)
                df = pd.DataFrame({"Model": asset_hist,}).set_index(self.dates)
                plt.plot(df["Model"], label=f"Model Epoch {epoch}")   # Model performance
        
        
        plt.xlabel("Time")
        plt.ylabel("Portfolio Value")
        plt.title(f"Model vs Buy & Hold Strategy for {self.stock}")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.performance_dir + f"{self.stock}.png", dpi=300, bbox_inches="tight")
        plt.show()
    
    @staticmethod
    def test_model(_test_method, model, data, short, verbose=True):
        
        result, truth = _test_method(model, data)
        
        # Hold return
        asset_hold_hist = 1 + truth
        asset_hold_hist = torch.cumprod(asset_hold_hist, dim=0).cpu().numpy()
        
        # Model return
        truth = torch.zeros_like(result)
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
            
        return asset_hist, asset_hold_hist
    
    @abstractmethod
    def _test_method(model, data):
        """
        - input: model, data, short
        - output: predictions of doc, shape (test_len, )
        """
        raise NotImplementedError
        
    
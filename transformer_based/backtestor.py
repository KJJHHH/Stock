
import sys 
sys.path.append('../')
import matplotlib.pyplot as plt
import numpy as np
import pickle, json
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from transformer_based.utils import *
from transformer_based.datas import TransformerData
from transformer_based.models import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Backtestor():
    def __init__(self, stock, model, data, model_dir="./"):
        self.stock = stock[0]
        self.data = data
        self.model = model
        self.model_dir = model_dir
        
        self.test_loader = self.getTestLoader("test")
        self.test_len = self.getTestLen()
        self.test_dates = self.getTestTime()
        
        print("Backtesting " + self.model.__name__ + " on " + self.stock)
        
    def loadModel(self, epoch):
        if epoch == "best":
            self.model.load_state_dict(torch.load(f'{self.model_dir}{self.model.__name__}-result/{self.stock}_best.pt'))
        else:
            self.model.load_state_dict(torch.load(f'{self.model_dir}{self.model.__name__}-temp/{self.stock}_epoch_{epoch}.pt'))
    
    def getTestLoader(self, dataset = "test"):
        if dataset == "test":
            loader = self.data.testloader
        if dataset == "valid":
            loader = self.data.validloader
        if dataset == "train":
            loader = self.data.trainloader

        return loader
    
    def getTestLen(self):
        test_len = 0
        for x, y in self.test_loader:
            test_len += x.shape[0]
        return test_len
    
    def getTestTime(self):
        return self.data.data.index[-self.test_len:]

    def testBuyHold(self):
        
        test_data = self.data.data_origin.loc[self.test_dates]["Adj Close"]
        test_data /= test_data.iloc[0]
            
        return test_data
    
    
    @staticmethod
    def testModel(model, loader, src, short, verbose=False):
        
        def cumProduct(result, truth, short):
            truth[result >= 0] = 1 + truth[result >= 0]
            if short:
                truth[result < 0] = 1 - truth[result < 0]
            else:
                truth[result < 0] = 1
            return torch.cumprod(truth, dim=0)
        
        with torch.no_grad():
            # testloader number of batch = 1
            model.eval()
            for x_test, y_test in loader:
                x_test, y_test, src = x_test.to(device), y_test.to(device), src.to(device)
                _, result = model(src=src, tgt=x_test)
                result = result[:, -1, -1]
                truth = y_test[:, -1, -1]
                
        asset_hist = cumProduct(result, truth, short)
        asset_hist = asset_hist.cpu().numpy()
        
        if verbose:
            print(result)
            
        return asset_hist[-1], asset_hist
    
    def test(self, ckpts: list = [], short: bool = True):
        """
        Backtest with test set
        """
        
        ckpts.append("best")
        ckpts.append("buyhold")
        
        plt.figure(figsize=(10, 6)) 
        
        for epoch in ckpts:
            
            assert epoch is not int or epoch % 10 == 0, "Epoch must be multiple of 10"
            
            if epoch == "buyhold":
                plt.plot(self.testBuyHold(), linestyle="dashed", label=f"BuyHold")  # Buy & Hold strategy
                continue
                
            print(f"Predicting epoch {epoch}...")
            self.loadModel(epoch)        
            asset, asset_hist = self.testModel(self.model, self.test_loader, self.data.src, short, verbose=True)
            df = pd.DataFrame({"Model": asset_hist,}).set_index(self.test_dates)
            plt.plot(df["Model"], label=f"Model Epoch {epoch}")   # Model performance
        
        
        plt.xlabel("Time")
        plt.ylabel("Portfolio Value")
        plt.title(f"Model vs Buy & Hold Strategy for {self.stock}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.model_dir}{self.model.__name__}-result/{self.stock}.png", dpi=300, bbox_inches="tight")
        plt.show()
        



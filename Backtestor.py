
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
from Trainer import Seq2SeqPredictor

device = torch.device('cpu')

class Backtestor():
    def __init__(self, stock, model, data, model_dir="./"):
        self.stock = stock
        self.model = model
        self.model_dir = model_dir
        self.data = data
        
        print("Backtesting " + self.model.__name__ + " on " + self.stock)
        
    def loadModel(self, epoch):
        if epoch == "best":
            self.model.load_state_dict(torch.load(f'{self.model_dir}{self.model.__name__}-result/{self.stock}_best.pt'))
            self.model.to(device)
        else:
            self.model.load_state_dict(torch.load(f'{self.model_dir}{self.model.__name__}-temp/{self.stock}_epoch_{epoch}.pt'))
            self.model.to(device)
    
    
    def testing(self, loader):
        with torch.no_grad():
            # testloader number of batch = 1
            self.model.eval()
            for x_test, y_test in loader:
                _, result = self.model(src=self.data.src, tgt=x_test, test=True) # train, test set train = True, valid set False
                result = result[:, -1, -1].cpu().numpy()
                truth = y_test[:, -1, -1].cpu().numpy()
        return result, truth
    
    def getLoader(self, dataset = "test"):
        if dataset == "test":
            loader = self.data.testloader
        if dataset == "valid":
            loader = self.data.validloader
        if dataset == "train":
            loader = self.data.trainloader

        return loader
    
    def predict(self, dataset = "test", epoch="best"):
        print(f"Predicting epoch {epoch}...")
        
        loader = self.getLoader(dataset)
        self.loadModel(epoch)        
        result, truth = self.testing(loader)
        
        return result, truth

    def backtestModel(self, result, truth, thres: int = 0, short: bool = False):
        asset = 1
        asset_hist = []
        for doc_1, pred in zip(truth, result):
            if pred > thres:
                asset *= (1 + doc_1/100)
            if short and pred < thres:
                asset *= (1 - doc_1/100)
            asset_hist.append(asset)
            
        return asset_hist

    def backtestBuyHold(self, truth):
        asset_buyhold = 1
        asset_buyhold_hist = []

        for doc_1 in truth:
            asset_buyhold *= (1 + doc_1/100)
            asset_buyhold_hist.append(asset_buyhold)

        return asset_buyhold_hist
    
    def getTimeTest(self, len_test):
        return self.data.data.index[-len_test:]

    def main(self, epochs: list = [], short: bool = True):
        
        epochs.append("best")
        epochs.append("buyhold")
        
        plt.figure(figsize=(10, 6)) 
        
        for epoch in epochs:
            
            assert epoch is not int or epoch % 10 == 0, "Epoch must be multiple of 10"
            
            if epoch == "buyhold":
                asset_hist = self.backtestBuyHold(truth)
                time = self.getTimeTest(result.shape[0])
                df = pd.DataFrame({"BuyHold": asset_hist}).set_index(time)
                plt.plot(df["BuyHold"], linestyle="dashed", label=f"BuyHold")  # Buy & Hold strategy
                continue
                
            result, truth = self.predict(dataset="test", epoch=epoch)
            asset_hist = self.backtestModel(result, truth, 0, short)
            time = self.getTimeTest(result.shape[0])
            df = pd.DataFrame({"Model": asset_hist,}).set_index(time)
            plt.plot(df["Model"], label=f"Model Epoch {epoch}")   # Model performance
        
        
        plt.xlabel("Time")
        plt.ylabel("Portfolio Value")
        plt.title(f"Model vs Buy & Hold Strategy for {self.stock}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.model_dir}{self.model.__name__}-result/{self.stock}.png", dpi=300, bbox_inches="tight")
        plt.show()
        



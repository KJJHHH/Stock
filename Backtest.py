
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
from Predictor import Seq2SeqPredictor

device = torch.device('cpu')

class Backtestor():
    def __init__(self, stock, model, data, model_dir="./"):
        self.stock = stock
        self.model = model(train=False)
        self.model_dir = model_dir
        self.data = data
        
        print("Backtesting " + self.model.__name__ + " on " + self.stock)
        
    def loadBestModel(self):
        self.model.load_state_dict(torch.load(f'{self.model.__name__}-result/{self.stock}_best.pt')).to("cpu")
    
    def loadCkptModel(self, epoch):
        self.model.load_state_dict(torch.load(f'{self.model.__name__}-temp/{self.stock}_epoch_{epoch}.pt')).to("cpu")
    
    def testing(self, loader):
        print("Testing...")
        with torch.no_grad():
            # testloader number of batch = 1
            self.model.eval()
            for x_test, y_test in loader:
                _, result = self.model(src=corp.src, tgt=x_test, train=True) # train, test set train = True, valid set False
                result = result[:, -1, -1].cpu().numpy()
                truth = y_test[:, -1, -1].cpu().numpy()

        return result, truth
    
    def predict(self, dataset = "test", epoch=None):
        if dataset == "test":
            loader = self.data.testloader
        if dataset == "valid":
            loader = self.data.validloader
        if dataset == "train":
            loader = self.data.trainloader
        
        if epoch is not None:
            self.loadBestModel()
        else:
            assert epoch != None, "Need epoch checkpoint"
            assert epoch % 10 == 0, "Epoch must be multiple of 10"
            self.loadCkptModel()
        
        result, truth = self.testing(loader)
        
        return result, truth

    def backtest(self, result, truth, thres: int = 0, short: bool = False):
        
        asset = 1
        asset_buyhold = 1
        asset_hist = []
        asset_buyhold_hist = []
        
        for doc_1, pred in zip(truth, result):
            
            if pred > thres:
                asset *= (1 + doc_1/100)
            
            if short and pred < thres:
                asset *= (1 - doc_1/100)
                    
            asset_hist.append(asset)
            asset_buyhold *= (1 + doc_1/100)
            asset_buyhold_hist.append(asset_buyhold)
            
        return asset_hist, asset_buyhold_hist

    def getTimeTest(self, len_test):
        return self.data.data.index[-len_test:]

    def plotResult(self, epoch = None, short = False):
        # Plot best
        if epoch == None:
            result, truth = self.predict()
            result_val, _ = self.predict(test=False)
            thres = result_val.mean()
            asset_hist, asset_buyhold_hist = self.backtest(result, truth, thres, short)
            time = self.getTimeTest(result.shape[0])
            df = pd.DataFrame({"Model": asset_hist, "BuyHold": asset_buyhold_hist}).set_index(time)

            plt.plot(df["Model"], label=f"Model best")   # Model performance
            plt.plot(df["BuyHold"], linestyle="dashed", label=f"BuyHold")  # Buy & Hold strategy
        
        else:
            
            result_val, _ = self.predict(test=False, model_type="checkpoint", epoch=epoch)
            result, truth = self.predict(model_type="checkpoint", epoch=epoch)
            thres = result_val.mean()
            asset_hist, asset_buyhold_hist = self.backtest(result, truth, thres, short)
            time = self.getTimeTest(result.shape[0])
            df = pd.DataFrame({"Model": asset_hist, "BuyHold": asset_buyhold_hist}).set_index(time)

            plt.plot(df["Model"], label=f"Model Epoch {epoch}")   # Model performance    
            
    def main(self):
        pass
        


if __name__ == "__main__":

    stock_list = ["2884.TW", "2454.TW"]
    short = True

    for stock in stock_list:
        
        corp = Data(stock)
        corp.prepareData()

        plt.figure(figsize=(12, 6))  # Create a new figure for each stock

        for epoch in [0, 10, 20]:  # Loop over epochs
            evalPlot(epoch, short)
        
        evalPlot(short=short)

        # Add labels and legend
        plt.xlabel("Time")
        plt.ylabel("Portfolio Value")
        plt.title(f"Model vs Buy & Hold Strategy for {stock}")
        plt.legend()
        plt.grid(True)
        
        # Save each stock's figure separately
        plt.savefig(f"result/{stock}.png", dpi=300, bbox_inches="tight")

    plt.show()  # Display both figures

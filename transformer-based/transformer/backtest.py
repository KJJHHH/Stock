
import sys 
sys.path.append('../')
import matplotlib.pyplot as plt
import numpy as np
import pickle, json
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils import *
from datas import *
from Transformers import Transformer

device = torch.device('cpu')


def predict(test = True, model_type="best", epoch=None):
    
    MODEL = "Transformer"
    if model_type == "best":
        model = Transformer(test=True)
        model.load_state_dict(torch.load(f'result/{MODEL}_{stock}_best.pt'))
        model.to("cpu")
    else:
        assert epoch != None, "Need epoch checkpoint"
        assert epoch % 10 == 0, "Epoch must be multiple of 10"
        model = Transformer(test=True)
        model.load_state_dict(torch.load(f'result/ckpt-{MODEL}_{stock}_{epoch}.pt'))
        model.to("cpu")
    
    loader = corp.testloader if test else corp.trainloader
    
    with torch.no_grad():
        model.eval()
        for x_test, y_test in loader:
            # testloader number of batch = 1
            _, result = model(src=corp.src, tgt=x_test, train=True) # train, test set train = True, valid set False
            result = result[:, -1, -1].cpu().numpy()
            truth = y_test[:, -1, -1].cpu().numpy()
            
    return result, truth

def backtest(result, truth, thres: int = 0, short: bool = False):
    
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

def getTimeTest(len_test):
    return corp.data.index[-len_test:]



def evalPlot(epoch = None, short = False):
    
    # Plot best
    if epoch == None:
        result, truth = predict()
        result_val, _ = predict(test=False)
        thres = result_val.mean()
        asset_hist, asset_buyhold_hist = backtest(result, truth, thres, short)
        time = getTimeTest(result.shape[0])
        df = pd.DataFrame({"Model": asset_hist, "BuyHold": asset_buyhold_hist}).set_index(time)

        plt.plot(df["Model"], label=f"Model best")   # Model performance
        plt.plot(df["BuyHold"], linestyle="dashed", label=f"BuyHold")  # Buy & Hold strategy
    
    else:
        
        result_val, _ = predict(test=False, model_type="checkpoint", epoch=epoch)
        result, truth = predict(model_type="checkpoint", epoch=epoch)
        thres = result_val.mean()
        asset_hist, asset_buyhold_hist = backtest(result, truth, thres, short)
        time = getTimeTest(result.shape[0])
        df = pd.DataFrame({"Model": asset_hist, "BuyHold": asset_buyhold_hist}).set_index(time)

        plt.plot(df["Model"], label=f"Model Epoch {epoch}")   # Model performance    


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

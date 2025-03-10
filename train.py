import sys 
sys.path.append('../')
import torch

from transformer_based.utils import *
from transformer_based.datas import *
from transformer_based.models import *
from Predictor import Seq2SeqPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    stock = input("Enter stock id, eg. 2884.TW") or "2884.TW"
    
    models = ["Transformer", "Decoder-Only"]
    MODEL = models[int(input("Select model (enter number) -> 0: Transformer, 1: Decoder-Only") or 0)]
    
    """
    Init model in Predictor.py and Backtest.py (different arguement)
    """
    
    if MODEL == "Transformer":        
        data = TransformerData(stock)
        data.prepareData()
        
        model_dir = "transformer_based/"
        model = Transformer(
            d_model=6, 
            dropout=0.5, 
            d_hid=128, 
            nhead=2, 
            nlayers_e=64, 
            nlayers_d=16, 
            ntoken=10, 
            src_len=data.src.shape[1],
            train=True
        ).to(device)
        
    if MODEL == "Decoder-Only":
        model = None
        model_dir = "transformer_based/"
        
        data = TransformerData(stock)
        data.prepareData()
        
    predictor = Seq2SeqPredictor(stock, model=model, data=data, model_dir=model_dir, num_epochs=200, lr=0.001)
    predictor.train()

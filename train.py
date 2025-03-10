import sys 
sys.path.append('../')
import torch

from transformer_based.utils import *
from transformer_based.datas import *
from Predictor import Seq2SeqPredictor
from transformer_based.models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    stock = input("Enter stock id, eg. 2884.TW") or "2884.TW"
    
    models = ["Transformer", "Decoder-Only"]
    MODEL = models[int(input("Select model (enter number) -> 0: Transformer, 1: Decoder-Only") or 0)]
    
    if MODEL == "Transformer":
        model = Transformer
        model_dir = "transformer_based/"
        
    if MODEL == "Decoder-Only":
        model = None
        model_dir = "transformer_based/"
        
    predictor = Seq2SeqPredictor(stock, model=model, model_dir=model_dir, num_epochs=200, lr=0.001)
    predictor.train()

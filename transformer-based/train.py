import sys 
sys.path.append('../')
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import *
from datas import *
from accelerate import Accelerator

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from accelerate import Accelerator

from Transformer.Predictor import Seq2SeqPredictor
from Transformer.Transformers import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    stock = input("Enter stock id, eg. 2884.TW") or "2884.TW"
    
    models = ["Transformer", "Decoder-Only"]
    MODEL = models[int(input("Select model (enter number) -> 0: Transformer, 1: Decoder-Only") or 0)]
    
    if MODEL == "Transformer":
        model = Transformer
        model_dir = "Transformer/"
        
    if MODEL == "Decoder-Only":
        model = None
        model_dir = "Decoder-Only/"
        
    predictor = Seq2SeqPredictor(stock, model=model, model_dir=model_dir, num_epochs=200, lr=0.001)
    predictor.train()

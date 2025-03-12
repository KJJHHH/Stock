from transformer_based.models import Transformer
from transformer_based.datas import TransformerData
from Backtestor import Backtestor

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import sys 
sys.path.append('../')
import torch

from transformer_based.utils import *
from transformer_based.datas import *
from transformer_based.models import *
from Trainer import Seq2SeqPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    
    import argparse

    # Create argument parser
    parser = argparse.ArgumentParser(description="A simple argument parser example")
    parser.add_argument("task", help="train or test")  # Positional argument
    parser.add_argument("-m", "--model", help="Select model: [Transformer, Decoder only]")  # Positional argument
    parser.add_argument("-s", "--stock", nargs='+', help="Enter stock id, eg. 2884.TW")  # Positional argument
    
    # Parse arguments
    args = parser.parse_args()

    task = args.task
    stock = args.stock
    MODEL = args.model
    
    
    
    """
    Model dir: models in same model dir use same data
    |transformer_based
    |----|models.Transformer
    |----|...
    |cv_based
    |----|...
    """
    if MODEL == "Transformer":   
        
        ckpt_dir = "transformer_based/transformer-temp/"
        performance_dir = "transformer_based/transformer-result/"
        model = Transformer(
            d_model=6, 
            dropout=0.5, 
            d_hid=128, 
            nhead=2, 
            nlayers_e=64, 
            nlayers_d=16, 
            ntoken=10, 
            src_len=data.src.shape[1],
        ).to(device)
        
        config = {
            "name": MODEL,

            "arch": {
                "type": MODEL,
                "args": {}
            },
            "optimizer": {
                "type": "Adam",
                "args":{
                    "lr": 0.001,
                    "weight_decay": 0.00001,
                    "amsgrad": True
                }
            },
            "lr_scheduler": {
                "type": "StepLR",
                "args": {
                    "step_size": 1,
                    "gamma": 0.5
                }
            },
            "start_date": "2017-01-01",
            "end_date": "2025-01-01",
            "epochs": 200,
            "val_type": "loss"

        }
        
    if MODEL == "Decoder-Only":
        pass
    
    if task == "train":
        predictor = Seq2SeqPredictor(
            stock=stock, 
            model=model, 
            config=config,  
            dirs=[ckpt_dir, performance_dir]
        )
        predictor.train()
        
    if task == "test":
        testor = Backtestor(stock, model=model, data=data, model_dir=model_dir)
        testor.test(ckpts=[0, 20, 40, 60, 190], short=True)

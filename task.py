import torch

from datas import *
from models import *
from trainers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    
    import argparse

    # Create argument parser
    parser = argparse.ArgumentParser(description="A simple argument parser example")
    parser.add_argument("task", help="train or test")  
    parser.add_argument("-m", "--model", help="Select model: [Transformer, Decoder only]")  
    parser.add_argument("-s", "--stock", nargs='+', help="Enter stock id, eg. 2884.TW")  

    # Parse arguments
    args = parser.parse_args()

    task = args.task
    stock_list = args.stock
    MODEL = args.model
    
    ckpt_dir = "results/" + MODEL + "-temp/"
    performance_dir = "results" + MODEL + "-result/"
    
    """
    Model dir: models in same model dir use same data
    |transformer_based
    |----|models.Transformer
    |----|...
    |cv_based
    |----|...
    """
    # MODEL type
    if MODEL == "Transformer": 
        
        config = {
            # data
            "ntoken": 5,
            "start_date": "2016-01-01",
            "end_date": "2024-12-28",
            
            # model
            "name": MODEL,
            "epochs": 200,
            # loss or  asset
            "val_type": "asset",
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
        }
        
        # Target stock data
        data = TransformerBasedData(
            stock=stock_list[0],
            start_date=config["start_date"],
            end_date=config["end_date"],
            window=config["ntoken"],
            batch_size=64,
            )
        model = Transformer(
            d_model=6, 
            dropout=0.5, 
            d_hid=128, 
            nhead=2, 
            nlayers_e=64, 
            nlayers_d=16, 
            ntoken=config["ntoken"], 
            src_len=data.src_len,
        ).to(device)
        trainer = TransformerTrainer
        
    if MODEL == "Decoder-Only":
        pass
    
    if MODEL == "Resnet":
        
        config = {
            # data
            "ntoken": 50,
            "start_date": "2016-01-01",
            "end_date": "2024-12-28",
            
            # model
            "name": MODEL,
            "epochs": 200,
            # loss or  asset
            "val_type": "asset",
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
        }
        
        data = CVBasedData(
            stock=stock_list[0],
            start_date=config["start_date"],
            end_date=config["end_date"],
            window=config["ntoken"],
            batch_size=64,
            )
        model = ResNet(BasicBlock, [3, 4, 6, 3], 1).to(device)
        trainer = ResnetTrainer
    
    # Tasks
    if task == "train":
        trainer = trainer(
            stock_list=stock_list, 
            data=data,
            model=model, 
            config=config,  
            dirs=[ckpt_dir, performance_dir]
        )
        trainer.train()
        
    if task == "test":
        testor = Backtestor(stock_list, model=model, data=data, model_dir=model_dir)
        testor.test(ckpts=[0, 20, 40, 60, 190], short=True)

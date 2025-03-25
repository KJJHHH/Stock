from __future__ import annotations
import torch

from base.base_testor import Backtestor
from models import *
from datas import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerBacktestor(Backtestor):
    def __init__(self, stock, config, dirs):
        super().__init__(stock, config, dirs)
    
    def _data_obj(self, stock):
        return TransformerData(
            stock=stock,
            config=self.config,
            )
    
    def _init_data(self):
        data = self._data_obj(self.stock_target)
        self.src_len = data.src.shape[1]  
        return data
    
    def _init_model(self):
        return Transformer(
            d_model=self.config["d_model"],
            dropout=self.config["dropout"],
            d_hid=self.config["d_hid"],
            nhead=self.config["nhead"],
            nlayers_e=self.config["nlayers_e"],
            nlayers_d=self.config["nlayers_d"],
            ntoken=self.config["ntoken"],
            ).to(device) 
        
    @staticmethod
    def _test_method(model, data):
        
        loader = data[0]
        src = data[1]
        
        model.eval()
        with torch.no_grad(): # testloader number of batch = 1
            for x_test, y_test in loader:
                x_test, y_test, src = x_test.to(device), y_test.to(device), src.to(device)
                result = model(src=src, tgt=x_test)
                result = result[:, -1, -1].detach()
                truth = y_test[:, -1, -1].detach()   
                    
        
            return result, truth


class ResnetBacktestor(Backtestor):
    def _backtest_method(self, model, data):
        loader = data
        with torch.no_grad():
            # testloader number of batch = 1
            model.eval()
            for x_test, y_test in loader:
                x_test, y_test, src = x_test.to(device), y_test.to(device), src.to(device)
                result = model(src=src, tgt=x_test)
                result = result
                truth = y_test
                
        return result, truth
from __future__ import annotations
import torch

from base.base_testor import Backtestor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerBacktestor(Backtestor):
    def __init__(self, stock, model, data, dirs):
        super().__init__(stock, model, data, dirs)
    
    @staticmethod
    def _test_method(model, data):
        
        loader = data[0]
        src = data[1]
        
        model.eval()

        with torch.no_grad(): # testloader number of batch = 1
            for x_test, y_test in loader:
                x_test, y_test, src = x_test.to(device), y_test.to(device), src.to(device)
                _, result = model(src=src, tgt=x_test)
                result = result[:, -1, -1]
                truth = y_test[:, -1, -1]
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
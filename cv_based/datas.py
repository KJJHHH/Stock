import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from base.datas import BaseData
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CVBasedData(BaseData):
    def __init__(self, 
        stock: str, 
        start_date: str = '2016-01-02',
        end_date: str = '2024-12-31',
        window: int = 50,
        
        batch_size: int = 64, 
        percentage_test: float = 0.05, 
        percentage_valid: float = 0.05, 
        ) -> None:
        
        super().__init__(stock, start_date, end_date, window, batch_size)

        self._prepare_model_datas(percentage_test, percentage_valid)        
    
    def _prepare_model_datas(self, percentage_test: float = 0.2, percentage_valid: float = 0.2):
        """get train, valid, test 
        - preprocessing
        - split data
        - dataloader     
        """
        
        self.data = self.fetchPrice()
        self.data_origin = self.data.copy()
        self.createVarTarget()
        self.clean()
        self.normalize()
        
        # windows and tensor
        self._get_date()
        X_list, y_list = self._window_Xy()
        X, y = self.toTensor(X_list, y_list)
        
        # data sizes
        self.splitSize(percentage_test, percentage_valid)        

        # data split
        x_train, x_test, y_train, y_test = self.split(X, y, self.test_len)
        x_train, x_valid, y_train, y_valid = self.split(x_train, y_train, self.valid_len)
        
        # data for training
        self.getLoaders((x_train, x_valid, x_test, y_train, y_valid, y_test), self.batch_size)
        
        # print data shape
        print(f"""Data Shape: 
        x_train: {x_train.shape}, 
        x_valid: {x_valid.shape}, 
        x_test: {x_test.shape}, 
        y_train: {y_train.shape}, 
        y_valid: {y_valid.shape}, 
        y_test: {y_test.shape}
        """)

    def _get_date(self):
        """
        Get the date for start of train to test end
        """
        
        # remove first window-1 and last one since y shift 1
        self.dates = self.data.index[self.window-1:len(self.data)-1]
        self.train_dates = ...
        self.valid_dates = ...
        self.test_dates = ...
    
    def _window_Xy(self): # df: before split
        
        x_list, y_list = [], []
        
        for i in tqdm(range(len(self.data)-self.window)): 
            x_window = self.data.iloc[i:i+self.window]
            y_window = self.data.iloc[i+1:i+self.window+1]  
            x_values = x_window[['do', 'dh', 'dl', 'dc', 'dv', 'doc']].values  
            y_values = y_window[['do', 'dh', 'dl', 'dc', 'dv', 'doc']].values
            x_list.append(x_values)
            y_list.append(y_values)
        
        assert len(x_list) == len(self.dates), "Mismatch time index and data"
        
        return x_list, y_list
    
    def _process_X(self, x):
        N = len(x)
        X = []
        x = torch.tensor(x, dtype=torch.float32).to(device)
        for i in tqdm(range(N)):
            X_element = []
            for j in range(self.window):
                X_element.append(self._gaf(x[i][j])).unsqueeze(0)
            X_element = torch.cat(X_element, dim=0).unsqueeze(0)
            X.append(X_element)
        X = torch.cat(X, dim=0)
        return X
    
    def _gaf(X):
        X = X.reshape(-1)
        X_diff = X.unsqueeze(0) - X.unsqueeze(1) # Pairwise differences
        GAF = torch.cos(X_diff)# Gramian Angular Field

        return GAF

if __name__ == "__main__":
    
    data = CVBasedData(stock='AAPL', percentage_test=0.05, percentage_valid=0.05)
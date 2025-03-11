import numpy as np 
import pandas as pd
import pickle
import torch
from sklearn.preprocessing import StandardScaler
import yfinance as yf

from torch.utils.data import DataLoader, TensorDataset

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransformerData():
    def __init__(self,
        stock: str, 
        start_date: str = '2016-01-02',
        end_date: str = '2024-12-31',
        window: int = 5,
        
        batch_size: int = 64, 
        percentage_test: float = 0.05, 
        percentage_valid: float = 0.05, 
        src_len: int = 0
        ):
        
        self.stock = stock
        self.start_date = start_date
        self.end_date = end_date
        self.window = window      
        
        self.data = self.fetchPrice()
        
        """Shape
        - loaders: (batch size, seq len, features)
        - src: (1, total seq len, features)
        """
        self.train_len = None
        self.valid_len = None
        self.test_len = None
        self.trainloader = None
        self.validloader = None
        self.testloader = None
        self.src = None
        
        # For testing result
        self.time = None
        self.data_origin = self.data.copy()
        
        # Prepare data
        self.prepareData(batch_size, percentage_test, percentage_valid, src_len)
    
    def fetchPrice(self):
        data = yf.Ticker(self.stock)
        data = data.history(start=self.start_date, end=self.end_date)
        return data
    
    def createVarTarget(self):
        self.data['do'] = self.data['Open'].pct_change() * 100
        self.data['dh'] = self.data['High'].pct_change() * 100
        self.data['dl'] = self.data['Low'].pct_change() * 100
        self.data['dc'] = self.data['Close'].pct_change() * 100
        self.data['dv'] = self.data['Volume'].pct_change() * 100
        self.data['doc'] = ((self.data['Close'] - self.data['Open'])/self.data['Open'])
        
        """
        self.data['doc_1'] = \
            ((self.data['Close'].shift(-1) - self.data['Open'].shift(-1))/self.data['Open'].shift(-1))*100
        """
            
        self.data = self.data[['do', 'dh', 'dl', 'dc', 'dv', 'doc']] 
        
    def clean(self):
        # Some value is inf in "dv"
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data.dropna(inplace=True)
    
    def normalize(self):
        """Normalize

        Args:
            size (int, optional): need <= train size. Defaults to 2500.
        """
        scaler = StandardScaler()
        scaler.fit(self.data[['do', 'dh', 'dl', 'dc', 'dv']][:self.train_len])
        self.data[['do', 'dh', 'dl', 'dc', 'dv']] = scaler.transform(self.data[['do', 'dh', 'dl', 'dc', 'dv']])

    def getDate(self):
        # remove first window-1 and last one since y shift 1
        self.time = self.data.index[self.window-1:len(self.data)-1]
    
    def windowXYByDate(self): 
        """Transform to training form of data and get the date of each sample of data
        """
        self.getDate()
        
        x_list, y_list = [], []
        
        # y shift 1 from x
        for i in range(len(self.data)-self.window): 
            x_window = self.data.iloc[i:i+self.window]
            y_window = self.data.iloc[i+1:i+self.window+1]  
            x_values = x_window[['do', 'dh', 'dl', 'dc', 'dv', 'doc']].values  
            y_values = y_window[['do', 'dh', 'dl', 'dc', 'dv', 'doc']].values
            x_list.append(x_values)
            y_list.append(y_values)
        
        assert len(x_list) == len(self.time), "Mismatch time index and data"
        
        return x_list, y_list
    
    def toTensor(self, X: list, y: list):
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    def splitSize(self, percentage_test: float = 0.2, percentage_valid: float = 0.2):
        self.test_len = int(len(self.data) * percentage_test)
        self.train_len = len(self.data) - self.test_len
        self.valid_len = int(self.train_len * percentage_valid)
        self.train_len = self.train_len - self.valid_len
    
    def trainTestSplit(self, X: torch.tensor, y: torch.tensor):
        """get train, valid, test for transfomer decoder
        """
        
        def split(X: torch.tensor, y: torch.tensor, test_size: int):
            train_size = len(X) - test_size
            x_train = X[:train_size]
            x_test = X[train_size:]
            y_train = y[:train_size]
            y_test = y[train_size:]
            
            return x_train, x_test, y_train, y_test, 
    
        x_train, x_test, y_train, y_test = split(X, y, self.test_len)
        x_train, x_valid, y_train, y_valid = split(x_train, y_train, self.valid_len)
        
        print(f"""Data Shape: \
            x_train: {x_train.shape}, \
            x_valid: {x_valid.shape}, \
            x_test: {x_test.shape}, \
            y_train: {y_train.shape}, \
            y_valid: {y_valid.shape}, \
            y_test: {y_test.shape}""")
        
        return (x_train, x_valid, x_test, y_train, y_valid, y_test)
    
    def getSrc(self, x_train: torch.tensor, src_len: int = 0):   
        """get src data for transformer encoder
        """
        
        assert src_len <= len(x_train), "src_len must be less than or equal to train size"
        
        if src_len == 0:
            src_len = len(x_train)
        self.src = x_train[:src_len][:, -1, :].unsqueeze(0)
    
    def getLoaders(self, datas, batch_size):
        
        def loader(X: torch.tensor, y: torch.tensor, batch_size: int = 128):
            dataset = TensorDataset(X, y)
            dataloader = DataLoader(dataset, batch_size, shuffle=False, drop_last=True)
            return dataloader
        
        x_train, x_valid, x_test, y_train, y_valid, y_test = datas
        self.trainloader = loader(x_train, y_train, batch_size)
        self.validloader = loader(x_valid, y_valid, len(x_valid))
        self.testloader = loader(x_test, y_test, len(x_test))
        
    def prepareData(self, 
        batch_size: int = 128, 
        percentage_test: float = 0.2, 
        percentage_valid: float = 0.2, 
        src_len: int = 0
        ):
        
        self.createVarTarget()
        self.clean()
        self.splitSize(percentage_test=percentage_test, percentage_valid=percentage_valid)
        self.normalize()
        X_list, y_list = self.windowXYByDate()
        X, y = self.toTensor(X_list, y_list)
        datas = self.trainTestSplit(X, y)
        self.getSrc(datas[0], src_len)
        self.getLoaders(datas, batch_size)
    
if __name__ == "__main__":
    data = TransformerData(stock="2330.TW")
    print(data.src.shape)
    print(len(data.trainloader))
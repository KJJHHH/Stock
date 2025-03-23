import numpy as np 
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from abc import abstractmethod

from torch.utils.data import DataLoader, TensorDataset

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BaseData():
    def __init__(self,
        stock: str, 
        config: dict,
        ):
        
        self.stock = stock
        self.start_date = config['train_start']
        self.end_date = config["end_date"]
        self.window = config["ntoken"] 
        self.batch_size = config["batch_size"]
        
        self.train_start = pd.Timestamp(config["train_start"])
        self.valid_start = pd.Timestamp(config["valid_start"])
        self.test_start = pd.Timestamp(config["test_start"])
        self.train_dates = ...
        self.valid_dates = ...
        self.test_dates = ...
        self.train_len = ...
        self.valid_len = ...
        self.test_len = ...
        
        self.trainloader = ...
        self.validloader = ...
        self.testloader = ...
        
        """
        self._prepare_model_datas(percentage_test, percentage_valid) 
        """
    
    def _fetch_price(self):
        
        data = yf.download(self.stock, start=self.start_date, end=self.end_date, interval="1d", group_by='ticker', auto_adjust=False, prepost=False, threads=True, proxy=None)
        data.columns = data.columns.droplevel(0)
        return data
    
    def _create_var_target(self):
        """Variables to predict "doc"
        """
        self.data['do'] = self.data['Open'].pct_change() * 100
        self.data['dh'] = self.data['High'].pct_change() * 100
        self.data['dl'] = self.data['Low'].pct_change() * 100
        self.data['dc'] = self.data['Close'].pct_change() * 100
        self.data['dac'] = self.data['Adj Close'].pct_change() * 100
        self.data['dv'] = self.data['Volume'].pct_change() * 100
        self.data['doc'] = ((self.data['Adj Close'] - self.data['Open'])/self.data['Open'])
        
        columns = ['do', 'dh', 'dl', 'dc', 'dv', 'dac', 'doc']
            
        self.data = self.data[columns] 
        
    def _clean(self):
        # Some value is inf in "dv"
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data.dropna(inplace=True)
    
    def _normalize(self):
        """Normalize

        Args:
            size (int, optional): need <= train size. Defaults to 2500.
        """
        scaler = StandardScaler()
        columns = ['do', 'dh', 'dl', 'dc', 'dv', 'dac']
        scaler.fit(self.data[columns][:self.train_len])
        self.data[columns] = scaler.transform(self.data[columns])

    def getLoaders(self, datas, batch_size):
        
        def loader(X: torch.tensor, y: torch.tensor, batch_size: int = 128):
            dataset = TensorDataset(X, y)
            dataloader = DataLoader(dataset, batch_size, shuffle=False, drop_last=True)
            return dataloader
        
        X_train, y_train, X_test, y_test, X_valid, y_valid = datas
        self.trainloader = loader(X_train, y_train, batch_size)
        self.validloader = loader(X_valid, y_valid, len(X_valid))
        self.testloader = loader(X_test, y_test, len(X_test))
        
    @abstractmethod
    def prepare_data(self, percentage_test: float, percentage_valid: float):
        """get train, valid, test
        TODO: 
        - preprocessing
        - split data
        - dataloader     
        """
        self.data = self._fetch_price()
        self.data_origin = self.data.copy()
        self._create_var_target()
        self._clean()
        self._normalize()
        ...
        raise NotImplementedError
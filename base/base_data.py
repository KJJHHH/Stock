import numpy as np 
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
        start_date: str = '2016-01-02',
        end_date: str = '2024-12-31',
        window: int = 5,
        batch_size: int = 64, 
        ):
        
        self.stock = stock
        self.start_date = start_date
        self.end_date = end_date
        self.window = window    
        self.batch_size = batch_size
        
        self.train_len = None
        self.valid_len = None
        self.test_len = None
        self.trainloader = None
        self.validloader = None
        self.testloader = None
        
        # Dates, total date, train dates, ...
        self.dates = None
        self.train_dates = None
        self.valid_dates = None
        self.test_dates = None
        
        """
        self._prepare_model_datas(percentage_test, percentage_valid) 
        """
    
    def fetchPrice(self):
        
        data = yf.download(self.stock, start=self.start_date, end=self.end_date, interval="1d", group_by='ticker', auto_adjust=False, prepost=False, threads=True, proxy=None)
        data.columns = data.columns.droplevel(0)
        return data
    
    def createVarTarget(self):
        """Variables to predict "doc"
        """
        self.data['do'] = self.data['Open'].pct_change() * 100
        self.data['dh'] = self.data['High'].pct_change() * 100
        self.data['dl'] = self.data['Low'].pct_change() * 100
        self.data['dc'] = self.data['Close'].pct_change() * 100
        self.data['dac'] = self.data['Adj Close'].pct_change() * 100
        self.data['dv'] = self.data['Volume'].pct_change() * 100
        self.data['doc'] = ((self.data['Close'] - self.data['Open'])/self.data['Open'])
        
        columns = ['do', 'dh', 'dl', 'dc', 'dv', 'dac', 'doc']
            
        self.data = self.data[columns] 
        
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
        columns = ['do', 'dh', 'dl', 'dc', 'dv', 'dac']
        scaler.fit(self.data[columns][:self.train_len])
        self.data[columns] = scaler.transform(self.data[columns])
    
    def toTensor(self, X: list, y: list):
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    def splitSize(self, percentage_test: float, percentage_valid: float):        
        self.test_len = int(len(self.data) * percentage_test)
        self.train_len = len(self.data) - self.test_len
        self.valid_len = int(self.train_len * percentage_valid)
        self.train_len = self.train_len - self.valid_len
    
    def split(self, X: torch.tensor, y: torch.tensor, test_size: int):
        train_size = len(X) - test_size
        x_train = X[:train_size]
        x_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        return x_train, x_test, y_train, y_test, 
        
    def getLoaders(self, datas, batch_size):
        
        def loader(X: torch.tensor, y: torch.tensor, batch_size: int = 128):
            dataset = TensorDataset(X, y)
            dataloader = DataLoader(dataset, batch_size, shuffle=False, drop_last=True)
            return dataloader
        
        x_train, x_valid, x_test, y_train, y_valid, y_test = datas
        self.trainloader = loader(x_train, y_train, batch_size)
        self.validloader = loader(x_valid, y_valid, len(x_valid))
        self.testloader = loader(x_test, y_test, len(x_test))
        
    @abstractmethod
    def _prepare_model_datas(self, percentage_test: float, percentage_valid: float):
        """get train, valid, test
        TODO: 
        - preprocessing
        - split data
        - dataloader     
        """
        self.data = self.fetchPrice()
        self.data_origin = self.data.copy()
        self.createVarTarget()
        self.clean()
        self.normalize()
        ...
        raise NotImplementedError
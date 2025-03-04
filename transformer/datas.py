import numpy as np 
import pandas as pd
import pickle
import torch
from sklearn.preprocessing import StandardScaler
from utils import *
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Data():
    def __init__(self,
        stock: str, 
        start_date: str = '2020-01-02',
        end_date: str = '2024-12-31',
        batch_size: int = 64,
        window: int = 10
        ):
        self.stock = stock
        self.start_date = start_date
        self.end_date = end_date
        self.batch_size = batch_size
        self.window = window      
        
        self.data = self.fetchPrice()
        self.train_size = None
        self.valid_size = None
        self.test_size = None
        
        # For testing result
        self.time = None
        self.data_origin = self.data.copy()
    
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
        self.data['doc_1'] = \
            ((self.data['Close'].shift(-1) - self.data['Open'].shift(-1))/self.data['Open'].shift(-1))*100
            
        self.data = self.data[['do', 'dh', 'dl', 'dc', 'dv', 'Close', 'doc_1']] 
        
    def clean(self):
        # Some value is inf in "dv"
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data.dropna(inplace=True)
    
    def splitSize(self, percentage_test: int = 0.05, percentage_valid = 0.05):
        self.test_size = int(len(self.data) * percentage_test)
        self.train_size = len(self.data) - self.test_size
        self.valid_size = int(self.train_size * percentage_valid)
        self.train_size = self.train_size - self.valid_size
    
    def normalize(self, size: str = 2500):
        """Normalize

        Args:
            size (int, optional): need <= train size. Defaults to 2500.
        """
        scaler = StandardScaler()
        scaler.fit(self.data[['do', 'dh', 'dl', 'dc', 'dv', 'Close']][:self.train_size])
        self.data[['do', 'dh', 'dl', 'dc', 'dv', 'Close']] = scaler.transform(self.data[['do', 'dh', 'dl', 'dc', 'dv', 'Close']])

    def getDate(self):
        self.time = self.data.index
    
    def windowXYByDate(self, window_size: int = 100): 
        self.getDate()
        x_list, y_list = [], []
        for i in range(len(self.data)-window_size+1): 
            window = self.data.iloc[i:i+window_size]  
            x_values = window[['do', 'dh', 'dl', 'dc', 'dv', 'Close']].T.values  
            y_values = window[['doc_1']].iloc[-1].T.values
            x_list.append(x_values)
            y_list.append(y_values)
        return x_list, y_list
    
    def toTensor(self, X: list, y: list):
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    def trainTestSplit(self, X: torch.tensor, y: torch.tensor):
        """get train, valid, test for transfomer decoder

        Args:
            X (_type_): _description_
            y (_type_): _description__

        Returns:
            _type_: _description_
        """
        
        def split(X: torch.tensor, y: torch.tensor, test_size: int):
            train_size = len(X) - test_size
            x_train = X[:train_size]
            x_test = X[train_size:]
            y_train = y[:train_size]
            y_test = y[train_size:]
            
            return x_train, x_test, y_train, y_test, 
    
        x_train, x_test, y_train, y_test = split(X, y, self.test_size)
        x_train, x_valid, y_train, y_valid = split(x_train, y_train, self.valid_size)
        
        print(f"""Shape:
            x_train: {x_train.shape}, 
            x_valid: {x_valid.shape},
            x_test: {x_test.shape},
            y_train: {y_train.shape},
            y_valid: {y_valid.shape},
            y_test: {y_test.shape}""")
        
        return (x_train, x_valid, x_test, y_train, y_valid, y_test)
    
    def getSrc(self, x_train: torch.tensor, size: int = None):   
        """get source data for transformer encoder

        Args:
            x_train
        Returns:
            torch.tensor
        """
        if size == None:
            size = len(x_train)
        return x_train[:size]
    
    def getLoaders(self, datas):
        def loader(X: torch.tensor, y: torch.tensor, batch_size: int = 128):
            dataset = TensorDataset(X, y)
            dataloader = DataLoader(dataset, batch_size, shuffle=False, drop_last=True)
            return dataloader
        
        x_train, x_valid, x_test, y_train, y_valid, y_test = datas
        trainloader = loader(x_train, y_train)
        validloader = loader(x_valid, y_valid, 16)
        testloader = loader(x_test, y_test,)
        
        return (trainloader, validloader, testloader)
    
    def prepareData(self):
        self.createVarTarget()
        self.clean()
        self.splitSize()
        self.normalize()
        X_list, y_list = self.windowXYByDate(self.window)
        X, y = self.toTensor(X_list, y_list)
        datas = self.trainTestSplit(X, y)
        src = self.getSrc(datas[0])
        loaders = self.getLoaders(datas)
        
        return src, loaders
    
if __name__ == "__main__":
    data = Data(stock="2330.TW")
    src, loaders = data.prepareData()
    print(src.shape)
    print(len(loaders[0]))
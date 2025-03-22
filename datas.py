import torch
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from base.base_data import BaseData

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransformerData(BaseData):
    def __init__(self,
        stock: str, 
        window: int = 5,
        
        batch_size: int = 64, 
        src_len = None
        ) -> None:
        super().__init__(stock, window, batch_size)        
        """Shape
        - loaders: (batch size, seq len, features)
        - src: (1, total seq len, features)
        """
        self.src = ...
        self.src_len = src_len
        
        # Prepare data
        self.data = self._fetch_price()
        self.data_origin = self.data.copy()
        self._create_var_target()
        self._clean()
        self.prepare_data()
        # self._normalize()
        
    def __get_src(self, x_train: torch.tensor, src_len: int = 0):   
        """get src data for transformer encoder
        """
        
        if self.src_len == 0:
            self.src_len = len(x_train)
            
        assert self.src_len <= len(x_train) , "src_len must be less than or equal to train size"
        
        self.src = x_train[:src_len][:, -1, :].unsqueeze(0).to(device)
       
    def prepare_data(self): 
        """Transform to training form of data and get the date of each sample of data
        """
        
        def append_data(samples, lists):
            """
            - example
            X_test.append(x_window.values)
            y_test.append(y_window.values) 
            test_dates.append(date.values)
            """
            for s, l in zip(samples, lists):
                l.append(s)
        
        X_train, y_train, X_valid, y_valid, X_test, y_test, train_dates, valid_dates, test_dates = \
            [], [], [], [], [], [], [], [], []        
        
        for i in range(len(self.data)-self.window): 
            # y shift 1 from x
            x_window = self.data.iloc[i:i+self.window][['do', 'dh', 'dl', 'dv', 'dac', 'doc']].values
            y_window = self.data.iloc[i+1:i+self.window+1][['do', 'dh', 'dl', 'dv', 'dac', 'doc']].values
            date = self.data.index[i+self.window]
            
            if date < self.valid_start:
                append_data((x_window, y_window, date), (X_train, y_train, train_dates))
            elif date < self.test_start:
                append_data((x_window, y_window, date), (X_valid, y_valid, valid_dates))
            else:
                append_data((x_window, y_window, date), (X_test, y_test, test_dates))
        
        self.train_dates = train_dates
        self.valid_dates = valid_dates
        self.test_dates = test_dates
        
        X_train, y_train, X_test, y_test, X_valid, y_valid = \
            torch.tensor(X_train, dtype = torch.float32), torch.tensor(y_train, dtype = torch.float32), \
            torch.tensor(X_test, dtype = torch.float32), torch.tensor(y_test, dtype = torch.float32), \
            torch.tensor(X_valid, dtype = torch.float32), torch.tensor(y_valid, dtype = torch.float32)
        
        self.src_len = X_train.shape[0]        
        self.__get_src(X_train, self.src_len)
        self.getLoaders((X_train, y_train, X_test, y_test, X_valid, y_valid), self.batch_size)
        
        print(f"""Data Shape: 
        x_train: {X_train.shape}, 
        x_valid: {X_valid.shape}, 
        x_test: {X_test.shape}, 
        y_train: {y_train.shape}, 
        y_valid: {y_valid.shape}, 
        y_test: {y_test.shape},
        src: {self.src.shape}""")
        
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

        print("Initializing data...")
        self.prepare_data(percentage_test, percentage_valid)        
    
    def prepare_data(self, percentage_test: float = 0.2, percentage_valid: float = 0.2):
        """get train, valid, test 
        - preprocessing
        - split data
        - dataloader     
        """
        
        self.data = self._fetch_price()
        self.data_origin = self.data.copy()
        self._create_var_target()
        self._clean()
        self._normalize()
        
        # windows and tensor
        self._get_date()
        X_list, y_list = self._window_Xy()
        X, y = self._to_tensor(X_list, y_list)
        X = self._process_X(X)
        
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
        
        for i in range(len(self.data)-self.window): 
            x_window = self.data.iloc[i:i+self.window]
            y_window = self.data.iloc[i+1:i+self.window+1]  
            x_values = x_window[['do', 'dh', 'dl', 'dc', 'dv', 'doc']].values  
            y_values = y_window[['do', 'dh', 'dl', 'dc', 'dv', 'doc']].values[-1][-1]
            x_list.append(x_values)
            y_list.append(y_values)
        
        assert len(x_list) == len(self.dates), "Mismatch time index and data"
        
        return x_list, y_list
    
    def _process_X(self, x):
        
        N = len(x)
        D = x.shape[-1]
        X = torch.tensor([])
        
        for i in tqdm(range(N)):
            X_element = torch.cat([self._gaf(x[i, :, j]) for j in range(D)], dim = 0).unsqueeze(0)
            X = torch.cat((X, X_element), dim=0)
        return X
    
    def _gaf(self, X):
        X = X.reshape(-1)
        X_diff = X.unsqueeze(0) - X.unsqueeze(1) # Pairwise differences
        GAF = torch.cos(X_diff).unsqueeze(0) # Gramian Angular Field
        return GAF

if __name__ == "__main__":
    
    data_class = CVBasedData(stock='AAPL', percentage_test=0.05, percentage_valid=0.05)
    data_class = TransformerData(stock="2330.TW")
    print(data_class.src.shape)
    print(len(data_class.trainloader))
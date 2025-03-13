import torch

import warnings
warnings.filterwarnings("ignore")

from base.datas import BaseData

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransformerData(BaseData):
    def __init__(self,
        stock: str, 
        start_date: str = '2016-01-02',
        end_date: str = '2024-12-31',
        window: int = 5,
        
        batch_size: int = 64, 
        percentage_test: float = 0.05, 
        percentage_valid: float = 0.05, 
        src_len = None
        ) -> None:
        super().__init__(stock, start_date, end_date, window, batch_size)        
        """Shape
        - loaders: (batch size, seq len, features)
        - src: (1, total seq len, features)
        """
        self.src = None
        self.src_len = src_len
        
        # Prepare data
        self.data = self.fetchPrice()
        self.data_origin = self.data.copy()
        self.createVarTarget()
        self.clean()
        self.normalize()
        self._prepare_model_datas(percentage_test, percentage_valid)
    
    def _prepare_model_datas(self, percentage_test: float = 0.2, percentage_valid: float = 0.2):
        """get train, valid, test for transfomer decoder
        TODO:
        - split data by dates        
        """
        # windows and tensor
        self._get_date()
        X_list, y_list = self._window_xy_by_date()
        X, y = self.toTensor(X_list, y_list)
        
        # data sizes
        self.splitSize(percentage_test, percentage_valid)        

        # data split
        x_train, x_test, y_train, y_test = self.split(X, y, self.test_len)
        x_train, x_valid, y_train, y_valid = self.split(x_train, y_train, self.valid_len)
        
        # data for training
        self._get_src(x_train, self.src_len)
        self.getLoaders((x_train, x_valid, x_test, y_train, y_valid, y_test), self.batch_size)
        
        # print data shape
        print(f"""Data Shape: 
        x_train: {x_train.shape}, 
        x_valid: {x_valid.shape}, 
        x_test: {x_test.shape}, 
        y_train: {y_train.shape}, 
        y_valid: {y_valid.shape}, 
        y_test: {y_test.shape},
        src: {self.src.shape}""")
        
    def _get_src(self, x_train: torch.tensor, src_len: int = 0):   
        """get src data for transformer encoder
        """
        
        if self.src_len is None:
            self.src_len = len(x_train)
            
        assert self.src_len <= len(x_train) , "src_len must be less than or equal to train size"
        
        self.src = x_train[:src_len][:, -1, :].unsqueeze(0)
    
    def _get_date(self):
        """
        Get the date for start of train to test end
        """
        
        # remove first window-1 and last one since y shift 1
        self.dates = self.data.index[self.window-1:len(self.data)-1]
        self.train_dates = ...
        self.valid_dates = ...
        self.test_dates = ...
        
    def _window_xy_by_date(self): 
        """Transform to training form of data and get the date of each sample of data
        """
        
        x_list, y_list = [], []
        
        # y shift 1 from x
        for i in range(len(self.data)-self.window): 
            x_window = self.data.iloc[i:i+self.window]
            y_window = self.data.iloc[i+1:i+self.window+1]  
            x_values = x_window[['do', 'dh', 'dl', 'dc', 'dv', 'doc']].values  
            y_values = y_window[['do', 'dh', 'dl', 'dc', 'dv', 'doc']].values
            x_list.append(x_values)
            y_list.append(y_values)
        
        assert len(x_list) == len(self.dates), "Mismatch time index and data"
        
        return x_list, y_list
    
if __name__ == "__main__":
    data = TransformerData(stock="2330.TW")
    print(data.src.shape)
    print(len(data.trainloader))
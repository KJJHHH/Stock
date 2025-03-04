import numpy as np 
import pickle
import torch
from sklearn.preprocessing import StandardScaler
from utils import *
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def data(
    stock_symbol: str = '2454.TW', 
    num_class: int = 2, 
    end_date: str = '2024-12-31',
    batch_size: int = 64,
    window: int = 10
    ):
    
    stock_price_data = fetch_stock_price(stock_symbol=stock_symbol, start_date='2012-01-02',end_date=end_date)

    # pctchange: (today - yesterday)/yesterday
    stock_price_data['do'] = stock_price_data['Open'].pct_change() * 100
    stock_price_data['dh'] = stock_price_data['High'].pct_change() * 100
    stock_price_data['dl'] = stock_price_data['Low'].pct_change() * 100
    stock_price_data['dc'] = stock_price_data['Close'].pct_change() * 100
    stock_price_data['dv'] = stock_price_data['Volume'].pct_change() * 100
    
    # do_1, dc_1, doc_1: tmr's information
    stock_price_data['do_1'] = stock_price_data['do'].shift(-1)
    stock_price_data['dc_1'] = stock_price_data['dc'].shift(-1)
    stock_price_data['doc_1'] = \
        ((stock_price_data['Close'].shift(-1) - stock_price_data['Open'].shift(-1))/stock_price_data['Open'].shift(-1))*100

    stock_price_data = stock_price_data.dropna()
    df = stock_price_data
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()

    df['Close_origin'] = df['Close']
    scaler = StandardScaler()
    scaler.fit(df[['do', 'dh', 'dl', 'dc', 'dv', 'Close']][:2500])
    df[['do', 'dh', 'dl', 'dc', 'dv', 'Close']] = scaler.fit_transform(df[['do', 'dh', 'dl', 'dc', 'dv', 'Close']])


    X, y, date = window_x_y(df, num_class, window)
    
    percentage_test = 0.05
    percentage_valid = 0.05
    X, x_test, y, y_test = train_test(X, y, percentage_test)
    x_train, x_valid, y_train, y_valid = train_valid(X, y, percentage_valid)
    
    src = getSrc(df, num_class, len(x_train))
    
    print(f'x_train_len: {len(x_train)}, valid_len: {len(x_valid)}, test_len: {len(x_test)}')

    trainloader, validloader, testloader = (
        loader(
            torch.tensor(x_train, dtype=torch.float32), 
            torch.tensor(y_train, dtype=torch.float32), 
            batch_size=batch_size), 
        loader(
            torch.tensor(x_valid, dtype=torch.float32), 
            torch.tensor(y_valid, dtype=torch.float32), 
            batch_size=batch_size),
        loader(
            torch.tensor(x_test, dtype=torch.float32), 
            torch.tensor(y_test, dtype=torch.float32), 
            batch_size=batch_size)
    )
    
    # test start date
    test_date = df.index[-len(y_test):]
    
    return trainloader, validloader, testloader, test_date, df, src
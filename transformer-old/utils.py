from tqdm import tqdm
import numpy as np
import yfinance as yf
import torch
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gaf(X):
    X = X.reshape(-1)
    X_diff = X.unsqueeze(0) - X.unsqueeze(1) # Pairwise differences
    # GAF = torch.cos(X_diff)# Gramian Angular Field
    GAF = X_diff

    return GAF

def mask(GAF):    
    mask =  torch.triu(torch.ones(GAF.shape[0], GAF.shape[1])).to(device)
    GAF = GAF * mask
    return GAF

def normalize(x, mean, std):
    return (x - mean) / std

def fetch_stock_price(stock_symbol, start_date, end_date):
    stock = yf.Ticker(stock_symbol)
    stock_data = stock.history(start=start_date, end=end_date)

    return stock_data


def window_x_y(df, num_class, window_size=100): # df: before split
    x1_list, y1_list, date = [], [], []
    for i in range(len(df)-window_size+1): # Create data with window
        window = df.iloc[i:i+window_size]  # Extract the window of data
        x1_values = window[['do', 'dh', 'dl', 'dc', 'dv', 'Close']].T.values  # Adjust column names as needed
        if num_class == 1:
            y1_values = window[['doc_1']].iloc[-1].T.values
        if num_class == 2:
            y1_values = window[['do_1', 'dc_1']].iloc[-1].T.values
        x1_list.append(x1_values)
        y1_list.append(y1_values)
        date.append(window.index[-1])
    x = np.array(x1_list)
    y = np.array(y1_list)
    return x, y, date

def getSrc(df, num_class, src_size = 2000):    
    x, y, date = window_x_y(df, num_class)
    src = x[:src_size]
    return torch.tensor(src).to(dtype=torch.float32)   

def process_x(x):
    X = []
    x = torch.tensor(x, dtype=torch.float32).to(device)
    for i in tqdm(range(len(x))):
        X_element = []
        for j in range(len(x[i])):
            X_element.append(mask(gaf(x[i][j])).unsqueeze(0))
        X_element = torch.cat(X_element, dim=0).unsqueeze(0)
        X.append(X_element)
    X = torch.cat(X, dim=0)
    return X

def train_test(X, y, percentage_test):
    test_size = int(percentage_test * len(X))
    train_size = len(X) - test_size
    x_train = X[:train_size]
    x_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]
    return x_train, x_test, y_train, y_test, 

def train_valid(X, y, percentage_valid):
    valid_size = int(percentage_valid  * len(X))
    train_size = len(X) - valid_size
    x_train = X[:train_size]
    x_valid = X[train_size:]
    y_train = y[:train_size]
    y_valid = y[train_size:]
    return x_train, x_valid, y_train, y_valid

def loader(x, y, batch_size = 16):
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size, shuffle=False, drop_last=True)
    return dataloader

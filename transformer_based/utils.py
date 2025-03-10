from tqdm import tqdm
import numpy as np
import yfinance as yf
import torch

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


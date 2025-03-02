import torch
import torch.nn as nn
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_encoder(model, dataloader):
    
    s_pred = []
    s_true = []
    # print(next(model.parameters()).device)
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x = x.permute(0, 2, 1).to(device)
            y = y.to(device)
            y_pred = model(x)
            s_pred.append(y_pred.cpu().detach())
            s_true.append(y.cpu())
    y_pred_tensor = torch.concat(s_pred)
    y_test_tensor = torch.concat(s_true)
    accuracy = (torch.sign(y_pred_tensor) == torch.sign(y_test_tensor)).sum() / len(y_test_tensor)
    return y_pred_tensor, accuracy

def test_encoderdecoder(model, src, dataloader):    
    """
    Return:
    1. Concatenated y_pred for all batch
    2. Y_pred for last batch
    3. acc
    """
    s_pred = []
    s_true = []
    # print(next(model.parameters()).device)
    for x, y in dataloader:
        batch_size = x.shape[0]
        break
    with torch.no_grad():
        src = src.permute(1, 0, 2)
        for x, y in tqdm(dataloader):
            if x.size(0) != batch_size:
                continue
            x = x.permute(0, 2, 1).to(device)
            y = y.to(device)
            memory, y_pred = model(src, x, True)
            s_pred.append(y_pred.cpu().detach())
            s_true.append(y.cpu())
    y_pred_tensor = torch.concat(s_pred)
    y_test_tensor = torch.concat(s_true)
    accuracy = (torch.sign(y_pred_tensor) == torch.sign(y_test_tensor)).sum() / len(y_test_tensor)
    return y_pred_tensor, y_pred, accuracy

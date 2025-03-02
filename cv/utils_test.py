import torch
import torch.nn as nn
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(model, dataloader):
    s_pred = []
    s_true = []
    # print(next(model.parameters()).device)
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            s_pred.append(y_pred.cpu().detach())
            s_true.append(y.cpu())
    y_pred_tensor = torch.concat(s_pred)
    y_test_tensor = torch.concat(s_true)
    accuracy = (torch.sign(y_pred_tensor) == torch.sign(y_test_tensor)).sum() / len(y_test_tensor)
    return y_pred_tensor, accuracy

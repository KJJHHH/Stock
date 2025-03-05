import sys 
sys.path.append('../')
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import *
from datas import *
from Transformers import *
from accelerate import Accelerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

stock = "2884.TW"
MODEL = "Transformer"

num_epochs = 200
lr = 0.0001

corp = Data(stock)
corp.prepareData()

accelerator = Accelerator(mixed_precision='fp16')
device = accelerator.device

model = Transformer()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=len(corp.trainloader)*1, gamma=0.9)     

(
model, 
optimizer, 
corp.trainloader, 
corp.validloader, 
scheduler,
) = accelerator.prepare(model, optimizer, corp.trainloader, corp.validloader, scheduler)

print(next(model.parameters()).device)

last_epoch = 0
not_improve_cnt = 0
min_val_loss = float("inf")

corp.src = corp.src.to(device)

for epoch in range(last_epoch, num_epochs):
    
    # Training
    model.train()
    loss_train_mean = 0
    for x, y in tqdm(corp.trainloader): 
        optimizer.zero_grad()       
        memory, outputs = model(src=corp.src, tgt=x, train=True)    
        loss = criterion(outputs, y)
        accelerator.backward(loss)
        optimizer.step()
        loss_train_mean += loss.item()
    
    # Train loss
    loss_train_mean /= len(corp.trainloader)
    
    # Scheduler 
    if epoch > 50:
        scheduler.step()
        
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f'result/ckpt-{MODEL}_{stock}_{epoch}.pt')
    
    # Validating
    loss_valid_mean = 0
    with torch.no_grad():
        model.eval()
        for x_val, y_val in corp.validloader:
            
            _, outputs_val = model(tgt=x_val, train=False, memory=memory)
            loss = criterion(outputs_val, y_val)
            loss_valid_mean += loss.item()
            
        loss_valid_mean /= len(corp.validloader)
            
        if loss_valid_mean < min_val_loss:
            min_val_loss = loss_valid_mean
            not_improve_cnt = 0
            print(f'New best model found in epoch {epoch} with val loss: {min_val_loss}')
            torch.save(model.state_dict(), f'result/{MODEL}_{stock}_best.pt')  
        else:
            not_improve_cnt += 1
            if not_improve_cnt >= 50:
                print(f'Early stopping at epoch {epoch}')
                break
            
    print(f"Epoch {epoch} | training loss: {loss_train_mean:.3f} evaluating loss: {loss_valid_mean:.3f}")
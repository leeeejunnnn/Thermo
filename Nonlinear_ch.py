#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

from data_pipeline import data_pipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())
print(torch.cuda.get_device_name(device))
#%% DataLoader


#%% Model
model = nn.Sequential(
    nn.Linear(1200,600),
    nn.LeakyReLU(0.2),
    nn.Linear(600,200),
    nn.LeakyReLU(0.2),
    nn.Linear(200,100),
    nn.LeakyReLU(0.2),    
    nn.Linear(100,50),
    nn.LeakyReLU(0.2),        
    nn.Linear(50,25),
    nn.LeakyReLU(0.2),          
    nn.Linear(25,1),
)

#%%
gpu = torch.device('cuda')
loss_func = nn.L1Loss().to(gpu)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
model = model.to(gpu)
x = x.to(gpu)
y_noise = y_noise.to(gpu)
num_epoch = 200
loss_array = []
for epoch in range(num_epoch):
    optimizer.zero_grad()
    output = model(x)
    
    loss = loss_func(output,y_noise)
    loss.backward()
    optimizer.step()
    
    loss_array.append(loss)
    if epoch % 100 == 0:
        print('epoch:', epoch, ' loss:', loss.item())

#%%
plt.plot(loss_array)
plt.show()
plt.figure(figsize=(10,10))
x = x.cpu().detach().numpy()
y_noise = y_noise.cpu().detach().numpy()
output = output.cpu().detach().numpy()
plt.scatter(x, y_noise, s=1, c="gray")
plt.scatter(x, output, s=1, c="red")
plt.show()
# %%

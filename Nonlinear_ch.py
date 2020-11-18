#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import os
from glob import glob
import time
import numpy as np
import matplotlib.pyplot as plt

from data_pipeline import data_pipeline

#%%
# device GPU / CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())
print(torch.cuda.get_device_name(device))

# Data parameters 
Data_dir = '/home/sss-linux1/project/leejun/Thermo/Experiment/'

# NN training parameters
TENSORBOARD_STATE = True
num_epoch = 400
BATCH_SIZE = 64
val_ratio = 0.3
Learning_rate = 0.001
L2_decay = 1e-8
LRSTEP = 5
GAMMA = 0.1

#%% DataLoader
dataset = data_pipeline(Data_dir)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, (320000, 12800) )
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, shuffle=False, num_workers=0)
#%% Model
model = nn.Sequential(
    nn.Linear(1200,600),
    nn.BatchNorm1d(600),
    nn.LeakyReLU(0.2),
    nn.Linear(600,200),
    nn.BatchNorm1d(200),
    nn.LeakyReLU(0.2),
    nn.Linear(200,100),
    nn.BatchNorm1d(100),
    nn.LeakyReLU(0.2),    
    nn.Linear(100,50),
    nn.BatchNorm1d(50),
    nn.LeakyReLU(0.2),        
    nn.Linear(50,25),
    nn.BatchNorm1d(25),
    nn.LeakyReLU(0.2),          
    nn.Linear(25,1),
)

#%%
model = model.to(device)
loss_func = nn.L1Loss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate, weight_decay=L2_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=LRSTEP, gamma=GAMMA)

#%%
ckpt_dir = './Checkpoint'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
ckpt_path = '%s%s.pt' % (ckpt_dir, '/Checkpoint_exp')


'''#%%
if TENSORBOARD_STATE:
    summary = SummaryWriter()

it = 0
train_losses = []
validation_losses = []
best_validation_acc = 0

for epoch in range(num_epoch):      
    # Train
    model.train()
    one_epoch_start = time.time()
    print('# Epoch {}, Learning Rate: {:.0e}'.format(epoch,scheduler.get_lr()[0]))        
    for x, target in train_loader:
        it += 1

        # Inputs to device
        x = x.to(device, dtype=torch.float)
        target = target.to(device)

        # Feed data into the network and get outputs
        output = model(x)

        # Compute loss
        loss = loss_func(output,target)

        # Back propagtion
        loss.backward()

        # Update optimizer
        optimizer.step()

        if it == 1:
            one_iter_elapsed = time.time()-one_epoch_start
            print('Iter:{} / Train loss: {:.4f}'.format(it, loss.item()))
        if it % 500 == 0:
            print('Iter:{} / Train loss: {:.4f}'.format(it, loss.item()))
            if TENSORBOARD_STATE:
                summary.add_scalar('loss/train_loss', loss.item(), it)
    train_losses.append(loss.item())
    
    # Update learning rate
    scheduler.step()

    # Validation
    model.eval()
    n = 0.
    validation_loss = 0.
    validation_acc = 0.

    for x_val, target_val in val_loader:
        x_val = x_val.to(device, dtype=torch.float)
        target_val = target_val.to(device)

        logits_val = model(x_val)
        validation_loss += F.cross_entropy(logits_val,target_val).item()
        validation_acc += (logits_val.argmax(dim=1) == target_val).float().sum().item()
        n += x_val.size(0)

    validation_loss /= n
    validation_acc /= n
    print('Validation loss: {:.4f}, Validation accuracy: {:.4f}'.format(validation_loss, validation_acc))
    if TENSORBOARD_STATE:
        summary.add_scalar('loss/validation_loss',validation_loss, it)
    validation_losses.append(validation_loss)

    if validation_acc > best_validation_acc:
        best_validation_acc = validation_acc
        ckpt = {'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_validation_acc':best_validation_acc}
        torch.save(ckpt,ckpt_path)
        print('Higher validation accuracy, Checkpoint Saved!')
    
    curr_time = time.time()
    print("one epoch time = %.2f" %(curr_time-one_epoch_start))
    print('########################################################')

if TENSORBOARD_STATE:
    summary.close()

plt.plot(train_losses, label='train loss')
plt.plot(validation_losses, label='validation loss')
plt.legend()

'''
#%%
loss_array = []
for epoch in range(num_epoch):
    for x, target in train_loader:
        optimizer.zero_grad()
        x = x.to(device, dtype=torch.float)
        target = target.to(device)
        output = model(x)

        loss = loss_func(output,target)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print('epoch:', epoch, ' loss:', loss.item())
        loss_array.append(loss)
    np.save('/home/sss-linux1/project/leejun/Thermo/loss.npy',loss_array)    
    
    scheduler.step()
    
ckpt = {'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        }
torch.save(ckpt,ckpt_path)
print('Higher validation accuracy, Checkpoint Saved!')

plt.plot(loss_array, label='train loss')
plt.legend()
plt.show()

#%%

plt.plot(loss_array)
plt.show()
plt.figure(figsize=(10,10))
x = x.cpu().detach().numpy()
y_noise = y_noise.cpu().detach().numpy()
output = output.cpu().detach().numpy()
plt.scatter(x, y_noise, s=1, c="gr  ay")
plt.scatter(x, output, s=1, c="red")
plt.show()
# %% Validataion
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
ckept_load = torch.load('Checkpoint/Checkpoint_exp_400_0015.pt', map_location=device)
model.load_state_dict(ckept_load['model'])
loss_func = nn.L1Loss().to(device)
#%%
output_array = []
loss_array = []
target_array = []
for x, target in val_loader:
    x = x.to(device, dtype=torch.float)
    target = target.to(device)
    target_array.append(target.cpu().data.numpy())
    output = model(x)

    loss = output-target
    output_array.append(output.cpu().data.numpy())
    loss_array.append(loss.cpu().data.numpy())
#%%
output_array = np.vstack(output_array)
loss_array = np.vstack(loss_array)
target_array = np.vstack(target_array)
plt.scatter(output_array, target_array)
plt.show()
plt.plot(loss_array)
plt.show()
plt.plot(sorted(loss_array))
plt.show()
print(np.mean(abs(loss_array)))
# %%

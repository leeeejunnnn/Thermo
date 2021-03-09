#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

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
#print ('Current cuda device ', torch.cuda.current_device())
#print(torch.cuda.get_device_name(device))

# Data parameters 
Data_dir = '/home/sss-linux1/project/leejun/Thermo/Experiment/'

# NN training parameters
TENSORBOARD_STATE = True
num_epoch = 100
BATCH_SIZE = 32
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
#%%
import os
from glob import glob

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

# %%
data_dir = './Experiment/'

#%%
class data_pipeline(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.num_data = []
        self.data = []
        data_list = sorted(glob(self.data_dir + '*time*.npy'))
        target_list = sorted(glob(self.data_dir + '*ref*.npy'))
        for i in range(len(data_list)):
            data_a = np.load(data_list[i])
            target_a = np.load(target_list[i])
            
            
            
        data_a = np.load(a[1])
        self.num_data.append(data_a.shape[0])
        self.data = data_a

    def __len__(self):
        length = len(self.data) - self.num_dataset
        return length

    def __getitem__(self, idx):
        target = 0
        x = self.data[idx:(idx + self.num_dataset),:]
        r = np.zeros((14,14))
        for i in range(14):
            for j in range(14):
                r[i,j] = np.mean(x[:,i] * x[:,j])
        r = (r - r.min()) / (r.max()-r.min())

        return r, target

# %%


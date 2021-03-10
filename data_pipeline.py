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
        self.num_target = []
        self.data = []
        self.target = []
        data_list = sorted(glob(self.data_dir + '*time*_1.npy'))
        target_list = sorted(glob(self.data_dir + '*ref*.npy'))
        for i in range(len(data_list)):
            data_a = np.load(data_list[i], allow_pickle=True)
            data_a = data_a.reshape(-1,600)
            target_a = np.load(target_list[i], allow_pickle=True)
            target_a = target_a.reshape(-1,1)
            print(np.mean(target_a))
            self.num_data.append(data_a.shape[0])
            self.num_target.append(target_a.shape[0])
            self.data.append(data_a)
            self.target.append(target_a)
        self.data = np.vstack(self.data)
        self.target = np.vstack(self.target)

    def __len__(self):
        return (self.data.shape[0])

    def __getitem__(self, idx):
        target = self.target[idx,:]
        x = self.data[idx,:]
        return x, target



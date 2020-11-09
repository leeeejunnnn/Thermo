#%%
import os
from glob import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

#%%
filepath = 'Experiment/20190225_drycoating/image_raw_sp25_thk100_registered.mat'
arrays = {}
f = h5py.File(filepath)
for k, v in f.items():
    arrays[k] = np.array(v)
image_timeser_100um = np.array(arrays['image_raw'])
np.save('image_time_0225_100um', arr=image_timeser_100um)
#%%
filepath = 'Experiment/20190225_drycoating/image_raw_sp25_thk200_registered.mat'
arrays = {}
f = h5py.File(filepath)
for k, v in f.items():
    arrays[k] = np.array(v)
image_timeser_200um = np.array(arrays['image_raw'])
np.save('image_time_0225_200um', arr=image_timeser_200um)

filepath = 'Experiment/20190225_drycoating/image_raw_sp25_thk300_registered.mat'
arrays = {}
f = h5py.File(filepath)
for k, v in f.items():
    arrays[k] = np.array(v)
image_timeser_300um = np.array(arrays['image_raw'])
np.save('image_time_0225_300um', arr=image_timeser_300um)

filepath = 'Experiment/20190225_drycoating/image_raw_sp25_thk400_registered.mat'
arrays = {}
f = h5py.File(filepath)
for k, v in f.items():
    arrays[k] = np.array(v)
image_timeser_400um = np.array(arrays['image_raw'])
np.save('image_time_0225_400um', arr=image_timeser_400um)

filepath = 'Experiment/20190225_drycoating/image_raw_sp25_thk500_registered.mat'
arrays = {}
f = h5py.File(filepath)
for k, v in f.items():
    arrays[k] = np.array(v)
image_timeser_500um = np.array(arrays['image_raw'])
np.save('image_time_0225_500um', arr=image_timeser_500um)

filepath = 'Experiment/20190225_drycoating/image_raw_sp25_thk600_registered.mat'
arrays = {}
f = h5py.File(filepath)
for k, v in f.items():
    arrays[k] = np.array(v)
image_timeser_600um = np.array(arrays['image_raw'])
np.save('image_time_0225_600um', arr=image_timeser_600um)

filepath = 'Experiment/20190225_drycoating/image_raw_sp25_thk700_registered.mat'
arrays = {}
f = h5py.File(filepath)
for k, v in f.items():
    arrays[k] = np.array(v)
image_timeser_700um = np.array(arrays['image_raw'])
np.save('image_time_0225_700um', arr=image_timeser_700um)

filepath = 'Experiment/20190225_drycoating/image_raw_sp25_thk800_registered.mat'
arrays = {}
f = h5py.File(filepath)
for k, v in f.items():
    arrays[k] = np.array(v)
image_timeser_800um = np.array(arrays['image_raw'])
np.save('image_time_0225_800um', arr=image_timeser_800um)

#%%

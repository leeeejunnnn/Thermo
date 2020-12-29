#%%
import os
from glob import glob
import h5py
from scipy import io
import numpy as np
import matplotlib.pyplot as plt

#%%raw data
filepath = '/home/sss-linux1/project/leejun/Thermo/Experiment/20190225_drycoating/image_raw_sp25_thk100_registered_1.mat'
matfile = io.loadmat(filepath)
image_timeser_100um = np.array(matfile['image_raw_registered1'])
np.save('image_time_0225_100um_1', arr=image_timeser_100um)

filepath = 'Experiment/20190225_drycoating/image_raw_sp25_thk200_registered_1.mat'
matfile = io.loadmat(filepath)
image_timeser_200um = np.array(matfile['image_raw_registered1'])
np.save('image_time_0225_200um_1', arr=image_timeser_200um)

filepath = 'Experiment/20190225_drycoating/image_raw_sp25_thk300_registered_1.mat'
matfile = io.loadmat(filepath)
image_timeser_300um = np.array(matfile['image_raw_registered1'])
np.save('image_time_0225_300um_1', arr=image_timeser_300um)

filepath = 'Experiment/20190225_drycoating/image_raw_sp25_thk400_registered_1.mat'
matfile = io.loadmat(filepath)
image_timeser_400um = np.array(matfile['image_raw_registered1'])
np.save('image_time_0225_400um_1', arr=image_timeser_400um)

filepath = 'Experiment/20190225_drycoating/image_raw_sp25_thk500_registered_1.mat'
matfile = io.loadmat(filepath)
image_timeser_500um = np.array(matfile['image_raw_registered1'])
np.save('image_time_0225_500um_1', arr=image_timeser_500um)

filepath = 'Experiment/20190225_drycoating/image_raw_sp25_thk600_registered_1.mat'
matfile = io.loadmat(filepath)
image_timeser_600um = np.array(matfile['image_raw_registered1'])
np.save('image_time_0225_600um_1', arr=image_timeser_600um)

filepath = 'Experiment/20190225_drycoating/image_raw_sp25_thk700_registered_1.mat'
matfile = io.loadmat(filepath)
image_timeser_700um = np.array(matfile['image_raw_registered1'])
np.save('image_time_0225_700um_1', arr=image_timeser_700um)

filepath = 'Experiment/20190225_drycoating/image_raw_sp25_thk800_registered_1.mat'
matfile = io.loadmat(filepath)
image_timeser_800um = np.array(matfile['image_raw_registered1'])
np.save('image_time_0225_800um_1', arr=image_timeser_800um)

#%%refference
filepath = 'Experiment/reference/New/image_ref_sp25_thk100_registered.mat'
matfile = io.loadmat(filepath)
image_timeser_100um = np.array(matfile['image_ref_registered'])
np.save('image_ref_0225_100um', arr=image_timeser_100um)

filepath = 'Experiment/reference/New/image_ref_sp25_thk200_registered.mat'
matfile = io.loadmat(filepath)
image_timeser_200um = np.array(matfile['image_ref_registered'])
np.save('image_ref_0225_200um', arr=image_timeser_200um)

filepath = 'Experiment/reference/New/image_ref_sp25_thk300_registered.mat'
matfile = io.loadmat(filepath)
image_timeser_300um = np.array(matfile['image_ref_registered'])
np.save('image_ref_0225_300um', arr=image_timeser_300um)

filepath = 'Experiment/reference/New/image_ref_sp25_thk400_registered.mat'
matfile = io.loadmat(filepath)
image_timeser_400um = np.array(matfile['image_ref_registered'])
np.save('image_ref_0225_400um', arr=image_timeser_400um)

filepath = 'Experiment/reference/New/image_ref_sp25_thk500_registered.mat'
matfile = io.loadmat(filepath)
image_timeser_500um = np.array(matfile['image_ref_registered'])
np.save('image_ref_0225_500um', arr=image_timeser_500um)

filepath = 'Experiment/reference/New/image_ref_sp25_thk600_registered.mat'
matfile = io.loadmat(filepath)
image_timeser_600um = np.array(matfile['image_ref_registered'])
np.save('image_ref_0225_600um', arr=image_timeser_600um)

filepath = 'Experiment/reference/New/image_ref_sp25_thk700_registered.mat'
matfile = io.loadmat(filepath)
image_timeser_700um = np.array(matfile['image_ref_registered'])
np.save('image_ref_0225_700um', arr=image_timeser_700um)

filepath = 'Experiment/reference/New/image_ref_sp25_thk800_registered.mat'
matfile = io.loadmat(filepath)
image_timeser_800um = np.array(matfile['image_ref_registered'])
np.save('image_ref_0225_800um', arr=image_timeser_800um)

# %%

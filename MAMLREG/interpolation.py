#%%
from scipy.interpolate import interp2d
import numpy as np


#%%
A = np.array([[ 0.45717218,  0.44250104,  0.47812272,  0.49092173,  0.46002069],
   [ 0.29829681,  0.26408021,  0.3709202 ,  0.44823109,  0.49311853],
   [ 0.05469835,  0.01048596,  0.17398291,  0.30088943,  0.39783137],
   [-0.20463768, -0.24610673, -0.0713164 ,  0.08406331,  0.22047102],
   [-0.4074527 , -0.43573695, -0.31062521, -0.15750053, -0.00222392]])
# %%
f = interp2d(np.arange(0,500,100), np.arange(0,500,100), A)
z = f(np.arange(500), np.arange(500))
# %%
import matplotlib.pyplot as plt
plt.imshow(z)
plt.show()

#%%
plt.imshow(A)
plt.show()
# %%
plt.plot(z[-1])

# %%
plt.plot(A[-1])

#%%
red_40 = np.load('../Experiment/210107/10sec/rered_40_80_ext_10.npy')
print(red_40.shape)
red_120 = np.load('../Experiment/210107/10sec/rered_120_160_ext_10.npy')
print(red_120.shape)
#%%
red = np.append(red_40, red_120, axis=2)
red = np.transpose(red, (1,2,0))
print(red.shape)
np.save('../Experiment/210107/10sec/exp_red',red)
print('saved!')
#%%
plt.imshow(red[:, :, 252])
plt.show()
plt.plot(red[200, 900, :])
plt.show()


# %%
red = np.load('../Experiment/210107/lightgreen_40_80_ext_5.npy')
print(red.shape)
plt.imshow(red[:,:,270])
plt.show()
# %%

import pandas as pd

ref_red = pd.read_excel('../Experiment/210107/ref_red.xlsx')
#%%

print(ref_red.shape)
plt.imshow(ref_red)
plt.show()
# %%

#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt

#%%
num_data = 5000
noise = init.normal_(torch.FloatTensor(num_data,1), std=30)
x = init.uniform_(torch.Tensor(num_data,1),-10,10)
def func(x): return 0.5*(x**3) - 0.5*(x**2) - torch.sin(2*x)*90 + 1 
y_noise = func(x) + noise

#%%
model = nn.Sequential(
    nn.Linear(1,5),
    nn.LeakyReLU(0.2),
    nn.Linear(5,10),
    nn.LeakyReLU(0.2),
    nn.Linear(10,10),
    nn.LeakyReLU(0.2),    
    nn.Linear(10,10),
    nn.LeakyReLU(0.2),        
    nn.Linear(10,5),
    nn.LeakyReLU(0.2),          
    nn.Linear(5,1),
)

#%%
gpu = torch.device('cuda')
loss_func = nn.L1Loss().to(gpu)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
model = model.to(gpu)
x = x.to(gpu)
y_noise = y_noise.to(gpu)
num_epoch = 20000
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

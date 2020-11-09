#%%
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

# %%

cuda = torch.device("cuda")
torch.manual_seed(1)

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]]).to(device=cuda)
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]]).to(device=cuda)

# %%
model = nn.Linear(3,1).cuda()
# %%
print(list(model.parameters()))
# %%
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
# %%
nb_epochs = 200000
for epoch in range(nb_epochs+1):

    # H(x) 계산
    prediction = model(x_train)
    # model(x_train)은 model.forward(x_train)와 동일함.

    # cost 계산
    cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산
    cost.backward()
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 10000 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))
# %%
new_var = torch.FloatTensor([[73, 80, 75]])
pred_y = model(new_var).cuda()
print(pred_y)
# %%
print(list(model.parameters()))
# %%

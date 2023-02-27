import torch
from torch import nn
from torch import optim
from torch.utils import data
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.fc = nn.Linear(2, 1)
        pass

    def forward(self, x):
        x = F.relu(self.fc(x))
        return x


def load_array(data_arrays, batch_size, is_train=True):  # @save
    """构造⼀个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# 生成数据
# 这里是随机生成的，所以数据并不一定是线性可分的
features = torch.randn(1000, 2)
labels = torch.randn(1000, 1)

model = Linear()
loss = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

dataloader = load_array((features, labels), 10)

N_EPOCHS = 10
for epoch in range(N_EPOCHS):
    epoch_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        train_loss = loss(outputs, labels)
        train_loss.backward()
        optimizer.step()
        epoch_loss += train_loss
    print("Epoch: {} Loss: {}"
          .format(epoch, epoch_loss / len(dataloader)))

print(model(torch.tensor([100., 2.])))

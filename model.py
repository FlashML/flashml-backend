import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv2D_0 = nn.Conv2d(3, 6, 5)
        self.relu_0 = F.relu
        self.pool_0 = nn.MaxPool2d(2, 2)
        self.conv2D_1 = nn.Conv2d(6, 16, 5)
        self.relu_1 = F.relu
        self.pool_1 = nn.MaxPool2d(2, 2)
        self.linear_0 = nn.Linear(400, 120)
        self.relu_2 = F.relu
        self.linear_1 = nn.Linear(120, 84)
        self.relu_3 = F.relu
        self.linear_2 = nn.Linear(84, 10)


    def forward(self, x):
        x = self.conv2D_0(x)
        x = self.relu_0(x)
        x = self.pool_0(x)
        x = self.conv2D_1(x)
        x = self.relu_1(x)
        x = self.pool_1(x)
        x = x = x.view(-1, 400)
        x = self.linear_0(x)
        x = self.relu_2(x)
        x = self.linear_1(x)
        x = self.relu_3(x)
        x = self.linear_2(x)
        return x


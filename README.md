# flash_ml
```
from flash_ml/build_model import build model

example_input = [["input", 32, 32, 3], ["conv2d", 6, 5], ["relu"], ["maxpool2d", 2, 2],
            ["conv2d", 16, 5], ["relu"], ["maxpool2d", 2, 2], ["dense", 120],
            ["relu"], ["dense", 84], ["relu"], ["dense", 10]]
            
build_model(example_input)
```

example_output

<i> model.py </i>
```
    1 import torch
    2 import torch.nn as nn
    3
    4 class Net(nn.module):
    5    def __init__(self):
    6        super(Net, self).__init__()
    7       self.conv2D_0 = nn.Conv2d(3, 6, 5)
    8       self.relu_0 = F.relu
    9       self.pool_0 = nn.MaxPool2d(2, 2)
   10       self.conv2D_1 = nn.Conv2d(3, 16, 5)
   11       self.relu_1 = F.relu
   12       self.pool_1 = nn.MaxPool2d(2, 2)
   13       self.linear_0 = nn.Linear(3, 120)
   14       self.relu_2 = F.relu
   15       self.linear_1 = nn.Linear(3, 84)
   16       self.relu_3 = F.relu
   17       self.linear_2 = nn.Linear(3, 10)
   18
   19
   20    def forward(self, x):
   21       x = self.conv2D_0(x)
   22       x = self.relu_0(x)
   23       x = self.pool_0(x)
   24       x = self.conv2D_1(x)
   25       x = self.relu_1(x)
   26       x = self.pool_1(x)
   27       x = self.linear_0(x)
   28       x = self.relu_2(x)
   29       x = self.linear_1(x)
   30       x = self.relu_3(x)
   31       x = self.linear_2(x)
   32        return x
```


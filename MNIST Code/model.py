from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn

class ConvNet(nn.Module):
    
    def __init__(self):
        super(ConvNet, self).__init__()

        self.activation = nn.ReLU()
        self.dp0 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(784, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.dp1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.dp2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(4096, 4096)
        self.bn3 = nn.BatchNorm1d(4096)
        self.dp3 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(4096, 10)
        self.bn4 = nn.BatchNorm1d(10)

    def forward(self, x):
        x = self.dp0(x.view(-1, 784))
        x = self.dp1(self.activation(self.bn1(self.fc1(x))))
        x = self.dp2(self.activation(self.bn2(self.fc2(x))))
        x = self.dp3(self.activation(self.bn3(self.fc3(x))))
        x = self.bn4(self.fc4(x))
        
        return x
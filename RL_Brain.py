import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


#residualBlock
class ResidualBlock(nn.Module):
    def __init__(self, state):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(state, 256, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256,256,kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(256)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out
    
#Policy Network
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim = 128):
        super(PolicyNet,self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)
    


import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

#############################
#Build policy net work for actor
#############################
class PolicyNet(nn.Module):
    def __init__(self, inchannels, height, width, action_dim, hidden_dim=128):
        super(PolicyNet, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ) for _ in range(8)
        ])
        self.fc1 = nn.Linear(256 * height * width, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x
    
#############################
#Build value net work for critic
#############################
class ValueNet(nn.Module):
    def __init__(self, inchannels, height, width, hidden_dim):
        super(ValueNet, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ) for _ in range(8)
        ])
        self.fc1 = nn.Linear(256 * height * width, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class PPO:
    def __init__(self, in_channels, height, width, n_hiddens, n_actions,
                 actor_lr, critic_lr, gamma, lmbda,epochs,eps,device):
        self.actor = PolicyNet(inchannels=in_channels, height=height,width=width, action_dim=n_actions,hidden_dim= n_hiddens).to(device=device)
        self.critic = ValueNet(inchannels=in_channels, height=height,width=width,hidden_dim= n_hiddens).to(device=device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        state = torch.tensor(state[np.newaxis, :]).to(self.device)
        probs = self.actor(state)
        action_list = torch.distributions.Categorical(probs)
        action = action_list.sample().item()
        return action
    
    def learn(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).to(self.device).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
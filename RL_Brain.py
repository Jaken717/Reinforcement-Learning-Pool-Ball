import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# #residualBlock
# class ResidualBlock(nn.Module):
#     def __init__(self, state):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(state, 256, kernel_size = 3, padding = 1)
#         self.bn1 = nn.BatchNorm2d(256)
#         self.conv2 = nn.Conv2d(256,256,kernel_size = 3, padding = 1)
#         self.bn2 = nn.BatchNorm2d(256)

#     def forward(self, x):
#         residual = x
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += residual
#         out = F.relu(out)
#         return out
    
#Policy Network
class PolicyNet(nn.Module):
    def __init__(self, inchannels,height,width, action_dim, hidden_dim = 128): #state_dim == ResNet
        super(PolicyNet,self).__init__()
        self.conv1 = nn.Conv2d(inchannels, 256, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ) for _ in range(8)
        ])
        self.fc1 = nn.Linear(256 * height * width, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,action_dim)

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim = 1)
        return x
    
#Value Network
class ValueNet(nn.Module):
    def __init__(self,inchannels,height,width,hidden_dim):
        super(ValueNet, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, 256, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ) for _ in range(8)
        ])
        self.fc1  = nn.Liear(255*height*width, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, 1)
        
    def forward(self, x,a):
        x = F.relu(self.bn1(self.conv1(x)))
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class ActorCritic:
    def __init__(self, in_channels, height, width, n_hiddens, n_actions,
                 actor_lr, critic_lr, gamma, device):
        self.gamma = gamma
        self.device = device

        self.actor = PolicyNet(in_channels, height, width, n_hiddens, n_actions).to(device)
        self.critic = ValueNet(in_channels, height, width, n_hiddens).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    def take_action(self, state):
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = int(action_dist.sample().item())
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(self.device)

        td_value = self.critic(states)
        td_target = rewards + self.gamma * self.critic(next_states) * (1-dones)
        td_delta = td_target - td_value
        
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
# #Actor-critic
# class ActorCritic:
#     def __init__(self, state_dim, hidden_dim,action_dim, actor_lr,critic_lr, gamma):
#         self.gamma = gamma

#         #build policy network
#         self.actor = PolicyNet(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim)

#         #build value network
#         self.critic = ValueNet(state_dim=state_dim, hidden_dim=hidden_dim)

#         #optimize policy network
#         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = actor_lr)

#         #optimize value network
#         self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

#     #select action
#     def take_action(self,state):
#         #change dimision numpy[n_states] -> [1,n_states] -> tensor
#         state = torch.tensor(state[np.newais, :])
#         #action value function, the probability of actions for current state
#         probs = self.actor(state)
#         #create a data distribution base on probs
#         action_dist = torch.distributions.Categorical(probs)
#         action = int(action_dist.sample().item())
#         return action
    
#     #update the module
#     def update(self, transition_dict):
#         #training set
#         states = torch.tensor(transition_dict['states'], dtype = torch.float)
#         actions = torch.tensor(transition_dict['actions']).view(-1, 1)
#         rewards = torch.tensor(transition_dict['rewards'], dtype = torch.float).view(-1, 1)
#         next_states = torch.tensor(transition_dict['next_states'], dtype = torch.float)
#         dones = torch.tensor(transition_dict['dones'], dtype = torch.float).view(-1,1)

#         #predict the value of current state
#         td_value = self.critic(states)
#         #current state_value
#         td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
#         #calculate the difference between state_value and predicted value
#         td_delta = td_target - td_value

#         #log function for all the action value to corresponding state
#         log_probs = torch.log(self.actor(states).gather(1,actions))
#         #policy gradient loss
#         actor_loss = torch.mean(-log_probs * td_delta.detach())
#         #value function loss, between predicted value and target value
#         critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

#         #assign zero to optimizers
#         self.actor_optimizer.zero_grad()
#         self.critic_optimizer.zero_grad()

#         #spread backward
#         actor_loss.backward()
#         critic_loss.backward()

#         #update parameters
#         self.actor_optimizer.step()
#         self.critic_optimizer.step()
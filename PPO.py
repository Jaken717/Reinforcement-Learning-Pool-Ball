import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

#############################
#Build policy net work for actor
#############################
class Targetball_PolicyNet(nn.Module):
    def __init__(self, inchannels, height, width, action_dim, hidden_dim=128):
        super(Targetball_PolicyNet, self).__init__()
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
    

class Targetpocket_PolicyNet(nn.Module):
    def __init__(self, inchannels, height, width, action_dim, hidden_dim=128):
        super(Targetpocket_PolicyNet, self).__init__()
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
    
class Angle_PolicyNet(nn.Module):
    def __init__(self, inchannels, height, width, action_dim, hidden_dim=128):
        super(Angle_PolicyNet, self).__init__()
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
    
class Power_PolicyNet(nn.Module):
    def __init__(self, inchannels, height, width, action_dim, hidden_dim=128):
        super(Power_PolicyNet, self).__init__()
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

class poolaction:
    def __init__(self, target_ball, target_hole, angle, power):
        self.target_ball = target_ball
        self.target_hole = target_hole
        self.angle = angle
        self.power = power


class PPO:
    def __init__(self, in_channels, height, width, n_hiddens, n_actions,
                 actor_lr, critic_lr, gamma, lmbda,epochs,eps,device):
        self.actor_ball = Targetball_PolicyNet(inchannels=in_channels, height=height,width=width, action_dim=n_actions,hidden_dim= n_hiddens).to(device=device)
        self.actor_pocket = Targetpocket_PolicyNet(inchannels=in_channels, height=height,width=width, action_dim=n_actions,hidden_dim= n_hiddens).to(device=device)
        self.actor_angle = Angle_PolicyNet(inchannels=in_channels, height=height,width=width, action_dim=n_actions,hidden_dim= n_hiddens).to(device=device)
        self.actor_power = Power_PolicyNet(inchannels=in_channels, height=height,width=width, action_dim=n_actions,hidden_dim= n_hiddens).to(device=device) 
        self.critic = ValueNet(inchannels=in_channels, height=height,width=width,hidden_dim= n_hiddens).to(device=device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_ball(self, state):
        state = torch.tensor(state[np.newaxis, :]).to(self.device)
        probs = self.actor_ball(state)
        action_list = torch.distributions.Categorical(probs)
        action_ball = action_list.sample().item()
        return action_ball
    
    def take_pocket(self, state):
        state = torch.tensor(state[np.newaxis, :]).to(self.device)
        probs = self.actor_pocket(state)
        action_list = torch.distributions.Categorical(probs)
        action_pocket = action_list.sample().item()
        return action_pocket
    
    def take_angle(self, state):
        state = torch.tensor(state[np.newaxis, :]).to(self.device)
        probs = self.actor_angle(state)
        action_list = torch.distributions.Categorical(probs)
        action_angle = action_list.sample().item()
        return action_angle
    
    def take_power(self, state):
        state = torch.tensor(state[np.newaxis, :]).to(self.device)
        probs = self.actor_power(state)
        action_list = torch.distributions.Categorical(probs)
        action_power = action_list.sample().item()
        return action_power
    
    def take_action(self,state):
        action_ball = self.take_ball(state)
        action_pocket = self.take_pocket(state)
        action_angle = self.take_angle(state)
        action_power = self.take_power(state)

        angle = {0: -3, 1: -2, 2: -1, 3: 0, 4: 1, 5: 2, 6: 3}.get(action_angle)
        power = {0: 3, 1: 5, 2: 7}.get(action_power)

        action = poolaction(action_ball + 1, action_pocket, angle, power)

        return action
    
    def learn(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).to(self.device).view(-1, 1)
        #solve by using action sets
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.device).view(-1, 1)

        next_q_target = self.critic(next_states)
        td_target = rewards + self.gamma * next_q_target * (1 - dones)
        td_value = self.critic(states)
        td_delta = td_target - td_value

        td_delta = td_delta.cpu().detach().numpy()
        advantage = 0
        advantage_list = []

        for delta in td_delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)

        advantage_list.reverse()
        #numpy --> tensor [b,1]
        advantage = torch.tensor(advantage_list, dtype=torch.float).to(self.device)

        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1,actions))
            #compare the old and new policy
            ratio = torch.exp(log_probs - old_log_probs)
            #optimize the left 
            surr1 = ratio * advantage
            #optimize right, if ratio < 1 - eps output 1-eps, if ratio > 1+ eps output 1+eps
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage

            actor_loss = torch.mean(-torch.min(surr1,surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            #clean the gradient
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            #spread backward
            actor_loss.backward()
            critic_loss.backward()

            #gradient update
            self.actor_optimizer.step()
            self.critic_optimizer.step()
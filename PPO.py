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
    def __init__(self, in_channels, height, width, n_hiddens, balls, pockets, angles, powers,
                 actor_lr, critic_lr, gamma, lmbda,epochs,eps,device):
        self.actor_ball = Targetball_PolicyNet(inchannels=in_channels, height=height,width=width, action_dim=balls,hidden_dim= n_hiddens).to(device=device)
        self.actor_pocket = Targetpocket_PolicyNet(inchannels=in_channels, height=height,width=width, action_dim=pockets,hidden_dim= n_hiddens).to(device=device)
        self.actor_angle = Angle_PolicyNet(inchannels=in_channels, height=height,width=width, action_dim=angles,hidden_dim= n_hiddens).to(device=device)
        self.actor_power = Power_PolicyNet(inchannels=in_channels, height=height,width=width, action_dim=powers,hidden_dim= n_hiddens).to(device=device) 
        self.critic = ValueNet(inchannels=in_channels, height=height,width=width,hidden_dim= n_hiddens).to(device=device)

        self.ball_optimizer = torch.optim.Adam(self.actor_ball.parameters(), lr = actor_lr)
        self.pocket_optimizer = torch.optim.Adam(self.actor_pocket.parameters(), lr = actor_lr)
        self.angle_optimizer = torch.optim.Adam(self.actor_angle.parameters(), lr = actor_lr)
        self.power_optimizer = torch.optim.Adam(self.actor_power.parameters(), lr = actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_ball(self, state):
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float).to(self.device)
        # state = torch.tensor(state).unsqueeze(0).float().to(self.device)  # Ensure state is float
        # state = state.permute(0, 3, 1, 2)
        # print(f"shape of state {state.shape}")
        probs = self.actor_ball(state)
        action_list = torch.distributions.Categorical(probs)
        action_ball = action_list.sample().item()
        return action_ball

    def take_pocket(self, state):
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float).to(self.device)
        # state = torch.tensor(state).unsqueeze(0).float().to(self.device)  # Ensure state is float
        # state = state.permute(0, 3, 1, 2)
        probs = self.actor_pocket(state)
        action_list = torch.distributions.Categorical(probs)
        action_pocket = action_list.sample().item()
        return action_pocket

    def take_angle(self, state):
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float).to(self.device)
        # state = torch.tensor(state).unsqueeze(0).float().to(self.device)  # Ensure state is float
        # state = state.permute(0, 3, 1, 2)
        probs = self.actor_angle(state)
        action_list = torch.distributions.Categorical(probs)
        action_angle = action_list.sample().item()
        return action_angle

    def take_power(self, state):
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float).to(self.device)
        # state = torch.tensor(state).unsqueeze(0).float().to(self.device)  # Ensure state is float
        # state = state.permute(0, 3, 1, 2)
        probs = self.actor_power(state)
        action_list = torch.distributions.Categorical(probs)
        action_power = action_list.sample().item()
        return action_power

    def take_action(self, state):
        action_ball = self.take_ball(state)
        action_pocket = self.take_pocket(state)
        action_angle = self.take_angle(state)
        action_power = self.take_power(state)

        angle = {0: -3, 1: -2, 2: -1, 3: 0, 4: 1, 5: 2, 6: 3}.get(action_angle)
        power = {0: 3, 1: 5, 2: 7}.get(action_power)

        action = poolaction(action_ball + 1, action_pocket, angle, power)

        return action, action_ball, action_pocket, action_angle, action_power

    
    def learn(self, transition_dict):
        # states_array = np.array([s for s in transition_dict['states']])
        # next_states_array = np.array([s for s in transition_dict['next_states']])

        # # Convert numpy arrays to tensors
        # states = torch.tensor(states_array, dtype=torch.float).to(self.device)
        # next_states = torch.tensor(next_states_array, dtype=torch.float).to(self.device)
    
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        action_ball = torch.tensor(transition_dict['action_ball'], dtype=torch.int64).to(self.device).view(-1, 1)
        action_pocket = torch.tensor(transition_dict['action_pocket'], dtype=torch.int64).to(self.device).view(-1, 1)
        action_angle = torch.tensor(transition_dict['action_angle'], dtype=torch.int64).to(self.device).view(-1, 1)
        action_power = torch.tensor(transition_dict['action_power'], dtype=torch.int64).to(self.device).view(-1, 1)
        #solve by using action sets
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device).view(-1, 1)
        # next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        next_states = torch.stack(transition_dict['next_states']).float().to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.device).view(-1, 1)

        states = states.permute(0, 3, 1, 2)  # Rearrange dimensions to [batch_size, channels, height, width]
        next_states = next_states.permute(0, 3, 1, 2)  # Rearrange dimensions to [batch_size, channels, height, width]

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

        old_log_probs_ball = torch.log(self.actor(states).gather(1, action_ball)).detach()
        old_log_probs_pocket = torch.log(self.actor(states).gather(1, action_pocket)).detach()
        old_log_probs_angle = torch.log(self.actor(states).gather(1, action_angle)).detach()
        old_log_probs_power = torch.log(self.actor(states).gather(1, action_power)).detach()

        for _ in range(self.epochs):
            log_probs_ball = torch.log(self.actor(states).gather(1,action_ball))
            log_probs_pocket = torch.log(self.actor(states).gather(1,action_pocket))
            log_probs_angle = torch.log(self.actor(states).gather(1,action_angle))
            log_probs_power = torch.log(self.actor(states).gather(1,action_power))
            #compare the old and new policy
            ratio_ball = torch.exp(log_probs_ball - old_log_probs_ball)
            ratio_pocket = torch.exp(log_probs_pocket - old_log_probs_pocket)
            ratio_angle = torch.exp(log_probs_angle - old_log_probs_angle)
            ratio_power = torch.exp(log_probs_power - old_log_probs_power)
            #optimize the left 
            surr1_ball = ratio_ball * advantage
            surr1_pocket = ratio_pocket * advantage
            surr1_angle = ratio_angle * advantage
            surr1_power = ratio_power * advantage
            #optimize right, if ratio < 1 - eps output 1-eps, if ratio > 1+ eps output 1+eps
            surr2_ball = torch.clamp(ratio_ball, 1-self.eps, 1+self.eps) * advantage
            surr2_pocket = torch.clamp(ratio_pocket, 1-self.eps, 1+self.eps) * advantage
            surr2_angle = torch.clamp(ratio_angle, 1-self.eps, 1+self.eps) * advantage
            surr2_power = torch.clamp(ratio_power, 1-self.eps, 1+self.eps) * advantage

            actor_ball_loss = torch.mean(-torch.min(surr1_ball,surr2_ball))
            actor_pocket_loss = torch.mean(-torch.min(surr1_pocket,surr2_pocket))
            actor_angle_loss = torch.mean(-torch.min(surr1_angle,surr2_angle))
            actor_power_loss = torch.mean(-torch.min(surr1_power,surr2_power))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            #clean the gradient
            self.ball_optimizer.zero_grad()
            self.pocket_optimizer.zero_grad()
            self.angle_optimizer.zero_grad()
            self.power_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            #spread backward
            actor_ball_loss.backward()
            actor_pocket_loss.backward()
            actor_angle_loss.backward()
            actor_power_loss.backward()
            critic_loss.backward()

            #gradient update
            self.ball_optimizer.step()
            self.pocket_optimizer.step()
            self.angle_optimizer.step()
            self.power_optimizer.step()
            self.critic_optimizer.step()
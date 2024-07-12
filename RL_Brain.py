import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# Policy Network
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

# Value Network
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

    def get_action(self, action_number):
        i = 0
        for tb in range(15):
            for th in range(6):
                for a in range(3):
                    for p in range(3):
                        i += 1
                        if i == action_number:
                            self.target_ball = tb + 1
                            self.target_hole = th
                            if a == 0:
                                self.angle = -3
                            elif a == 1:
                                self.angle = 0
                            elif a == 2:
                                self.angle = 3
                            if p == 0:
                                self.power = 3
                            elif p == 1:
                                self.power = 5
                            elif p == 2:
                                self.power = 7

class ActorCritic:
    def __init__(self, in_channels, height, width, n_hiddens, n_actions,
                 actor_lr, critic_lr, gamma, device):
        self.gamma = gamma
        self.device = device

        self.actor = PolicyNet(in_channels, height, width, n_actions, n_hiddens).to(device)
        self.critic = ValueNet(in_channels, height, width, n_hiddens).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def take_action(self, state):
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float).to(self.device)
        state = state.permute(0, 3, 1, 2)  # Rearrange dimensions to [batch_size, channels, height, width]
        probs = self.actor(state)

        # Check for NaNs and handle them
        if torch.any(torch.isnan(probs)):
            print("NaNs detected in action probabilities.")
            probs = torch.nan_to_num(probs, nan=1e-6)

        # Clamp values to avoid numerical issues
        probs = torch.clamp(probs, 1e-6, 1-1e-6)

        # Normalize again to ensure they sum to 1
        probs = probs / probs.sum(dim=1, keepdim=True)

        action_dist = torch.distributions.Categorical(probs)
        action_number = int(action_dist.sample().item())
        action = poolaction(target_ball=0, target_hole=0, angle=0, power=0)
        action.get_action(action_number=action_number)
        return action, action_number

    # def update(self, transition_dict):
    #     states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
    #     actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.int64).view(-1, 1).to(self.device)
    #     rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
    #     next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
    #     dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)

    #     states = states.permute(0, 3, 1, 2)  # Rearrange dimensions to [batch_size, channels, height, width]
    #     next_states = next_states.permute(0, 3, 1, 2)  # Rearrange dimensions to [batch_size, channels, height, width]

    #     td_value = self.critic(states)
    #     td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
    #     td_delta = td_target - td_value

    #     log_probs = torch.log(self.actor(states).gather(1, actions))
    #     actor_loss = torch.mean(-log_probs * td_delta.detach())
    #     critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

    #     self.actor_optimizer.zero_grad()
    #     self.critic_optimizer.zero_grad()
    #     actor_loss.backward()
    #     critic_loss.backward()
    #     self.actor_optimizer.step()
    #     self.critic_optimizer.step()
    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)

        states = states.permute(0, 3, 1, 2)  # Rearrange dimensions to [batch_size, channels, height, width]
        next_states = next_states.permute(0, 3, 1, 2)  # Rearrange dimensions to [batch_size, channels, height, width]

        print(f"States shape: {states.shape}")
        print(f"Actions shape: {actions.shape}")
        print(f"Actions values: {actions}")

        # Ensure actions are within valid range
        actions = torch.clamp(actions, 0, states.shape[1] - 1)

        td_value = self.critic(states)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - td_value

        print(f"TD Value shape: {td_value.shape}")
        print(f"TD Target shape: {td_target.shape}")
        print(f"TD Delta shape: {td_delta.shape}")

        log_probs = torch.log(self.actor(states).gather(1, actions))
        print(f"log_probs shape: {log_probs.shape}")

        actor_loss = torch.mean(-log_probs * td_delta.detach())
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

# import numpy as np
# import torch
# from torch import nn
# from torch.nn import functional as F

# # Policy Network
# class PolicyNet(nn.Module):
#     def __init__(self, inchannels, height, width, action_dim, hidden_dim=128):
#         super(PolicyNet, self).__init__()
#         self.conv1 = nn.Conv2d(inchannels, 256, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(256)
#         self.conv_layers = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(256, 256, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(256),
#                 nn.ReLU()
#             ) for _ in range(8)
#         ])
#         self.fc1 = nn.Linear(256 * height * width, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, action_dim)

#     def forward(self, x):
#         # print(f"Input shape: {x.shape}")
#         x = F.relu(self.bn1(self.conv1(x)))
#         # print(f"After conv1 shape: {x.shape}")
#         for conv_layer in self.conv_layers:
#             x = conv_layer(x)
#             # print(f"After conv_layer shape: {x.shape}")
#         x = x.reshape(x.size(0), -1)
#         # print(f"After reshape shape: {x.shape}")
#         x = F.relu(self.fc1(x))
#         x = F.softmax(self.fc2(x), dim=1)
#         return x
    
# # Value Network
# class ValueNet(nn.Module):
#     def __init__(self, inchannels, height, width, hidden_dim):
#         super(ValueNet, self).__init__()
#         self.conv1 = nn.Conv2d(inchannels, 256, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(256)
#         self.conv_layers = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(256, 256, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(256),
#                 nn.ReLU()
#             ) for _ in range(8)
#         ])
#         self.fc1 = nn.Linear(256 * height * width, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, 1)
        
#     def forward(self, x):
#         # print(f"ValueNet Input shape: {x.shape}")
#         x = F.relu(self.bn1(self.conv1(x)))
#         # print(f"After conv1 shape: {x.shape}")
#         for conv_layer in self.conv_layers:
#             x = conv_layer(x)
#             # print(f"After conv_layer shape: {x.shape}")
#         x = x.reshape(x.size(0), -1)
#         # print(f"After reshape shape: {x.shape}")
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
    
# class poolaction:
#     def __init__(self, target_ball, target_hole, angle, power):
#         self.target_ball = target_ball
#         self.target_hole = target_hole
#         self.angle = angle
#         self.power = power

#     def get_action(self, action_number):
#         i = 0
#         for tb in range(15):
#             for th in range(6):
#                 for a in range(3):
#                     for p in range(3):
#                         i += 1
#                         if i == action_number:
#                             self.target_ball = tb + 1
#                             self.target_hole = th
#                             if a == 0:
#                                 self.angle = -5
#                             elif a == 1:
#                                 self.angle = 0
#                             elif a == 2:
#                                 self.angle = 5
#                             if p == 0:
#                                 self.power = 3
#                             elif p == 1:
#                                 self.power = 5
#                             elif p == 2:
#                                 self.power = 7

# class ActorCritic:
#     def __init__(self, in_channels, height, width, n_hiddens, n_actions,
#                  actor_lr, critic_lr, gamma, device):
#         self.gamma = gamma
#         self.device = device

#         self.actor = PolicyNet(in_channels, height, width, n_actions, n_hiddens).to(device)
#         self.critic = ValueNet(in_channels, height, width, n_hiddens).to(device)
#         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
#         self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
    
#     def take_action(self, state):
#         state = torch.tensor(state[np.newaxis, :], dtype=torch.float).to(self.device)
#         state = state.permute(0, 3, 1, 2)  # Rearrange dimensions to [batch_size, channels, height, width]
#         # print(f"State shape before actor: {state.shape}")
#         probs = self.actor(state)
        
#         # Check for NaNs and handle them
#         if torch.any(torch.isnan(probs)):
#             # print("NaNs detected in action probabilities.")
#             probs = torch.nan_to_num(probs, nan=1e-6)
        
#         # Clamp values to avoid numerical issues
#         probs = torch.clamp(probs, 1e-6, 1-1e-6)
        
#         # Normalize again to ensure they sum to 1
#         probs = probs / probs.sum(dim=1, keepdim=True)

#         action_dist = torch.distributions.Categorical(probs)
#         action_number = int(action_dist.sample().item())
#         action = poolaction(target_ball=0, target_hole=0, angle=0, power=0)
#         action.get_action(action_number=action_number)
#         return action, action_number

#     def update(self, transition_dict):
#         states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
#         actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.int64).view(-1, 1).to(self.device)
#         rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
#         next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
#         dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)

#         states = states.permute(0, 3, 1, 2)  # Rearrange dimensions to [batch_size, channels, height, width]
#         next_states = next_states.permute(0, 3, 1, 2)  # Rearrange dimensions to [batch_size, channels, height, width]

#         td_value = self.critic(states)
#         td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
#         td_delta = td_target - td_value
        
#         log_probs = torch.log(self.actor(states).gather(1, actions))
#         actor_loss = torch.mean(-log_probs * td_delta.detach())
#         critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

#         self.actor_optimizer.zero_grad()
#         self.critic_optimizer.zero_grad()
#         actor_loss.backward()
#         critic_loss.backward()
#         self.actor_optimizer.step()
#         self.critic_optimizer.step()

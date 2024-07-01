import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# #net
# class Net(nn.Module):
#     def __init__(self, n_states, n_hiddens, n_actions):
#         super(Net, self).__init__()
#         #only one hidden layer
#         self.fc1 = nn.Linear(n_states, n_hiddens)
#         self.fc2 = nn.Linear(n_hiddens, n_actions)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         #calculate softmax for each batch, larger the q is, larger the possibility
#         x = F.softmax(x, dim = 1)
#         return x
    
# #Reinforcement Learning Module
# class PolicyGradient:
#     def __init__(self, n_states, n_hiddens, n_actions, learning_rate, gamma) -> None:
#         self.n_states = n_states #number of states
#         self.n_hiddens = n_hiddens
#         self.n_actions = n_actions #number of actions
#         self.learning_rate = learning_rate #decade
#         self.gamma = gamma #policy gradient
#         self._build_net() #build the net model

#     #build the net
#     def _build_net(self):
#         self.policy_net = Net(self.n_states, self.n_hiddens, self.n_actions)
#         self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr = self.learning_rate)

#     #choose actions
#     def take_action(self, stae):
#         #numpu[n_states] --> [1, n_states] --> tensor
#         state = torch.Tensor(state[np.newaxis, :])
#         #get the corresponding probability of each action [1, n_actions]-->[1,n_actions]
#         probs = self.policy_net(state)
#         #build the data distribution based on probabilities
#         action_dist = torch.distributions.Categorical(probs)
#         #extract actions based on every state
#         action = action_dist.sample()
#         #transfer the tensor data to int
#         action = action.item()
#         return action
    
#     #max value for each state
#     def max_q_value(self, state):
#         #[n_states]-->[1,n_states]
#         state = torch.tensor(state, dtype=torch.float).view(1, -1)
#         #get the max reward for each state [1,n_states]-->[1,n_action]-->[1]-->float
#         max_q = self.policy_net(state).max().item()
#         return max_q
    
#     #training module
#     def learn(self, transitions_dict):
#         #get all the chain information from this round
#         state_list = transitions_dict['states']
#         action_list = transitions_dict['actions']
#         reward_list = transitions_dict['rewards']

#         G = 0 #record correnet chain's return value
#         self.optimizer.zero_grad() #optimizer go zero
#         #gradient ascent
#         for i in reversed(range(len(reward_list))):
#             #get reward for each step
#             reward = reward_list[i]
#             #get the state for each step
#             state = torch.tensor(state_list[i], dtype = torch.float).view(1, -1)
#             #get action for each step
#             action = torch.tensor(action_list[i]).view(1,-1)
#             #action value [1,2]
#             q_value = self.policy_net(state)
#             #action for pobability
#             log_prob = torch.log(q_value.gather(1,action))
#             #calculate the current state_value = immediate reward + next clockwise state_value
#             G = reward + self.gamma * G
#             #calculate the loss 
#             loss = -log_prob * G
#             #spread backward
#             loss.backward()
#         #graduebt descent
#         self.optimizer.step()
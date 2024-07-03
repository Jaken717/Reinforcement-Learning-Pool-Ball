###############################################################
#Edition Date: Julay 1st, 2024
#Content: Operation Script for Reinforcement Learning
#Author: JK Liu
###############################################################
import pool_env
import numpy as np
import matplotlib.pyplot as plt
import torch
from pool_env import poolenv
from RL_Brain import ActorCritic

if torch.cuda.is_available():
    print('gpu')
    device = torch.device('cuda')
else:
    print("wrong")
    exit()



#set parameters
num_episode = 1500 #total times
gamma = 0.9
actor_lr = 1e-3 #policy network learning rate
critic_lr = 1e-2 #value network learning rate
hidden_dim = 256 #hidden neural numbers
return_list = [] #kepp the return value for each rounds

#environment loade
env = poolenv()
state_dim = (3, 17, 37)
action_dim = 15 * 6 * 3 * 2

#Model Initialization
agent = ActorCritic(in_channels=state_dim[0],
                    height=state_dim[1],
                    width=state_dim[2],
                    n_hiddens=hidden_dim,
                    n_actions=action_dim,
                    actor_lr=actor_lr,
                    critic_lr=critic_lr,
                    gamma=gamma,
                    device=device)

for i in range(num_episode):
    state = env.reset()
    done = False 
    episode_return = 0

    transidion_dict = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': [],
    }

    while not done:
        action = agent.take_action(state=state)
        next_state, reward, done= env.step(action)
        transidion_dict['states'].append(state)
        transidion_dict['actions'].append(action)
        transidion_dict['next_states'].append(next_state)
        transidion_dict['rewards'].append(reward)
        transidion_dict['dones'].append(done)
        state = next_state
        episode_return += reward

    return_list.append(episode_return)
    agent.update(transidion_dict)

    print(f'iter: {i}, return:{np.mean(return_list[-10:])}')

#Plot results
plt.plot(return_list)
plt.title('Return over time')
plt.show()
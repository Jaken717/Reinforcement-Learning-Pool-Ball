###############################################################
# Edition Date: July 1st, 2024
# Content: Operation Script for Reinforcement Learning
# Author: JK Liu
###############################################################
import pool_env
import numpy as np
import matplotlib.pyplot as plt
import torch
from pool_env import poolenv
from RL_Brain import ActorCritic
from RL_Brain import poolaction
import random
import gamestate
from ball import BallType
import os

def revise_action_number(action):
    i = 0
    for tb in range(15):
        for th in range(6):
            for a in range(7):
                for p in range(3):
                    i += 1
                    angle_val = {0: -3, 1: -2, 2: -1, 3: 0, 4: 1, 5: 2, 6: 3}.get(a)
                    power_val = {0: 3, 1: 5, 2: 7}.get(p)
                    if (action.target_ball == tb + 1 and
                        action.target_hole == th and
                        action.angle == angle_val and
                        action.power == power_val):
                        return i

def calibrate_by_rule(action, last_action, game_state):
    ball_found = False
    valid_balls = []
    holes = [0, 1, 2, 3, 4, 5]
    angles = [-3, -2, -1, 0, 1, 2, 3]
    powers = [3, 5, 7]

    # if game_state.ball_assignment:
    #     print(f"ball type {game_state.ball_assignment[game_state.current_player]} ")

    if game_state.potting_8ball[game_state.current_player]:
        valid_balls.append(8)
    else:
        for ball in game_state.balls:
            if game_state.ball_assignment is None and ball.number != 0:
                # print("is none")
                valid_balls.append(ball.number)
            else:
                if game_state.ball_assignment != None and game_state.ball_assignment[game_state.current_player] == BallType.Solid and ball.number < 8 and ball.number != 0:
                    valid_balls.append(ball.number)
                    # print("solid")
                elif game_state.ball_assignment != None and game_state.ball_assignment[game_state.current_player] == BallType.Striped and ball.number > 8:
                    valid_balls.append(ball.number)
                    # print("striped")
    # print(f"valid balls: {valid_balls}")

    for ball in valid_balls:
        if action.target_ball == ball:
            ball_found = True

    if not ball_found and valid_balls:
        # print("ball not found")
        action.target_ball = random.choice(valid_balls)
        action.target_hole = random.choice(holes)
        action.angle = random.choice(angles)
        action.power = random.choice(powers)
        # print("ball not found")
    if action == last_action and valid_balls:
        action.target_ball = random.choice(valid_balls)
        action.target_hole = random.choice(holes)
        action.angle = random.choice(angles)
        action.power = random.choice(powers)
    return action

if torch.cuda.is_available():
    print('gpu')
    device = torch.device('cuda')
else:
    print("wrong")
    exit()

# Set parameters
num_episode = 150000  # Total episodes
save_interval = 10000  # Save the model every 10000 episodes
gamma = 0.9
actor_lr = 1e-3  # Policy network learning rate
critic_lr = 1e-2  # Value network learning rate
hidden_dim = 256  # Hidden neural numbers
return_list = []  # Keep the return value for each round
plot_interval = 100  # Plot the return every 1000 episodes

# environment load
env = poolenv()
state_dim = (5, 17, 37)
action_dim = 15 * 6 * 7 * 3

# Model Initialization
agent = ActorCritic(in_channels=state_dim[0],
                    height=state_dim[1],
                    width=state_dim[2],
                    n_hiddens=hidden_dim,
                    n_actions=action_dim,
                    actor_lr=actor_lr,
                    critic_lr=critic_lr,
                    gamma=gamma,
                    device=device)

# Load existing model if available
start_episode = 0
if os.path.exists('checkpoint.pth'):
    checkpoint = torch.load('checkpoint.pth')
    agent.load_state_dict(checkpoint['model_state_dict'])
    start_episode = checkpoint['episode']
    return_list = checkpoint['return_list']
    print(f'Resuming training from episode {start_episode}')
else:
    print('First time Training')

for i in range(num_episode):
    state = env.reset()
    done = False
    episode_return = 0

    transition_dict = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': [],
    }

    last_action = poolaction(target_ball=0, target_hole=0, angle=0, power=0)

    while not done:
        action, action_number = agent.take_action(state=state)
        # print(f"action number choice is {action_number}")
        # print(f"Action choice: {action.target_ball}, {action.target_hole}, {action.angle}, {action.power}")
        # print(f"prev Action taken: {last_action.target_ball}, {last_action.target_hole}, {last_action.angle}, {last_action.power}")
        # print(f"Action taken: {action.target_ball}, {action.target_hole}, {action.angle}, {action.power}")
        action = calibrate_by_rule(action=action, last_action=last_action, game_state=env.game)
        action_number = revise_action_number(action=action)
        # print(f"action number is {action_number}")
        next_state, reward, done = env.step(action)
        transition_dict['states'].append(state)
        transition_dict['actions'].append(int(action_number))  # Ensure action_number is an integer
        transition_dict['next_states'].append(next_state)
        transition_dict['rewards'].append(reward)
        transition_dict['dones'].append(done)
        state = next_state
        last_action = action
        episode_return += reward
        if episode_return <= -400:
            done = True

    return_list.append(episode_return)
    agent.update(transition_dict)

    print(f'iter: {i}, current turn: {episode_return}, return: {np.mean(return_list[-10:])}')

    if (i + 1) % plot_interval == 0:
        plt.plot(return_list)
        plt.title('Return over time')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.show()

    if (i + 1) % save_interval == 0:
        torch.save({
            'episode': i + 1,
            'model_state_dict': agent.state_dict(),
            'return_list': return_list
        }, 'checkpoint.pth')
        print(f'Saved model at episode {i + 1}')

env.close()

# Plot results
plt.plot(return_list)
plt.title('Return over time')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.show()
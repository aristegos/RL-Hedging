import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import copy
import torch.optim as optim
import torch.distributions as D
from collections import deque
import random
import matplotlib.pyplot as plt
import scipy.stats as stats
from models import model_objective
            

class ReplayBuffer:
    def __init__(self, capacity=600000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def hot_start_critic_q_func(actor,
                            cost_critic,
                           risk_critic,
                           env,
                           episodes= 20000,
                           batch_size = 128,
                           lr = 0.0001,
                           tau=0.00001,
                           epsilon = 1,
                           epsilon_decay = 0.9995,
                           discount = 1.0,
                           eval_freq = 100,
                           min_epsilon = 0.0,
                          ):
    """
    Trains actor network on Black-Scholes or 
    """
    print("Training Critic (learning q function)...")

    # get targets
    target_cost_critic = copy.deepcopy(cost_critic)
    target_risk_critic = copy.deepcopy(risk_critic)

    # optimizers
    cost_critic_optimizer = torch.optim.Adam(cost_critic.parameters(), lr=lr)
    risk_critic_optimizer = torch.optim.Adam(risk_critic.parameters(), lr=lr)

    # misc
    buffer = ReplayBuffer(capacity = 600000)
    q_guess = []
    objective_per_ep = []
    w_t_per_ep = []
    initial_state = torch.FloatTensor(env.reset()).unsqueeze(0)

    for episode in range(episodes+1):
        # decay epsilon
        epsilon = max(epsilon * epsilon_decay,min_epsilon)

        # init
        state = env.reset()
        done = False
        total_reward = 0.0

        # run episode
        while not done:
            # Add exploratory noise (Gaussian)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                if torch.rand(1) <= epsilon:
                    action = 100*torch.rand(1)
                else:
                    action = actor(state_tensor)[0]
            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            total_reward = reward + discount * total_reward
            state = next_state
        
            # Update at end of episode if enough samples are available
            if len(buffer) > batch_size and episode > 10:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states_tensor = torch.FloatTensor(states)
                actions_tensor = torch.FloatTensor(actions)
                rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
                next_states_tensor = torch.FloatTensor(next_states)
                dones_tensor = torch.FloatTensor(dones).unsqueeze(1)
        
                # Critic update
                # get targets
                with torch.no_grad():
                    next_actions = actor(next_states_tensor)
                    
                    target_cost_q = target_cost_critic(next_states_tensor, next_actions)
                    expected_cost_q = rewards_tensor + (1 - dones_tensor) * (discount * target_cost_q)
                    
                    target_risk_q = target_risk_critic(next_states_tensor, next_actions)
                    expected_risk_q = rewards_tensor**2 + (1 - dones_tensor) * ((2*discount * rewards_tensor * target_cost_q) + (discount**2 * target_risk_q))
    
                # update cost_critic
                current_cost_q = cost_critic(states_tensor, actions_tensor)
                cost_critic_loss = nn.L1Loss()(current_cost_q, expected_cost_q)
                
                cost_critic_optimizer.zero_grad()
                cost_critic_loss.backward()
                cost_critic_optimizer.step()
    
                # update risk_critic
                current_risk_q = risk_critic(states_tensor, actions_tensor)
                risk_critic_loss = nn.L1Loss()(current_risk_q, expected_risk_q)
                
                risk_critic_optimizer.zero_grad()
                risk_critic_loss.backward()
                risk_critic_optimizer.step()
        
                # Update target networks
                for target_param, param in zip(target_cost_critic.parameters(), cost_critic.parameters()):
                    target_param.data.copy_(tau*param.data + (1-tau)*target_param.data)
                for target_param, param in zip(target_risk_critic.parameters(), risk_critic.parameters()):
                    target_param.data.copy_(tau*param.data + (1-tau)*target_param.data)
    
                #store results to memory
                #loss_history.append((cost_critic_loss.item(), risk_critic_loss.item(), actor_loss.item()))
    
            #print(f"Episode: {episode} | Actor Loss: {actor_loss.item():.3f} | Cost Loss: {cost_critic_loss.item():.3f}, Risk Loss: {risk_critic_loss.item():.3f}", end='\r')

        # store total reward to memory
        w_t_per_ep.append(total_reward)
        q_guess.append(model_objective(initial_state, actor(initial_state), cost_critic, risk_critic, env.risk_aversion).item())
        
        if episode >= 2:
            w_T_list = w_t_per_ep[-eval_freq:]
            w_T_mean = np.mean(w_T_list)
            w_T_std = np.std(w_T_list)
            objective = w_T_mean - env.risk_aversion * w_T_std
            mean_guess = np.mean(q_guess[-eval_freq:])
            if not np.isnan(objective):
                objective_per_ep.append(objective)
    
                # print progress
                if (episode+1) % eval_freq == 0:
                    print(f'Episode: {episode+1} | Mean Objective: {objective:.3f} | Guess: {mean_guess:.3f}, Diff: {mean_guess - objective:.3f} | e: {epsilon:.3f}')
                else:
                    print(f'Episode: {episode+1} | Mean Objective: {objective:.3f} | Guess: {mean_guess:.3f}, Diff: {mean_guess - objective:.3f} | e: {epsilon:.3f}', end = '\r')
            else:
                print(f'Episode: {episode+1} | Objective is nan, REINITIALIZE ALL NNs', end = '\r')

    return objective_per_ep, q_guess

# def hot_start_critic_q_func(actor, critic, env, episodes= 500, batch_size = 64, lr = 0.0005,tau=0.001, update_freq = 0.1, action_noise = 0.025, discount = 0.95):
#     """
#     Trains actor network on Black-Scholes or 
#     """
#     print("Training Critic (learning q function)...")

#     target_critic = copy.deepcopy(critic)
#     critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
#     buffer = ReplayBuffer()
#     loss_history = []

#     mean = torch.tensor([0.0])
#     std = torch.tensor([action_noise])
#     update_counter = 0

#     for episode in range(episodes+1):
#         state = env.reset()
#         done = False
#         while not done:
#             # Add exploratory noise (Gaussian)
#             state_tensor = torch.FloatTensor(state).unsqueeze(0)
#             with torch.no_grad():
#                 action = (actor(state_tensor) + torch.normal(mean,std))[0]
#             next_state, reward, done, _ = env.step(action)
#             buffer.push(state, action, reward, next_state, done)
#             state = next_state

#         # Update if enough samples are available
#             if len(buffer) > batch_size and np.random.rand() < update_freq:
#                 states, actions, rewards, next_states, dones = buffer.sample(batch_size)
#                 states_tensor = torch.FloatTensor(states)
#                 actions_tensor = torch.FloatTensor(actions)
#                 rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
#                 next_states_tensor = torch.FloatTensor(next_states)
#                 dones_tensor = torch.FloatTensor(dones).unsqueeze(1)
    
#                 # Critic update
#                 with torch.no_grad():
#                     next_actions = actor(next_states_tensor) + torch.normal(mean,std)
#                     target_q = target_critic(next_states_tensor, next_actions)
#                     expected_q = rewards_tensor + (1 - dones_tensor) * target_q * discount
#                 current_q = critic(states_tensor, actions_tensor)
#                 critic_loss = nn.MSELoss()(current_q, expected_q)
                
#                 critic_optimizer.zero_grad()
#                 critic_loss.backward()
#                 critic_optimizer.step()
    
#                 # Update target networks
#                 for target_param, param in zip(target_critic.parameters(), critic.parameters()):
#                     target_param.data.copy_(tau*param.data + (1-tau)*target_param.data)
    
#                 loss_history.append(critic_loss.item())
#                 update_counter += 1
                
#                 print(f"Episode: {episode}, Update: {update_counter}| Loss: {critic_loss.item():.3f}", end='\r')
                
#                 if update_counter % 250 ==0 and update_counter>=batch_size+50:
#                     print(f"Episode: {episode}, Update: {update_counter} | Loss Mean: {np.mean(loss_history[-50:]):.4f}, Loss StD: {np.std(loss_history[-env.n_steps:]):.4f}")
    
#     return loss_history


def hot_start_critic_value_func(actor, critic, env, episodes= 500, batch_size = 64, lr = 0.001, action_noise = 0.025):
    """
    Trains actor network on Black-Scholes or 
    """
    print("Training Critic (learning value function)...")

    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
    loss_history = []
    returns_memory = torch.empty(0)
    states_memory = torch.empty(0)

    mean = torch.tensor([0.0])
    std = torch.tensor([action_noise])

    for episode in range(1,episodes+1):
        state = env.reset()
        done = False
        trajectory = []
        while not done:
            # Add exploratory noise (Gaussian)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = (actor(state_tensor) + torch.normal(mean,std))[0]
            next_state, reward, done, _ = env.step(action)
            trajectory.append((state, action.item(), reward))
            state = next_state

        returns = []
        G = 0
        for (_,_,r) in reversed(trajectory):
            G = r + 0.9*G
            returns.insert(0,G)
        returns = torch.FloatTensor(returns)

        # Convert trajectory to tensors
        states = torch.FloatTensor(np.array([s for (s, a, r) in trajectory]))

        if not returns_memory.numel():
            returns_memory = returns
            states_memory = states
        else:
            returns_memory = torch.cat((returns_memory,returns))
            states_memory = torch.cat((states_memory,states))
            
        if episode % 50 == 0:
            N = returns_memory.shape[0]
            epoch_steps = round(N/batch_size)
            for batch in range(epoch_steps):
                #make batches
                indices = torch.randperm(N)[:batch_size]
                batch_returns = returns_memory[indices]
                batch_states = states_memory[indices,:]
                                
                est_values = critic(batch_states)
                loss = nn.MSELoss()(est_values,batch_returns.unsqueeze(1))

                critic_optimizer.zero_grad()
                loss.backward()
                critic_optimizer.step()

                loss_history.append(loss.item())
            
            returns_memory = torch.empty(0)
            states_memory = torch.empty(0)
            
        if episode % 50 ==0 and episode>=50:
            print(f"Episode: {episode} | Loss Mean: {np.mean(loss_history[-20:]):.4f}, Loss StD: {np.std(loss_history[-20:]):.4f}")
    return loss_history


# def hot_start_critic_value_func(actor, critic, env, episodes= 500, epochs = 10, lr = 0.001, action_noise = 0.025):
#     """
#     Trains actor network on Black-Scholes or 
#     """
#     print("Training Critic (learning value function)...")

#     critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
#     loss_history = []

#     mean = torch.tensor([0.0])
#     std = torch.tensor([action_noise])

#     for episode in range(episodes+1):
#         state = env.reset()
#         done = False
#         trajectory = []
#         while not done:
#             # Add exploratory noise (Gaussian)
#             state_tensor = torch.FloatTensor(state).unsqueeze(0)
#             with torch.no_grad():
#                 action = (actor(state_tensor) + torch.normal(mean,std))[0]
#             next_state, reward, done, _ = env.step(action)
#             trajectory.append((state, action.item(), reward))
#             state = next_state

#         returns = []
#         G = 0
#         for (_,_,r) in reversed(trajectory):
#             G = r + G
#             returns.insert(0,G)
#         returns = torch.FloatTensor(returns)
#         # returns = (returns - returns.mean()) / (returns.std() + 1e-10)

#         # Convert trajectory to tensors
#         states = torch.FloatTensor(np.array([s for (s, a, r) in trajectory]))
#         actions = torch.FloatTensor(np.array([[a] for (s, a, r) in trajectory]))

#         for epoch in range(epochs):
#             est_values = critic(states)
#             loss = nn.MSELoss()(est_values,returns.unsqueeze(1))

#             critic_optimizer.zero_grad()
#             loss.backward()
#             critic_optimizer.step()

#         loss_history.append(loss.item())
#         print(f"Episode: {episode} | Loss: {loss.item():.4f}", end='\r')
#         if episode % 50 ==0 and episode>=50:
#             print(f"Episode: {episode} | Loss Mean: {np.mean(loss_history[-20:]):.4f}, Loss StD: {np.std(loss_history[-20:]):.4f}")
#     return loss_history


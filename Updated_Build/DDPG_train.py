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
from hot_start_critic import ReplayBuffer
from models import model_objective
from evaluate import evaluation


def DDPG_train(actor,
               cost_critic,
               risk_critic,
               env,
               episodes= 50000,
               batch_size = 128,
               lr = 0.0001,
               tau=0.00001,
               epsilon = 1,
               epsilon_decay = 0.9995,
               discount = 0.95,
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
    target_actor = copy.deepcopy(actor)

    # optimizers
    cost_critic_optimizer = torch.optim.Adam(cost_critic.parameters(), lr=lr)
    risk_critic_optimizer = torch.optim.Adam(risk_critic.parameters(), lr=lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr)

    # misc
    buffer = ReplayBuffer()
    loss_history = []
    objective_per_ep = []
    w_t_per_ep = []

    for episode in range(episodes+1):
        # decay epsilon
        epsilon = max(epsilon * epsilon_decay ** episode,min_epsilon)

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
            total_reward += reward
            state = next_state
        
            # Update during episode if enough samples are available
            if len(buffer) > batch_size and episode > 10:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states_tensor = torch.FloatTensor(states)
                actions_tensor = torch.FloatTensor(actions)
                rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
                next_states_tensor = torch.FloatTensor(next_states)
                dones_tensor = torch.FloatTensor(dones).unsqueeze(1)
        
                # Updates
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
    
                #Actor update
                actor_action = actor(states_tensor)
                objective = model_objective(states_tensor, actor_action, cost_critic, risk_critic, env.risk_aversion)
                actor_loss = -torch.mean(objective)
                
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
        
                # Update target networks
                for target_param, param in zip(target_cost_critic.parameters(), cost_critic.parameters()):
                    target_param.data.copy_(tau*param.data + (1-tau)*target_param.data)
                for target_param, param in zip(target_risk_critic.parameters(), risk_critic.parameters()):
                    target_param.data.copy_(tau*param.data + (1-tau)*target_param.data)
                for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                    target_param.data.copy_(tau*param.data + (1-tau)*target_param.data)
    
                #store results to memory
                #loss_history.append((cost_critic_loss.item(), risk_critic_loss.item(), actor_loss.item()))
    
            #print(f"Episode: {episode} | Actor Loss: {actor_loss.item():.3f} | Cost Loss: {cost_critic_loss.item():.3f}, Risk Loss: {risk_critic_loss.item():.3f}", end='\r')

        # store total reward to memory
        w_t_per_ep.append(total_reward)

        # calculate objective function for episode
        if episode >= 2:
            w_T_list = w_t_per_ep[-eval_freq:]
            w_T_mean = np.mean(w_T_list)
            w_T_std = np.std(w_T_list)
            objective = w_T_mean - env.risk_aversion * w_T_std
            if not np.isnan(objective):
                objective_per_ep.append(objective)
    
                # print progress
                if (episode+1) % eval_freq == 0:
                    print(f'Episode: {episode+1} | Mean Objective: {objective:.3f} | PnL: {w_T_mean:.3f}, Risk: {env.risk_aversion * w_T_std:.3f}')
                else:
                    print(f'Episode: {episode+1} | Mean Objective: {objective:.3f}', end = '\r')
            else:
                print(f'Episode: {episode+1} | Objective is nan, REINITIALIZE ALL NNs', end = '\r')

    return objective_per_ep, loss_history
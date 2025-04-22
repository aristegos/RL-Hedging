import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
from collections import deque
import random
import matplotlib.pyplot as plt
import scipy.stats as stats
from deltas import bartlett_delta, BS_delta

def hot_start_gen_actor_samples(env, n_paths = 100):
    """
    Generates sample training data for actor using black-scholes solution

    Specifications:
    - Generates paths from HedgingEnv
    - Determines delta hedge (according to black scholes)
        - in order to calculate delta, we must estimate vol (or variance) as this is unknown to the agent
        - do so by calculating expanding window var estimate of returns (specify if we use exponential rolling or simple avg)
    - vol_learning_window: proportion of time steps at beginning of each path where the "expert" does not do anything
        --> idea is the "expert" must learn the var first in order to properly delta hedge
    - prints progress
    
    - Outputs:
        - X: matrix of var estimates and stock prices at each time step
        - y: vector of black scholes delta hedge
        - vol_est_actual: vector of estimated vol vs actual (for later analysis)
    """
    print("Generating Samples for hot start...")
    X = torch.zeros(1, 4)
    y = torch.zeros(1, 1)
    vol_est_actual = torch.zeros(1, 2)
    
    # generate training data
    for n in range(n_paths):
        state = env.reset()
        done = False
        while not done:
            S, sigma, a_prev, tau = state
            print(f'Progress: {100*(n*env.n_steps+(1-tau/env.T) * env.n_steps) / (n_paths*env.n_steps):.2f}%', end = '\r')
            # add expert action
            if env.stochastic_vol:
                delta_hedge = 100*bartlett_delta(env)
            else:
                delta_hedge = 100*BS_delta(env)

            X = torch.cat((X,torch.tensor([[S, sigma, a_prev, tau]])))
            y = torch.cat((y,torch.tensor([[delta_hedge]])))

            # next step
            state, reward, done, _ = env.step(delta_hedge)
                
    return X[1:], y[1:]
            



def hot_start_actor(actor_net, X, y, epochs= 10, batch_size = 64, lr = 0.01):
    """
    Trains actor network on Black-Scholes or 
    """
    print("Training Actor...")

    X = X.float().requires_grad_(True)
    y = y.float().requires_grad_(True)
    
    optimizer = torch.optim.Adam(actor_net.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss()
    loss_history = []
    
    N = X.shape[0]
    
    for epoch in range(epochs):
        epoch_steps = round(N/batch_size)
        for batch in range(epoch_steps):
            #make batches
            indices = torch.randperm(N)[:batch_size]
            batch_X = X[indices,:]
            batch_y = y[indices,:]

            #optimize
            optimizer.zero_grad()
            y_pred = actor_net.forward(batch_X)
            loss = loss_func(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            print(f"Epoch: {epoch+1}, Batch: {batch+1}/{epoch_steps} | Loss: {loss.item():.3f}", end='\r')
        if epoch % 10 == 0:
            print()
    return actor_net, loss_history
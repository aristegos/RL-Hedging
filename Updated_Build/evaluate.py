import numpy as np
import gym
import scipy.stats as stats
import torch
from deltas import bartlett_delta, BS_delta

class evaluation:
    def __init__(self):
        pass

    def eval_policy(self, env, actor, episodes = 1000, verbose = False):
        print("testing...")

        w_T_list = []
        
        for i in range(episodes):
            print(f'Episode: {i+1}/{episodes}', end = '\r')
            
            state = env.reset()
            done = False
            action_store = []
            total_reward = 0.0
            
            while not done:
                if actor == "Nothing":
                    action = torch.tensor([0.0])
                elif actor == "BS":
                    action = BS_delta(env)*100
                elif actor == "Bartlett":
                    action = bartlett_delta(env)*100
                else:
                    action = actor(torch.tensor(state))
                
                state, reward, done, _ = env.step(action)
                action_store.append(action)
                total_reward += reward

            w_T_list.append(total_reward)

            if (i+1) % 100 == 0 and verbose == True:
                w_T_mean = np.mean(w_T_list)
                w_T_std = np.std(w_T_list)
                objective = w_T_mean - env.risk_aversion * w_T_std
                print(f'Episode: {i+1} | Mean Objective: {objective:.3f} | PnL: {w_T_mean:.3f}, Risk: {env.risk_aversion * w_T_std:.3f}')

        w_T_mean = np.mean(w_T_list)
        w_T_std = np.std(w_T_list)
        objective = w_T_mean - env.risk_aversion * w_T_std
        
        return objective, w_T_list
                
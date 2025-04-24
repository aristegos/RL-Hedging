import torch
import torch.nn as nn
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 256)
        #self.norm1 = nn.LayerNorm(256,256)
        self.fc2 = nn.Linear(256, 256)
        #self.norm2 = nn.LayerNorm(256,256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = torch.sigmoid(self.fc3(x)) * 100.0
        return x

class DDPG_Cost_Critic(nn.Module):
    # Hedging Cost model
    def __init__(self, state_dim, action_dim):
        super(DDPG_Cost_Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim+action_dim, 512)
        #self.norm1 = nn.LayerNorm(256,256)
        self.fc2 = nn.Linear(512, 512)
        #self.norm2 = nn.LayerNorm(256,256)
        self.out = nn.Linear(512, 1)

    def forward(self, state, action):
        x = self.fc1(torch.cat([state, action], dim=1))
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        q_function = self.out(x)
        
        return q_function

class DDPG_Risk_Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DDPG_Risk_Critic, self).__init__()
        
        self.fc1 = nn.Linear(state_dim+action_dim, 512)
        #self.norm1 = nn.LayerNorm(256,256)
        self.fc2 = nn.Linear(512, 512)
        #self.norm2 = nn.LayerNorm(256,256)
        self.out = nn.Linear(512, 1)

    def forward(self, state, action):
        x = self.fc1(torch.cat([state, action], dim=1))
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        q_function = self.out(x)
        
        return q_function

def model_objective(states, actions, DDPG_Cost_Critic, DDPG_Risk_Critic, risk_aversion):
    costs =  DDPG_Cost_Critic(states, actions)
    risks = DDPG_Risk_Critic(states, actions)

    objective = costs - risk_aversion*torch.sqrt(torch.relu(risks - torch.square(costs)))
    return objective
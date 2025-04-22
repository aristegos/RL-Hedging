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

##############################
# Environment Implementation #
##############################

class HedgingEnv(gym.Env):
    """
    Custom hedging environment.

    - Dynamics: Underlying asset follows a geometric Brownian motion.
      If stochastic_vol==True, volatility updates via a simplified SABR model.
    - Reward: At each timestep the reward is the change in wealth adjusted by transaction costs,
      and a risk penalty scaling the variance (approximated here via the squared change).
    - At t=0, the agent buys the replicating portfolio.
    - At maturity, the terminal payoff of a European call is subtracted.
    """
    def __init__(self,
                 T=1.0,            # time horizon (e.g. 1 year)
                 n_steps=50,       # number of timesteps per episode
                 S0=100.0,         # initial asset price
                 sigma0=0.2,       # initial volatility
                 kappa=0.001,      # transaction cost parameter
                 risk_aversion=0.01,  # risk–penalty parameter lambda
                 strike=100.0,     # strike price of the option (European call)
                 nu=0.1,           # vol-of-vol (for SABR dynamics)
                 rho=-0.3,         # correlation between asset and volatility shocks
                 stochastic_vol=True):  # whether to use stochastic volatility dynamics
        super(HedgingEnv, self).__init__()
        self.T = T
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.S0 = S0
        self.sigma0 = sigma0
        self.kappa = kappa
        self.risk_aversion = risk_aversion
        self.strike = strike
        self.nu = nu
        self.rho = rho
        self.stochastic_vol = stochastic_vol

        # Continuous action: hedge position. (We assume it can be any real number.)
        # I think it makes sense to bound the action space to a more reasonable range, presumably between (-1 and 1) for the ranges of delta
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        # State: we use [S, sigma, previous hedge, normalized time]
        # probably should be bounded as well
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.reset()

    def option_price(self):   
        if self.stochastic_vol:
            # calculate sabr_implied_vol
            if self.S == self.strike:
                a = 1 + ((1/4)*(self.rho*self.nu*self.sigma) + (2-3*self.rho**2) * (self.nu**2/24))*self.tau
                a = self.sigma * a
                implied_vol = a
            else:
                a = 1 + ((1/4)*(self.rho*self.nu*self.sigma) + (2-3*self.rho**2) * (self.nu**2/24))*self.tau
                a = self.sigma * a
                z = (self.nu / self.sigma) * np.log(self.S / self.strike)
                x = np.log((np.sqrt(1-2*self.rho*z + z**2) + z - self.rho)/(1-self.rho))
                implied_vol = a * (z/x)
        else:
            implied_vol = self.sigma
        d1 = (np.log(self.S / self.strike) + (0.5 * implied_vol ** 2) * self.tau) / (implied_vol * np.sqrt(self.tau))
        d2 = d1 - implied_vol * np.sqrt(self.tau)
    
        return self.S * stats.norm.cdf(d1) - self.strike * stats.norm.cdf(d2)

    def update_dynamics(self):
        # Update underlying dynamics:
        Z1 = np.random.normal()
        if self.stochastic_vol:
            # Stochastic volatility update using a simplified SABR-like model (I assume beta = 1)
            Z2 = self.rho * Z1 + np.sqrt(1 - self.rho ** 2) * np.random.normal()
            self.S = self.S * np.exp((-0.5 * self.sigma**2) * self.dt + self.sigma * np.sqrt(self.dt) * Z1)
            self.sigma = self.sigma * np.exp((-0.5 * self.nu**2 * self.dt) + self.nu * np.sqrt(self.dt) * Z2)
        else:
            # Constant volatility dynamics
            self.S = self.S * np.exp((-0.5 * self.sigma**2) * self.dt + self.sigma * np.sqrt(self.dt) * Z1)
    
    def reset(self):
        self.t = 0
        self.tau = self.T
        self.S = self.S0
        self.sigma = self.sigma0
        self.a_prev = 0.0  # initial hedge (no position)
        self.v_prev = self.option_price()
        self.s_prev = 0.0
        self.state = np.array([self.S, self.sigma, self.a_prev, 0.0], dtype=np.float32)
        return self.state

    def step(self, action):
        # determine action
        if self.t < self.n_steps:
            action = float(action)
        else:
            action = 0.0
        done = False
        info = {}

        # update stock price and vol (if stochastic_vol == true)
        self.update_dynamics()
        self.t += 1

        # accounting PnL
        reward = self.a_prev*(self.S - self.s_prev) - self.kappa * abs(self.S * (action - self.a_prev))
        if self.t >= self.n_steps:
            done = True
            reward = reward - (max(self.S - self.strike,0) - self.v_prev) * 100
        else:
            reward = reward - (self.option_price() - self.v_prev) * 100

        self.a_prev = action
        self.v_prev = self.option_price()
        self.s_prev = self.S
        self.tau = max(self.T * (1- (self.t/self.n_steps)), 1e-10)

        self.state = np.array([self.S, self.sigma, self.a_prev, self.tau], dtype=np.float32)
        return self.state, reward, done, info
        
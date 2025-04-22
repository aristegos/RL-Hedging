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

def BS_delta(env):
    d1 = (np.log(env.S / env.strike) + 0.5 * env.sigma**2 * env.tau) / (env.sigma * np.sqrt(env.tau))
    delta = stats.norm.cdf(d1)
    return delta

def bartlett_delta(env):
    ds = 0.001
    dsigma = ds * env.nu * env.rho / env.S

    bs_price1, _ = sabr_implied_vol_and_price(env)
    bs_price2, _ = sabr_implied_vol_and_price(env, ds, dsigma)

    b_delta = (bs_price2 - bs_price1) / ds
    return b_delta


def sabr_implied_vol_and_price(env, ds=0, dsigma=0):
    if env.S == env.strike:
        a = 1 + ((1/4)*(env.rho*env.nu*(env.sigma+dsigma)) + (2-3*env.rho**2) * (env.nu**2/24))*env.tau
        a = (env.sigma+dsigma) * a
        implied_vol = a
    else:
        a = 1 + ((1/4)*(env.rho*env.nu*(env.sigma+dsigma)) + (2-3*env.rho**2) * (env.nu**2/24))*env.tau
        a = (env.sigma+dsigma) * a
        z = (env.nu / (env.sigma+dsigma)) * np.log((env.S+ds) / env.strike)
        x = np.log((np.sqrt(1-2*env.rho*z + z**2) + z - env.rho)/(1-env.rho))
        implied_vol = a * (z/x)

    d1 = (np.log((env.S+ds) / env.strike) + (0.5 * implied_vol ** 2) * env.tau) / (implied_vol * np.sqrt(env.tau))
    d2 = d1 - implied_vol * np.sqrt(env.tau)

    return (env.S+ds) * stats.norm.cdf(d1) - env.strike * stats.norm.cdf(d2), implied_vol
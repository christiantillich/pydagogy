#%% [markdown]
"""
# Objective

The goal for this doc is to reproduce the coin flip problem presented in Chapter
2 of Bayesian Analysis with Python. 

# Analysis
"""


import numpy as np
import scipy.stats as stats
import pymc as pm

def posterior_grid(grid_points = 100, heads=6, tosses=9):
    """A grid implementation for the coin flip problem"""
    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(5, grid_points)
    likelihood = stats.binom.pmf(heads, tosses, grid)
    unstd_posterior = likelihood * prior
    posterior = unstd_posterior / unstd_posterior.sum()
    return grid, unstd_posterior


# Coin flipping in PyMC
np.random.seed(123)
n_experiments = 4
theta_real = 0.35
data = stats.bernoulli.rvs(p=theta_real, size = n_experiments)

with pm.Model() as our_first_model:
    theta = pm.Beta('theta', alpha = 1, beta = 1) #Prior
    y = pm.Bernoulli('y', p = theta, observed = data) #Estimate y
    start = pm.find_MAP()
    step = pm.Metropolis()
    trace = pm.sample(1000, step=step, start = start, return_inferencedata = True)
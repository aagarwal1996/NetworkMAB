import numpy as np
import matplotlib.pyplot as plt
import random

from itertools import chain, combinations


#bandit algorithms
from bandit_algorithms.graph_utils import generate_all_arms
from bandit_algorithms.ucb import UCB1
from bandit_algorithms.exp3 import Exp3

#dgp
from synthetic_experiments.dgp import generate_fourier_coefficients, generate_all_fourier_characteristics
from synthetic_experiments.dgp import generate_noisy_reward, create_graph,reward_function






if __name__ == "__main__":
    N = 3
    s = 1
    T = 2**(N+1)
    n_arms = 2**N
    connections = create_graph(N, s)
    all_rewards = reward_function(N, connections)
    all_mean_rewards = np.mean(all_rewards, axis=0)
    all_arms = generate_all_arms(N)
    all_arms_to_col_index = {arm: i for i, arm in enumerate(all_arms)}
    col_index_to_all_arms = {i: arm for i, arm in enumerate(all_arms)}

    optimal_mean_reward, optimal_arm = np.max(all_mean_rewards), np.argmax(all_mean_rewards)
    print("optimal_reward", optimal_mean_reward)
    print("optimal_arm", col_index_to_all_arms[optimal_arm])
    
    ucb = UCB1(N)

    ucb_regret = []

    for _ in range(T):
        chosen_arm = ucb.select_arm()
        col_index = all_arms_to_col_index[chosen_arm]
        reward = all_rewards[:, col_index]
        mean_reward = np.mean(reward)
        noisy_reward = reward + np.random.normal(0, 1)
        ucb.update(chosen_arm, noisy_reward)
        ucb_regret.append(optimal_mean_reward - mean_reward)


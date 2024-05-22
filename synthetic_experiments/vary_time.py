import numpy as np
import matplotlib.pyplot as plt
import random

from itertools import chain, combinations


#bandit algorithms
from bandit_algorithms.graph_utils import generate_all_arms
from bandit_algorithms.ucb import UCB1
from bandit_algorithms.exp3 import Exp3
from bandit_algorithms.unknown_graph import lasso_graph
from bandit_algorithms.known_graph import OLS_graph


#dgp
from synthetic_experiments.dgp import generate_fourier_coefficients, generate_all_fourier_characteristics
from synthetic_experiments.dgp import generate_noisy_reward, create_graph,reward_function



if __name__ == "__main__":
    N = 8
    s = 4
    T = (2**N)*10
    n_arms = 2**N
    noise_scale = 1
    connections = create_graph(N, s)
    all_rewards = reward_function(N, s,connections)
    all_mean_rewards = np.mean(all_rewards, axis=0)
    all_arms = generate_all_arms(N)
    all_arms_to_col_index = {arm: i for i, arm in enumerate(all_arms)}
    col_index_to_all_arms = {i: arm for i, arm in enumerate(all_arms)}

    optimal_mean_reward, optimal_arm = np.max(all_mean_rewards), np.argmax(all_mean_rewards)
    print("optimal_reward", optimal_mean_reward)
    print("optimal_arm", col_index_to_all_arms[optimal_arm])
    print("connections", connections)
    
    ucb = UCB1(N)
    lasso = lasso_graph(N,T)
    ols = OLS_graph(N,T,connections)
    

    ucb_regret = []
    lasso_regret = []
    ols_regret = []
    random.seed(0)
    np.random.seed(0)

    for _ in range(T):
        chosen_arm = ols.select_arm()
        #print("chosen_arm", chosen_arm)
        col_index = all_arms_to_col_index[chosen_arm]
        reward = all_rewards[:, col_index]
        mean_reward = np.mean(reward)
        noisy_reward = reward + np.random.normal(0, noise_scale)
        ols.update(chosen_arm, noisy_reward)
        ols_regret.append(optimal_mean_reward - mean_reward)
    
    

    for _ in range(T):
        chosen_arm = lasso.select_arm()
        #print("chosen_arm", chosen_arm)
        col_index = all_arms_to_col_index[chosen_arm]
        reward = all_rewards[:, col_index]
        mean_reward = np.mean(reward)
        noisy_reward = reward + np.random.normal(0, noise_scale)
        lasso.update(chosen_arm, noisy_reward)
        lasso_regret.append(optimal_mean_reward - mean_reward)

    
    for _ in range(T):
        chosen_arm = ucb.select_arm()
        col_index = all_arms_to_col_index[chosen_arm]
        reward = all_rewards[:, col_index]
        mean_reward = np.mean(reward)
        noisy_reward = reward + np.random.normal(0, noise_scale)
        ucb.update(chosen_arm, noisy_reward)
        ucb_regret.append(optimal_mean_reward - mean_reward)


        # Plotting the regrets
    plt.figure(figsize=(12, 8))

    plt.plot(np.cumsum(ols_regret), label='OLS Regret')
    plt.plot(np.cumsum(lasso_regret), label='Lasso Regret')
    plt.plot(np.cumsum(ucb_regret), label='UCB Regret')

    plt.xlabel('Horizon',fontsize=14)
    plt.ylabel('Cumulative Regret',fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    
    plt.show()

    


    #plt.yscale('log')
    #plt.title('Regret Comparison of OLS, Lasso, and UCB')
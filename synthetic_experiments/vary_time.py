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
from bandit_algorithms.known_graph_global import global_OLS_graph


#dgp
from synthetic_experiments.dgp import generate_fourier_coefficients, generate_all_fourier_characteristics
from synthetic_experiments.dgp import generate_noisy_reward, create_graph,reward_function



if __name__ == "__main__":
    N = 9
    s = 4
    T = (2**N)*11
    n_arms = 2**N
    noise_scale = 0.2
    iterations = 3

    ucb_regret_all = []
    lasso_regret_all = []
    ols_regret_all = []
    global_ols_regret_all = []
    

    for iter in range(iterations):
        connections = create_graph(N, s)
        all_rewards = reward_function(N, s, connections)
        all_mean_rewards = np.mean(all_rewards, axis=0)
        all_arms = generate_all_arms(N)
        all_arms_to_col_index = {arm: i for i, arm in enumerate(all_arms)}
        col_index_to_all_arms = {i: arm for i, arm in enumerate(all_arms)}

        optimal_mean_reward, optimal_arm = np.max(all_mean_rewards), np.argmax(all_mean_rewards)
        print("optimal_reward", optimal_mean_reward)
        print("optimal_arm", col_index_to_all_arms[optimal_arm])
        print("connections", connections)
        
        ucb = UCB1(N)
        lasso = lasso_graph(N, T)
        ols = OLS_graph(N, T, connections)
        global_ols = global_OLS_graph(N, T, connections)
      
        
        ucb_regret = []
        lasso_regret = []
        ols_regret = []
        global_ols_regret = []
       
        random.seed(iter)
        np.random.seed(iter)

        algorithms = [(lasso, lasso_regret), (ols, ols_regret), (ucb, ucb_regret)]  # Add or remove algorithms here as needed

        for _ in range(T):
            for algorithm, regret_list in algorithms:
                chosen_arm = algorithm.select_arm()
                col_index = all_arms_to_col_index[chosen_arm]
                reward = all_rewards[:, col_index]
                mean_reward = np.mean(reward)
                noisy_reward = reward + np.random.normal(0, noise_scale)
                algorithm.update(chosen_arm, noisy_reward)
                regret_list.append(optimal_mean_reward - mean_reward)

        for _, regret_list in algorithms:
            if regret_list is ucb_regret:
                ucb_regret_all.append(np.cumsum(ucb_regret))
            elif regret_list is ols_regret:
                ols_regret_all.append(np.cumsum(ols_regret))
            elif regret_list is lasso_regret:
                lasso_regret_all.append(np.cumsum(lasso_regret))
            elif regret_list is global_ols_regret:
                global_ols_regret_all.append(np.cumsum(global_ols_regret))

    regret_means = {}
    regret_stds = {}

    if ucb_regret_all:
        regret_means['UCB Regret'] = np.mean(ucb_regret_all, axis=0)
        regret_stds['UCB Regret'] = np.std(ucb_regret_all, axis=0)
    if ols_regret_all:
        regret_means['OLS Regret'] = np.mean(ols_regret_all, axis=0)
        regret_stds['OLS Regret'] = np.std(ols_regret_all, axis=0)
    if lasso_regret_all:
        regret_means['Lasso Regret'] = np.mean(lasso_regret_all, axis=0)
        regret_stds['Lasso Regret'] = np.std(lasso_regret_all, axis=0)
    if global_ols_regret_all:
        regret_means['Global OLS Regret'] = np.mean(global_ols_regret_all, axis=0)
        regret_stds['Global OLS Regret'] = np.std(global_ols_regret_all, axis=0)

    # Plotting the regrets
    plt.figure(figsize=(12, 8))

    for label, regret_mean in regret_means.items():
        regret_std = regret_stds[label]
        plt.plot(regret_mean, label=label, linewidth=2)
        plt.fill_between(range(T), regret_mean - regret_std, regret_mean + regret_std, alpha=0.2)
    
    plt.xlabel('Horizon', fontsize=14)
    plt.ylabel('Cumulative Regret', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    
    plt.show()

    #plt.yscale('log')
    #plt.title('Regret Comparison of OLS, Lasso, and UCB')
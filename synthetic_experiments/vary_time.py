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
    N = 10
    s = 4
    T = (2**N)*10
    n_arms = 2**N
    noise_scale = 1
    iterations = 2

    ucb_regret_all = []
    lasso_regret_all = []
    ols_regret_all = []
    global_ols_regret_all = []
    

    for _ in range(iterations):
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
       
        random.seed(0)
        np.random.seed(0)

        for _ in range(T):
            chosen_arm = global_ols.select_arm()
            col_index = all_arms_to_col_index[chosen_arm]
            reward = all_rewards[:, col_index]
            mean_reward = np.mean(reward)
            noisy_reward = reward + np.random.normal(0, noise_scale)
            global_ols.update(chosen_arm, noisy_reward)
            global_ols_regret.append(optimal_mean_reward - mean_reward)

        for _ in range(T):
            chosen_arm = ols.select_arm()
            col_index = all_arms_to_col_index[chosen_arm]
            reward = all_rewards[:, col_index]
            mean_reward = np.mean(reward)
            noisy_reward = reward + np.random.normal(0, noise_scale)
            ols.update(chosen_arm, noisy_reward)
            ols_regret.append(optimal_mean_reward - mean_reward)
        
        for _ in range(T):
            chosen_arm = lasso.select_arm()
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
        
        # for _ in range(T):
        #     chosen_arm = exp3.select_arm()
        #     col_index = all_arms_to_col_index[chosen_arm]
        #     reward = all_rewards[:, col_index]
        #     mean_reward = np.mean(reward)
        #     noisy_reward = reward + np.random.normal(0, noise_scale)
        #     exp3.update(chosen_arm, noisy_reward)
        #     exp3_regret.append(optimal_mean_reward - mean_reward)

        ucb_regret_all.append(np.cumsum(ucb_regret))
        lasso_regret_all.append(np.cumsum(lasso_regret))
        ols_regret_all.append(np.cumsum(ols_regret))
        global_ols_regret_all.append(np.cumsum(global_ols_regret))

    ucb_regret_mean = np.mean(ucb_regret_all, axis=0)
    lasso_regret_mean = np.mean(lasso_regret_all, axis=0)
    ols_regret_mean = np.mean(ols_regret_all, axis=0)
    global_ols_regret_mean = np.mean(global_ols_regret_all, axis=0)

    ucb_regret_std = np.std(ucb_regret_all, axis=0)
    lasso_regret_std = np.std(lasso_regret_all, axis=0)
    ols_regret_std = np.std(ols_regret_all, axis=0)
    global_ols_regret_std = np.std(global_ols_regret_all, axis=0)

    # Plotting the regrets
    plt.figure(figsize=(12, 8))

    plt.plot(ols_regret_mean, label='OLS Regret', linewidth=2)
    plt.fill_between(range(T), ols_regret_mean - ols_regret_std, ols_regret_mean + ols_regret_std, alpha=0.2)
    plt.plot(lasso_regret_mean, label='Lasso Regret', linewidth=2)
    plt.fill_between(range(T), lasso_regret_mean - lasso_regret_std, lasso_regret_mean + lasso_regret_std, alpha=0.2)
    plt.plot(ucb_regret_mean, label='UCB Regret', linewidth=2)
    plt.fill_between(range(T), ucb_regret_mean - ucb_regret_std, ucb_regret_mean + ucb_regret_std, alpha=0.2)
    plt.plot(global_ols_regret_mean, label='Global OLS Regret', linewidth=2)
    plt.fill_between(range(T), global_ols_regret_mean - global_ols_regret_std, global_ols_regret_mean + global_ols_regret_std, alpha=0.2)
   
    
    plt.xlabel('Horizon', fontsize=14)
    plt.ylabel('Cumulative Regret', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    
    plt.show()

    #plt.yscale('log')
    #plt.title('Regret Comparison of OLS, Lasso, and UCB')
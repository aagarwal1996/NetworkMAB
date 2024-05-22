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


method_name = {
    "UCB": "UCB",
    "Lasso": "Network MAB (Unknown)",
    "OLS": "Network MAB (Known)",
    "Global OLS": "Global OLS"
}


if __name__ == "__main__":
    s = 4
    noise_scale = 0.7
    iterations = 3
    N_values = [5,6,7,8,9,10]  # List of N values to iterate over

    all_regret_means = {N: {} for N in N_values}
    all_regret_stds = {N: {} for N in N_values}

    for N in N_values:
        T = (2**N) * 11
        n_arms = 2**N

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
            regret_means['UCB'] = np.mean([regret[-1] for regret in ucb_regret_all])
            regret_stds['UCB'] = np.std([regret[-1] for regret in ucb_regret_all])
        if ols_regret_all:
            regret_means['OLS'] = np.mean([regret[-1] for regret in ols_regret_all])
            regret_stds['OLS'] = np.std([regret[-1] for regret in ols_regret_all])
        if lasso_regret_all:
            regret_means['Lasso'] = np.mean([regret[-1] for regret in lasso_regret_all])
            regret_stds['Lasso'] = np.std([regret[-1] for regret in lasso_regret_all])
        if global_ols_regret_all:
            regret_means['Global OLS'] = np.mean([regret[-1] for regret in global_ols_regret_all])
            regret_stds['Global OLS'] = np.std([regret[-1] for regret in global_ols_regret_all])
        all_regret_means[N] = regret_means
        all_regret_stds[N] = regret_stds
    # Plotting the regrets for different N values
    fig, ax = plt.subplots(figsize=(12, 8))

    for label in regret_means.keys():
        
        means = [all_regret_means[N][label] for N in N_values]
        stds = [all_regret_stds[N][label] for N in N_values]
        ax.plot(N_values, means, label=method_name[label], linewidth=2, marker='o')
        ax.fill_between(N_values, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.2)
    ax.set_xlabel('Number of units (N)', fontsize=18)
    ax.set_ylabel('Cumulative Regret', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=18)  # Increased legend font size
    ax.grid(True)

    plt.savefig('regret_comparison_plot.png', bbox_inches='tight')  # Save the plot
    plt.show()
    #plt.title('Regret Comparison of OLS, Lasso, and UCB')


import math
import random
import numpy as np


#graph imports
from bandit_algorithms.graph_utils import generate_all_arms
from bandit_algorithms.fourier_utils import generate_fourier_characteristics, generate_all_fourier_characteristics

#sklearn imports
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, RidgeCV
from itertools import chain, combinations


def generate_all_subsets(N):
    """
    Generate all subsets of the set [0, 1, ..., N-1]
    """
    return list(chain.from_iterable(combinations(range(N), r) for r in range(N+1)))

def filter_subsets(connections, all_subsets):
    """
    Filter subsets to include only those with indices in connections
    """
    return [subset for subset in all_subsets if set(subset).issubset(connections)]



class OLS_graph:
    def __init__(self, N,T,connections,num_actions = 2):
        self.N = N
        self.n_arms = num_actions**N
        self.all_arms = generate_all_arms(N,num_actions=2)
        self.explore_horizon = T**(2/3)
        self.counts = {arm: 0 for arm in self.all_arms} 
        self.num_pulls = 0
        self.optimal_arm = None
        self.pulled_arms = []
        self.observed_rewards = []
        self.connections = connections
        self.all_arms_to_col_index = {arm: i for i, arm in enumerate(self.all_arms)}
        self.col_index_to_all_arms = {i: arm for i, arm in enumerate(self.all_arms)}
        self.relevant_subsets = self._get_relevant_subsets()
        print(self.relevant_subsets)
    def select_arm(self):
        
        if self.num_pulls < self.explore_horizon:
            selected_arm = random.choice(self.all_arms)
            return selected_arm
        else:
            if self.optimal_arm is None:
                self.optimal_arm = self.find_optimal_arm()
            return self.optimal_arm
        
    def update(self, chosen_arm, reward):
        self.pulled_arms.append(chosen_arm)
        self.observed_rewards.append(reward)
        self.counts[chosen_arm] += 1
        self.num_pulls += 1
    
    def find_optimal_arm(self):
        boolean_arm_encoding = np.vstack(self.pulled_arms)
        boolean_arm_encoding = 2*boolean_arm_encoding - 1
        fourier_characteristics = generate_fourier_characteristics(boolean_arm_encoding)
        observed_rewards = np.vstack(self.observed_rewards)
        print(fourier_characteristics.shape)

        estimated_fourier_coeffs = np.zeros((self.N,2**self.N))
        for i in range(self.N):
            print("progress", i)
            unit_n_lasso = LinearRegression(fit_intercept=False)
            masking_vector = self.relevant_subsets[i,:]
            unit_n_lasso.fit(fourier_characteristics[:,masking_vector == 1], observed_rewards[:,i])
            dense_coef = np.zeros(2**self.N)
            dense_coef[masking_vector == 1] = unit_n_lasso.coef_
            estimated_fourier_coeffs[i,:] = dense_coef
        fourier_characters = generate_all_fourier_characteristics(self.N)
        estimated_reward  = estimated_fourier_coeffs @ fourier_characters.T
        estimated_mean_reward = np.mean(estimated_reward, axis=0)
        self.optimal_arm = self.col_index_to_all_arms[np.argmax(estimated_mean_reward)]
        return self.optimal_arm

    def _get_relevant_subsets(self):
        
        N = self.N
        connections = self.connections
        all_subsets = generate_all_subsets(N)
        unit_subsets = np.zeros((N,2**N))

        for index in range(N):
            connections_n = connections.get(index, [])
            filtered_subsets = filter_subsets(connections_n, all_subsets)
            for subset in filtered_subsets:
                subset_index = all_subsets.index(subset)
                unit_subsets[index,subset_index] = 1
        return unit_subsets

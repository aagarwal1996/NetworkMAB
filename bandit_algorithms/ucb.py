import math
import random
import numpy as np


#graph imports
from bandit_algorithms.graph_utils import generate_all_arms


class UCB1:
    def __init__(self, N,num_actions = 2):
        self.n_arms = num_actions**N
        self.all_arms = generate_all_arms(N,num_actions)
        self.counts = {arm: 0 for arm in self.all_arms}  # Counts of pulls for each arm
        self.values = {arm: np.zeros(N) for arm in self.all_arms}  # Average rewards for each arm (initialized as zero vectors)
          # Average rewards for each arm (initialized as zero vectors)
        
    def select_arm(self):
        total_counts = sum(self.counts.values())
        
        if total_counts < self.n_arms:
            return list(self.all_arms)[total_counts] 
        
        ucb_values = {}
        for arm in self.all_arms:
            bonus = math.sqrt((2 * math.log(total_counts)) / float(self.counts[arm]))
            # Calculate the average value for each component
            avg_value = np.mean(self.values[arm])
            ucb_values[arm] = avg_value + bonus

        max_arm = max(ucb_values, key=ucb_values.get)
        return max_arm

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * np.array(reward)
        self.values[chosen_arm] = new_value

# Example usage
if __name__ == "__main__":
    n_arms = 2
    N = 4  # Assuming each reward is a vector of dimension 5
    ucb = UCB1(N, n_arms)
    n_rounds = 1000

    # Simulate pulling arms
    for _ in range(n_rounds):
        chosen_arm = ucb.select_arm()
        print(chosen_arm)
        # Simulate reward for the chosen arm (random example)
        reward = np.random.normal(0, 1, N)  # Replace with actual reward logic
        ucb.update(chosen_arm, reward)

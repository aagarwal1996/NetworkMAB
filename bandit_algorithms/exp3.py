
#generic imports
import numpy as np
import pandas as pd
import pprint, random

#graph imports
from graph_utils import generate_all_arms

class Exp3:
    def __init__(self,N,num_actions = 2,gamma = 0.1,seed = 42):
        '''
        The Exp3 algorithm for graph bandit problems.
        gamma: float, exploration factor
        N: int, number of units
        '''
        self.gamma = gamma
        self.n_arms = num_actions**N
        self.weights = {}
        self.probabilities = {}
        self.all_arms = generate_all_arms(num_actions,N)
        
        for arm in self.all_arms:
            self.weights[arm] = 1
            self.probabilities[arm] = 1/self.n_arms
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
       

    def _update_probabilities(self):
        '''
        Updates the probabilities of the arms
        '''
        tot_weight = np.sum(list(self.weights.values()))
        for arm in self.all_arms:
            self.probabilities[arm] = (1-self.gamma)*(self.weights[arm]/tot_weight) + self.gamma/self.n_arms

    
    def select_arm(self):
        '''
        Selects an arm based on the probabilities
        '''
        arms_list = list(self.probabilities.keys())
        prob_list = list(self.probabilities.values())

        chosen_arm = np.random.choice(len(arms_list),p=prob_list)
        return arms_list[chosen_arm]
                             
    def update(self,chosen_arm,reward):
        '''
        Updates the weights of the arms
        '''
        reward = np.sum(reward)
        estimated_reward = reward/self.probabilities[chosen_arm]
        self.weights[chosen_arm] = self.weights[chosen_arm]*np.exp(self.gamma*estimated_reward/self.n_arms)
        self._update_probabilities()
        return self.weights
        
               
if __name__ == "__main__":
    pass
    # test_graph_generator = low_degree_graph(8,3)
    # test_graph = test_graph_generator.generate_max_degree_graph()
    # bandit_dgp = generate_bandit_data(test_graph)

    
    # #initialize Exp3 object
    # exp3 = Exp3(len(test_graph))
    
    # chosen_arm = exp3.select_arm()
    
    # pprint.pprint(f"chosen arm is {chosen_arm}")

    
    # reward = bandit_dgp.generate_reward(chosen_arm)
    # exp3.update(chosen_arm, reward)
    



        # for arm in self.all_arms:
        #     self.weights[arm] = 1
        #     self.probabilities[arm] = 1/self.n_arms
        #self.probabilities = self._update_probabilities()
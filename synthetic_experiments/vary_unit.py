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
from synthetic_experiments.dgp import generate_noisy_reward, create_graph

if __name__ == "__main__":
    N = 5
    s = 3
    T = 2**(N+1)
    n_arms = 2**N
    connections = create_graph(N, s)
    reward_function = generate_fourier_coefficients(N, connections)
    ucb = UCB1(N)


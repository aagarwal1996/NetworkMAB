import numpy as np
import matplotlib.pyplot as plt
import random


#bandit algorithms
from bandit_algorithms.ucb import UCB1
from bandit_algorithms.exp3 import EXP3

def create_graph(N,s):
    
    
    connections = {}
    for i in range(N):
        # Start with a list containing the index itself
        connections[i] = [i]
        # Add (s-1) unique random indices
        connections[i].extend(random.sample([x for x in range(N) if x != i], s - 1))
    
    return connections

# Example usage
N = 10
s = 3
connections = create_graph(N, s)
for i in range(N):
    print(f"Index {i} is connected to: {connections[i]}")

# generic imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#graph imports
import networkx as nx

#itertools imports
from itertools import product


def generate_all_arms(N,num_actions = 2):
    """
    Generates all possible A^N vectors where A is the number of actions and N is the length of each vector.
    
    Parameters:
    num_actions (int): The number of possible actions (A).
    N (int): The length of each vector.
    
    Returns:
    list: A list of tuples representing all possible A^N vectors.
    """
    return list(product(range(num_actions), repeat=N))

if __name__ == "__main__":
    # Test the generate_all_arms function
    num_actions = 3
    N = 2
    all_arms = generate_all_arms(num_actions, N)
    print(f"All possible {num_actions}^{N} vectors: {all_arms}")


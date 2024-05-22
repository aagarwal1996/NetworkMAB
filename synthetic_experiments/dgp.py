
import numpy as np
import matplotlib.pyplot as plt
import random



#bandit algorithms
from bandit_algorithms.graph_utils import generate_all_arms
from bandit_algorithms.fourier_utils import generate_all_fourier_characteristics
from itertools import chain, combinations
import random
import numpy as np



def print_matrix(matrix, all_subsets):
    """
    Print the matrix with subset indices as column headers
    """
    subset_labels = ['{' + ','.join(map(str, subset)) + '}' for subset in all_subsets]
    
    # Print header
    print(f"{'Index':<6} {' '.join([f'{label:<10}' for label in subset_labels])}")
    print('-' * (6 + 11 * len(subset_labels)))

    # Print matrix rows
    for i, row in enumerate(matrix):
        row_str = ' '.join([f'{value:<10.4f}' if value != 0 else ' ' * 10 for value in row])
        print(f"{i:<6} {row_str}")


def create_graph(N,s):
    
    connections = {}
    for i in range(N):
        # Start with a list containing the index itself
        connections[i] = [i]
        # Add (s-1) unique random indices
        connections[i].extend(random.sample([x for x in range(N) if x != i], s - 1))
    
    return connections


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

def generate_fourier_coefficients(N,s,connections_dict,seed = 42):
    """
    Generate a matrix of size N by 2^N with random numbers for the relevant subsets
    """
    all_subsets = generate_all_subsets(N)
    num_subsets = len(all_subsets)

    random.seed(seed)
    # Initialize the matrix with zeros
    matrix = np.zeros((N, num_subsets))
    
    for index in range(N):
        connections = connections_dict.get(index, [])
        filtered_subsets = filter_subsets(connections, all_subsets)
        for subset in filtered_subsets:
            subset_index = all_subsets.index(subset)
            matrix[index][subset_index] = random.uniform(0, 1/s)
    
    return matrix, all_subsets

def reward_function(N,s,connections_dict):
    """
    Calculate the reward for a given vector x
    """
    fourier_coeffs,all_subsets  = generate_fourier_coefficients(N,s,connections_dict) #N times 2^N
    fourier_characteristics = generate_all_fourier_characteristics(N) #2^N times 2^N
    print("fourier_characteristics", fourier_characteristics.shape)
    print("fourier_coeffs", fourier_coeffs.shape)
    reward  = fourier_coeffs @ fourier_characteristics.T
    return reward

def generate_noisy_reward(reward, noise_scale = 1.0):
    return reward + np.random.normal(0, noise_scale, len(reward))


if __name__ == "__main__":
    
    # Example usage
    N = 3
    s = 2
    random_graph = create_graph(N,s)

    print(f"Random Graph: {random_graph}")

    matrix, all_subsets = generate_fourier_coefficients(N, s,random_graph)


    print("Generated Matrix:")
    print_matrix(matrix, all_subsets)

    print("")
    print("")

    reward = reward_function(N, s,random_graph)
    print(reward.shape)

import numpy as np
from itertools import chain, combinations

from bandit_algorithms.graph_utils import generate_all_arms


def generate_all_subsets(N):
    """
    Generate all subsets of the set [0, 1, ..., N-1]
    """
    return list(chain.from_iterable(combinations(range(N), r) for r in range(N+1)))

def calculate_subset_products(x, all_subsets):
    """
    Calculate the product of the elements of x for each subset in all_subsets.
    
    Parameters:
    x (list): Input vector of length N with each entry being -1 or 1.
    all_subsets (list): List of all subsets of [0, 1, ..., N-1].
    
    Returns:
    list: A list of products corresponding to each subset.
    """
    subset_products = []
    for subset in all_subsets:
        product = 1
        for index in subset:
            product *= x[index]
        subset_products.append(product)
    return subset_products

# Example usage
N = 3
x = [1, -1, 1]  # Example vector of length N
all_subsets = generate_all_subsets(N)
subset_products = calculate_subset_products(x, all_subsets)


def generate_fourier_characteristics(X):
    """
    Generate the output matrix where each row is the subset products of the corresponding row in X.
    
    Parameters:
    X (ndarray): Input matrix with M rows and N columns, where each entry is -1 or 1.
    
    Returns:
    ndarray: Output matrix with M rows and 2^N columns.
    """
    M, N = X.shape
    all_subsets = generate_all_subsets(N)
    num_subsets = len(all_subsets)
    
    # Initialize the output matrix
    output_matrix = np.zeros((M, num_subsets))
    
    for i in range(M):
        x = X[i]
        subset_products = calculate_subset_products(x, all_subsets)
        output_matrix[i] = subset_products
    
    return output_matrix

def generate_all_fourier_characteristics(N):
    all_arms = generate_all_arms(N)
    all_arms_boolean_encoding = [2*np.array(arm) - 1 for arm in all_arms]
    all_arms_boolean_encoding = np.vstack(all_arms_boolean_encoding)
    return generate_fourier_characteristics(all_arms_boolean_encoding)
    



# # Example usage
# M = 3
# N = 3
# X = np.array([[1, -1, 1],
#               [-1, 1, -1],
#               [1, 1, -1]])  # Example matrix with M rows and N columns

# output_matrix = generate_fourier_characteristics(X)

# print("Input Matrix (X):")
# print(X)
# print("\nOutput Matrix:")
# print(output_matrix)
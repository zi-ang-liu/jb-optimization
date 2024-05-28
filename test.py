import numpy as np


def initialize_population(n, m):
    return np.array([np.random.permutation(n) for _ in range(m)])


n = 10  # population size
m = 5  # number of variables
population = initialize_population(n, m)
print(population)

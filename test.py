import numpy as np


def roulette_wheel_selection(population, fitness):
    n = len(population)
    p = fitness / np.sum(fitness)
    idx = np.random.choice(n, p=p)
    return population[idx]


population = np.random.rand(10, 2)
fitness = np.random.rand(10)
selected = roulette_wheel_selection(population, fitness)

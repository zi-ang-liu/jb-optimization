# Genetic Algorithm

```{prf:algorithm} Genetic Algorithm
:label: GA

**Inputs** population size $N$

**Output** best individual $\mathbf{x}_{best}$

1. Initialize population $P$ with $N$ individuals
2. While not converged:
    1. $P' \leftarrow \emptyset$
    2. While $|P'| < n$:
        1. Select parents $\mathbf{x}_1, \mathbf{x}_2$ from $P$
        2. Generate offspring $\mathbf{x}_1', \mathbf{x}_2'$ by crossover
        3. Mutate $\mathbf{x}_1'$ and $\mathbf{x}_2'$
        4. $P' \leftarrow P' \cup \{\mathbf{x}_1', \mathbf{x}_2'\}$
    3. $P \leftarrow P'$
3. $\mathbf{x}_{best} \leftarrow \arg\min_{\mathbf{x} \in P} f(\mathbf{x})$
```

## Overview

Genetic algorithms (GA) are a representive type of evolutionary algorithms. They are based on the idea of natural selection and evolution. In GA, the solutions are represented as chromosomes. The algorithm starts with a population of initial solutions and then evolves them to find the best solution. During the evolution, the chromosomes with better fitness are more likely to be selected for reproduction.

GA is a popular optimization algorithm and is widely used in various problems. It is particularly useful for problems binary variables and sequencing problems. By using special operations, GA can also be applied to continuous and integer optimization problems.

GA has four main operators: initialization, selection, crossover, and mutation.

## Initialization

GA is a population-based algorithm. It starts with a population of chromosomes. A common way to initialize the population is to randomly generate chromosomes within the search space.

## Selection

The selection operator is used to select the chromosomes for reproduction. The basic idea of these methods is the chromosomes with better fitness are more likely to be selected. The following are the most common selection methods:

- Tournament selection
- Roulette wheel selection
- Rank-based selection

### Tournament Selection

In tournament selection, a subset of the population is randomly selected and the chromosome with the best fitness is chosen for reproduction. The size of the subset is called the tournament size. The tournament size is a parameter of the algorithm and can be tuned.

### Roulette Wheel Selection

In roulette wheel selection, the probability of selecting a chromosome is related to its fitness. For minimization problems, we can first transfer the fitness of chromosome $i$ to 

$$
fitness(\mathbf{x}_i) \leftarrow \max_{j=1}^n f(\mathbf{x}_j) - f(\mathbf{x}_i)
$$

where $fitness(\mathbf{x}_i)$ is the fitness of chromosome $i$ and $n$ is the population size. Then, we can calculate the probability of selecting chromosome $i$ as 

$$p_i = \frac{f(\mathbf{x}_i)}{\sum_{j=1}^n f(\mathbf{x}_j)}$$

The chromosome is selected by generating a random number between 0 and 1 (spinning the roulette wheel).

```python
import numpy as np

def roulette_wheel_selection(population, fitness):
    n = len(population)
    p = fitness / np.sum(fitness)
    idx = np.random.choice(n, p=p)
    return population[idx]

population = np.random.rand(10, 2)
fitness = np.random.rand(10)
selected = roulette_wheel_selection(population, fitness)
```

### Rank-based Selection


## Crossover

The crossover operator is used to create new chromosomes by combining the information of two parent chromosomes. For different types of problems, there are different crossover methods. For binary variables, the most common crossover methods are *one-point crossover* and *two-point crossover*. The order-crossover (OX) can be used for sequencing problems. For continuous variables, the simulated binary crossover (SBX) is a popular method.

### One-Point and Two-Point Crossover

In one-point crossover, a random point is selected and the information of the two parent chromosomes is exchanged.

In two-point crossover, two random points are selected and the information between the two points is exchanged.

```python
import numpy as np


def one_point_crossover(x1, x2):
    n = len(x1)
    point = np.random.randint(1, n)
    y1 = np.concatenate([x1[:point], x2[point:]])
    y2 = np.concatenate([x2[:point], x1[point:]])
    return y1, y2


def two_point_crossover(x1, x2):
    n = len(x1)
    points = np.sort(np.random.choice(n, 2, replace=False))
    y1 = np.concatenate([x1[: points[0]], x2[points[0] : points[1]], x1[points[1] :]])
    y2 = np.concatenate([x2[: points[0]], x1[points[0] : points[1]], x2[points[1] :]])
    return y1, y2


x1 = np.array([1, 1, 1, 1, 1])
x2 = np.array([0, 0, 0, 0, 0])
y1, y2 = one_point_crossover(x1, x2)
print(y1, y2)

y1, y2 = two_point_crossover(x1, x2)
print(y1, y2)
```

### Order Crossover (OX)

In sequence problems, it is important to maintain the chromosome as a valid sequence, which should be a permutation of the numbers from 1 to $n$. The order-crossover (OX) is a method that can maintain the validity of the sequence.

The order-crossover (OX) operator works as follows:

1. Randomly select two points in the chromosome.
2. Copy the information between the two points from the first parent to the offspring.
3. Fill the remaining positions in the offspring with the remaining numbers from the second parent, preserving the order.

Given two parent chromosomes:

```bash
x1 = [3, 1, 2, 5, 4]
x2 = [2, 4, 3, 1, 5]
```

The order-crossover operator first selects two points, say 1 and 3. Then, it copies the information between the two points from the first parent to the offspring:

```bash
y1 = [?, 1, 2, ?, ?]
y2 = [?, 4, 3, ?, ?]
```

Next, it fills the remaining positions in the offspring with the remaining numbers from the second parent, preserving the order:

```bash
y1 = [4, 1, 2, 3, 5]
y2 = [1, 4, 3, 2, 5]
```

```python
import numpy as np


def order_crossover(x1, x2):
    n = len(x1)
    points = np.sort(np.random.choice(n, 2, replace=False))
    # Reserve elements between the two points
    reserved_x1 = x1[points[0] : points[1]]
    reserved_x2 = x2[points[0] : points[1]]
    # Remove reserved elements from parents while keeping the order
    y1 = np.array([i for i in x2 if i not in reserved_x1])
    y2 = np.array([i for i in x1 if i not in reserved_x2])
    # Insert reserved elements back to the parents
    y1 = np.insert(y1, points[0], reserved_x1)
    y2 = np.insert(y2, points[0], reserved_x2)
    return y1, y2


x1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
x2 = np.array([9, 3, 7, 8, 2, 6, 5, 1, 4])
y1, y2 = order_crossover(x1, x2)
print(y1)
print(y2)
```


### Simulated Binary Crossover (SBX)

In simulated binary crossover, the offspring is generated by 

$$
\mathbf{x}_1' = 0.5[(1+\beta)\mathbf{x}_1 + (1-\beta)\mathbf{x}_2]
$$

$$
\mathbf{x}_2' = 0.5[(1-\beta)\mathbf{x}_1 + (1+\beta)\mathbf{x}_2]
$$

where $\mathbf{x}_1$ and $\mathbf{x}_2$ are the parent chromosomes and $\beta$ is a parameter.

## Mutation

Using only crossover may not be enough to explore the search space. Mutation is used to introduce diversity in the population. Mutation randomly changes the information in the chromosome. The mutation rate is a parameter that controls the probability of mutation. For a $m$-bit chromosome, the mutation rate is usually set to $1/m$ {cite:p}`Kochenderfer2019-qw`.

The most common mutation methods are:

- Bit-flip mutation
- Swap mutation

### Bit-Flip Mutation

In bit-flip mutation, one or more bits in the chromosome are randomly selected and flipped.

In the following code, `n_points` is the number of bits to flip.

```python
import numpy as np


def bit_flip_mutation(x, n_points=1):
    n = len(x)
    idx = np.random.choice(n, n_points, replace=False)
    x[idx] = 1 - x[idx]
    return x


x = np.array([1, 0, 1, 1, 1])
x = bit_flip_mutation(x)
print(x)
```

### Swap Mutation

In swap mutation, two positions in the chromosome are randomly selected and the information at the two positions is swapped.

```python
import numpy as np


def swap_mutation(x):
    n = len(x)
    idx = np.random.choice(n, 2, replace=False)
    x[idx[0]], x[idx[1]] = x[idx[1]], x[idx[0]]
    return x


x = np.array([1, 2, 3, 4, 5])
x = swap_mutation(x)
print(x)
```

## Implementation

### Traveling Salesman Problem

The traveling salesman problem (TSP) is a classic optimization problem. Consider a salesman who wants to visit $n$ cities. The objective is to find the shortest path that visits each city exactly once and returns to the starting city.

Let $N = \{1, 2, \ldots, n\}$ be the set of cities, and let $c_{i,j}$ be the distance between city $i$ and city $j$. We use $x_{i,j}$ to represent the decision variable, which is 1 if the salesman goes from city $i$ to city $j$, and 0 otherwise. The TSP can be formulated as the following optimization problem:

$$
\begin{aligned}
\min \quad & \sum_{i \in N} \sum_{j \in N} c_{i,j} x_{i,j} \\
\text{s.t.} \quad & \sum_{i \in N} x_{i,j} = 1, \quad \forall j \in N \\
& \sum_{j \in N} x_{i,j} = 1, \quad \forall i \in N \\
& \sum_{i \in S} \sum_{j \in S} x_{i,j} \le |S| - 1, \quad \forall S \subset N, 2 \le |S| \le n-1 \\
& x_{i,j} \in \{0, 1\}, \quad \forall i, j \in N
\end{aligned}
$$

In this formulation, the objective is to minimize the total distance traveled. The first two constraints ensure that each city is visited exactly once. The third set of constraints are the subtour elimination constraints, which prevent the existence of sub-tours. 

One way to solve the TSP is to use a genetic algorithm. Instead of using a binary representation, we can use a permutation representation $\mathbf{x} = [x_1, x_2, \ldots, x_n]$, where $x_i$ is the index of a city. This representation ensures that each city is visited exactly once and there are no sub-tours.

The following code shows how to solve the TSP using a genetic algorithm.

```python
import numpy as np
import matplotlib.pyplot as plt


class TSP:
    """travelling salesman problem"""

    def __init__(self, n_cities, coordinates):
        self.n_cities = n_cities
        if coordinates is None:
            self.coordinates = np.random.rand(n_cities, 2)
        else:
            self.coordinates = coordinates
        self.get_distance_matrix()

    def get_distance_matrix(self):
        self.distance_matrix = np.zeros((n_cities, n_cities))
        for i in range(n_cities):
            for j in range(n_cities):
                self.distance_matrix[i, j] = np.linalg.norm(
                    self.coordinates[i] - self.coordinates[j]
                )

    def evaluate(self, x):
        # evaluate a path x
        assert len(x) == self.n_cities
        distance = 0
        for i in range(self.n_cities):
            distance += self.distance_matrix[x[i - 1], x[i]]
        return distance


class GeneticAlgorithm:

    def __init__(self, tsp, n_pop=20, n_gen=1000, mut_rate=0.05, elitism=True):
        self.tsp = tsp
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.mut_rate = mut_rate
        self.elitism = elitism

    def permutation_init(self):
        # initialize a population
        self.populations = np.array(
            [np.random.permutation(self.tsp.n_cities) for _ in range(self.n_pop)],
            dtype=int,
        )

    def evaluate(self):
        # evaluate the population
        self.fitness = np.array([self.tsp.evaluate(x) for x in self.populations])

    def roulette_wheel_selection(self):
        reverse_fitness = np.max(self.fitness) - self.fitness + 1e-6
        prob = reverse_fitness / np.sum(reverse_fitness)
        idx = np.random.choice(self.n_pop, 2, p=prob)
        parent1, parent2 = self.populations[idx]
        return parent1, parent2

    def order_crossover(self, x1, x2):
        n = len(x1)
        points = np.sort(np.random.choice(n, 2, replace=False))
        # Reserve elements between the two points
        reserved_x1 = x1[points[0] : points[1]]
        reserved_x2 = x2[points[0] : points[1]]
        # Remove reserved elements from parents while keeping the order
        y1 = np.array([i for i in x2 if i not in reserved_x1])
        y2 = np.array([i for i in x1 if i not in reserved_x2])
        # Insert reserved elements back to the parents
        y1 = np.insert(y1, points[0], reserved_x1)
        y2 = np.insert(y2, points[0], reserved_x2)
        return y1, y2

    def swap_mutation(self, x):
        n = len(x)
        idx = np.random.choice(n, 2, replace=False)
        y = x.copy()
        y[idx[0]], y[idx[1]] = y[idx[1]], y[idx[0]]
        return y

    def optimize(self):
        # initialize a population and evaluate it
        self.permutation_init()
        self.evaluate()

        # store the best path and its distance
        best_idx = np.argmin(self.fitness)
        self.best_path = self.populations[best_idx]
        best_distance = self.fitness[best_idx]

        # initialize a list to store the best distance in each generation
        self.best_distances = np.zeros(self.n_gen)
        self.best_distances[0] = best_distance

        # main loop
        for gen in range(1, self.n_gen):

            # create a new population
            new_populations = np.zeros_like(self.populations)

            # iterate until the new population is filled
            for i in range(0, self.n_pop, 2):

                # select parents
                parent1, parent2 = self.roulette_wheel_selection()

                # crossover
                child1, child2 = self.order_crossover(parent1, parent2)

                # mutation
                if np.random.rand() < self.mut_rate:
                    child1 = self.swap_mutation(child1)
                if np.random.rand() < self.mut_rate:
                    child2 = self.swap_mutation(child2)

                # add children to the new population
                new_populations[i] = child1
                new_populations[i + 1] = child2

            # replace the old population with the new population
            self.populations = new_populations
            self.evaluate()

            # elitism: replace the worst path with the best path
            if self.elitism:
                worst_idx = np.argmax(self.fitness)
                self.populations[worst_idx] = self.best_path
                self.fitness[worst_idx] = best_distance

            # update the best path and its distance
            best_idx = np.argmin(self.fitness)
            self.best_path = self.populations[best_idx]
            best_distance = self.fitness[best_idx]
            self.best_distances[gen] = best_distance

            print(f"Generation {gen}: best distance = {best_distance}")

        return self.best_path, self.best_distances


if __name__ == "__main__":

    # Random TSP instance
    n_cities = 15
    np.random.seed(0)
    tsp = TSP(n_cities, None)

    # Genetic Algorithm
    ga = GeneticAlgorithm(tsp)
    best_path, best_distances = ga.optimize()

    # Plot the best path
    plt.figure()
    plt.plot(tsp.coordinates[:, 0], tsp.coordinates[:, 1], "o")
    for i in range(n_cities):
        plt.text(tsp.coordinates[i, 0], tsp.coordinates[i, 1], str(i), fontsize=12)
    for i in range(n_cities):
        plt.plot(
            [tsp.coordinates[best_path[i - 1], 0], tsp.coordinates[best_path[i], 0]],
            [tsp.coordinates[best_path[i - 1], 1], tsp.coordinates[best_path[i], 1]],
            "r-",
        )
    plt.title("Best path")
    plt.savefig("ga_tsp_best_path.svg")

    # Plot the convergence curve
    plt.figure()
    plt.plot(best_distances)
    plt.xlabel("Generation")
    plt.ylabel("Distance")
    plt.title("Convergence curve")
    plt.savefig("ga_tsp_convergence_curve.svg")
```

The results are shown in the following figures:

```{figure} ../images/genetic_algorithm/ga_tsp_best_path.svg
---
width: 400px
name: ga_tsp_best_path
---
Best path found by genetic algorithm for TSP
```

```{figure} ../images/genetic_algorithm/ga_tsp_convergence_curve.svg
---
width: 400px
name: ga_tsp_convergence_curve
---
Convergence curve of genetic algorithm for TSP
```

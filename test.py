import numpy as np
import matplotlib.pyplot as plt


class TSP:
    """travelling salesman problem

    Attributes
    ----------
    distance_matrix (np.ndarray): distance matrix
    n_cities (int): number of cities

    Methods
    -------
    evaluate(x)
        evaluate the total distance of a path x
    """

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

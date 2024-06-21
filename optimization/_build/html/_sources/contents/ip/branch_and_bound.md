# General Branch-and-Bound

The branch-and-bound method is a general algorithm for solving combinatorial optimization problems by intelligently enumerating all the feasible points. 

The *branch* refers to the process of partitioning the solution space. The *bound* refers to lower bounds that are used to construct a proof of optimality.

One of the important applications of branch-and-bound is the ILP (Integer Linear Programming) problem. 

## ILP Problem


## General Branch-and-Bound Algorithm

```{prf:algorithm} General Branch-and-Bound
:label: bb-algorithm

1. Initialize current best node $v^*$
2. UB $\leftarrow \infty$
3. Create a queue $Q$
4. $Q.\text{put}(root)$
5. While $Q$ is not empty:
    1. $k \leftarrow$ $Q.\text{get}()$
    2. generate children of node $k$, child $i \in \mathcal{S}(k)$
    3. generate lower bounds $LB_i$ for each child $i$
    4. For each $i \in \mathcal{S}(k)$:
        1. If $LB_i \geq$ UB, kill child $i$
        2. Else if child $i$ is a leaf node:
            1. UB $\leftarrow$ value of child $i$
            2. $v^* \leftarrow i$
        3. Else:
            1. $Q.\text{put}(i)$
```


```python
'''
Branch and Bound algorithm for the shortest path problem

We branch from a node with the lowest lower bound using a priority queue.
'''

from queue import PriorityQueue
import networkx as nx

# shortest path problem
class ShortestPathProblem:
    def __init__(self, graph):
        self.graph = graph
    
    def branch(self, node):
        children = []
        weights = []
        # Generate children of the node
        for child in self.graph.successors(node):
            children.append(child)
            weights.append(self.graph[node][child]['weight'])

        return children, weights

# Node class in the branch and bound algorithm
class Node:
    def __init__(self, node_idx, lower_bound):
        self.node_idx = node_idx
        self.lower_bound = lower_bound
        self.solution = []

    def update_solution(self, parent_solution):
        self.solution = parent_solution.copy()
        self.solution.append(self.node_idx)

    def __lt__(self, other):
        return self.lower_bound < other.lower_bound

def branch_and_bound(problem):
    
    U = float('inf')
    current_best_solution = None
    current_best_value = None

    priority_queue = PriorityQueue(maxsize=0)

    # Initialize the priority queue with the root node
    root_node = Node(0, 0)
    root_node.solution.append(0)
    priority_queue.put((0, root_node))

    while not priority_queue.empty():

        _, node = priority_queue.get()

        # generate children of the node
        children, weights = problem.branch(node.node_idx)

        for child, weight in zip(children, weights):
            z = node.lower_bound + weight

            if z >= U:
                continue
            elif child == 10:
                U = z
                current_best_value = z
                current_best_solution = node.solution.copy()
            else:
                new_node = Node(child, z)
                new_node.update_solution(node.solution)
                priority_queue.put((z, new_node))

    return current_best_value, current_best_solution


if __name__ == '__main__':

    '''
    Define the shortest-path problem.
    Example from the book:
    Papadimitriou, Christos H., and Kenneth Steiglitz. 1998. Combinatorial Optimization: Algorithms and Complexity. Courier Corporation.    
    '''
    G = nx.DiGraph()

    # Add nodes
    G.add_nodes_from(range(0, 11))

    # Add edges
    G.add_edge(0, 1, weight=2)
    G.add_edge(0, 2, weight=3)
    G.add_edge(0, 3, weight=4)
    G.add_edge(1, 4, weight=7)
    G.add_edge(1, 5, weight=2)
    G.add_edge(1, 2, weight=3)
    G.add_edge(2, 5, weight=9)
    G.add_edge(2, 6, weight=2)
    G.add_edge(3, 6, weight=2)
    G.add_edge(4, 7, weight=1)
    G.add_edge(4, 8, weight=2)
    G.add_edge(5, 8, weight=3)
    G.add_edge(5, 6, weight=1)
    G.add_edge(6, 8, weight=5)
    G.add_edge(6, 9, weight=1)
    G.add_edge(7, 10, weight=4)
    G.add_edge(8, 10, weight=4)
    G.add_edge(9, 10, weight=2)
    G.add_edge(9, 8, weight=2)

    spp = ShortestPathProblem(G)


    value, solution = branch_and_bound(spp)
    print('Shortest path value:', value)
    print('Shortest path:', solution)
```

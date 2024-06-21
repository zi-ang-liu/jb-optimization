# Dijkstra's Algorithm

```{prf:algorithm} Dijkstra's Algorithm
:label: dijkstra-algorithm

**Input**: A graph $G = (V, E)$ and a source node $s$   
**Output**: The shortest path from $s$ to all $v \in V$

1. $W \leftarrow {s}$, $p(s) \leftarrow 0$
2. **for** $y \in V \setminus {s}$ **do**: $p(y) \leftarrow w_{sy}$
3. **while** $W \neq V$ **do**:
    1. $x \leftarrow \arg \min_{y \in V \setminus W} p(y)$
    2. $W \leftarrow W \cup {x}$
    3. **for** $y \in V \setminus W$ **do**:
        1. $p(y) \leftarrow \min(p(y), p(x) + w_{xy})$
```

```python
"""
Dijkstra's algorithm for the shortest path problem
"""

import networkx as nx


def dijkstra(graph, source):
    # Initialize the distance from the source node to all other nodes
    p = {}
    p[source] = 0
    for node in graph.nodes():
        if node != source and graph.has_edge(source, node):
            p[node] = graph[source][node]["weight"]
        elif node != source:
            p[node] = float("inf")

    # initialize the weitgh of the edges, if (x, y) is not in the graph, the weight is infinity
    weight = {}
    for x in graph.nodes():
        for y in graph.nodes():
            if graph.has_edge(x, y):
                weight[(x, y)] = graph[x][y]["weight"]
            else:
                weight[(x, y)] = float("inf")

    # Initialize the set of visited nodes
    W = set()
    W.add(source)

    # Main loop
    while W != set(graph.nodes()):
        # Find the node with the smallest distance
        x = min(
            (node for node in graph.nodes() if node not in W), key=lambda node: p[node]
        )
        W.add(x)
        for y in graph.nodes():
            if y not in W:
                p[y] = min(p[y], p[x] + weight[(x, y)])

    return p


if __name__ == "__main__":
    # Create a graph
    graph = nx.DiGraph()

    # Example 1 p. 130
    graph.add_weighted_edges_from(
        [
            (0, 1, 2),
            (0, 2, 1),
            (1, 2, 3),
            (1, 3, 3),
            (2, 4, 1),
            (4, 3, 2),
            (3, 5, 2),
            (4, 5, 5),
        ]
    )

    # Example 2
    # graph.add_weighted_edges_from(
    #     [
    #         (0, 1, 2),
    #         (0, 2, 3),
    #         (0, 3, 4),
    #         (1, 4, 7),
    #         (1, 5, 2),
    #         (1, 2, 3),
    #         (2, 5, 9),
    #         (2, 6, 2),
    #         (3, 6, 2),
    #         (4, 7, 1),
    #         (4, 8, 2),
    #         (5, 8, 3),
    #         (5, 6, 1),
    #         (6, 8, 5),
    #         (6, 9, 1),
    #         (7, 10, 4),
    #         (8, 10, 4),
    #         (9, 10, 2),
    #         (9, 8, 2),
    #     ]
    # )

    # Compute the shortest path from node 0
    print(dijkstra(graph, 0))
```
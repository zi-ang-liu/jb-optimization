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
'''
Dijkstra's algorithm for the shortest path problem
'''
import networkx as nx

def dijkstra(graph, source):
    # Initialize the distance from the source node to all other nodes
    distance[source] = 0
    for node in graph.nodes():
        if node != source and graph.has_edge(source, node):
            distance[node] = graph[source][node]['weight']
        else:
            distance[node] = float('inf')
    
    # Initialize the set of visited nodes
    W = set()
    W.add(source)

    # Main loop
    while W != set(graph.nodes()):
        # Find the node with the smallest distance
        x = min((node for node in graph.nodes() if node not in W), key=lambda node: distance[node])
        W.add(x)
        for y in graph.nodes():
            if y not in W:
                distance[y] = min(distance[y], distance[x] + graph[x][y]['weight'])
```
# Dijkstra's Algorithm

```{prf:algorithm} Dijkstra's Algorithm
:label: dijkstra-algorithm

**Input**: A graph $G = (V, E)$ and a source node $s$

**Output**: The shortest path from $s$ to all $v \in V$

1. $W \leftarrow {s}$, $p(s) \leftarrow 0$
2. **for** $y \in V \setminus {s}$ **do**: $p(y) \leftarrow w_{sy}$$
3. **while** $W \neq V$ **do**:
    1. $x \leftarrow \arg \min_{y \in V \setminus W} p(y)$
    2. $W \leftarrow W \cup {x}$
    3. **for** $y \in V \setminus W$ **do**:
        1. $p(y) \leftarrow \min(p(y), p(x) + w_{xy})$
```

# Particle Swarm Optimization

```{prf:algorithm} PSO
:label: pso

**Inputs:** population size $n$, dimension $d$, inertia weight $w$, acceleration coefficients $c_1, c_2$   
**Output:** best individual $\mathbf{g}$


1. Initialize $X$
2. **while** stopping criterion not met **do**
    1. **for** $i = 1, 2, \ldots, n$ **do**
        1. $V_i \leftarrow wV_i + c_1\mathbf{R}^\intercal (\mathbf{p}_i - X_i) + c_2\mathbf{R}^\intercal (\mathbf{g} - X_i)$
        2. $X_i \leftarrow X_i + V_i$
        3. **if** $f(X_i) < f(\mathbf{p}_i)$ **then**
            1. $\mathbf{p}_i \leftarrow X_i$
            2. **if** $f(X_i) < f(\mathbf{g})$ **then**
                1. $\mathbf{g} \leftarrow X_i$
```

## Notation

- $n$: population size
- $d$: dimension of particles
- $X$: population of particles, $X \in \mathbb{R}^{N \times D}$
- $X_i$: $i$-th particle
- $x_{i, j}$: $j$-th dimension of $X_i$
- $V$: velocity of particles
- $V_i$: velocity of $i$-th particle
- $v_{i, j}$: $j$-th dimension of $V_i$
- $w$: inertia weight
- $c_1, c_2$: acceleration coefficients
- $\mathbf{R}$: random vector in $[0, 1]^D$
- $\mathbf{p}_i$: personal best of particle $i$
- $\mathbf{g}$: global best of particles


## Update Equations

$$
V_i \leftarrow wV_i + c_1\mathbf{R}^\intercal (\mathbf{p}_i - X_i) + c_2\mathbf{R}^\intercal (\mathbf{g} - X_i)
$$

$$
X_i \leftarrow X_i + V_i
$$
# Forms of Linear Programming Problems

## General Form Problems

A linear programming problem of the following form is said to be in *general form*:

$$
\begin{align*}
\text{minimize} & \quad \mathbf{c}' \mathbf{x} \\
\text{subject to} & \quad \mathbf{a_i}' \mathbf{x} \leq b_i, \quad i \in M_1 \\
& \quad \mathbf{a_i}' \mathbf{x} = b_i, \quad i \in M_2 \\
& \quad \mathbf{a_i}' \mathbf{x} \geq b_i, \quad i \in M_3 \\
& \quad x_j \geq 0, \quad j \in N_1 \\
& \quad x_j \leq 0, \quad j \in N_2 \\
\end{align*}
$$

In this form, $\mathbf{c}$ is an $n$-dimensional vector, $\mathbf{x}$ is an $n$-dimensional vector of variables, $\mathbf{a_i}$ is an $n$-dimensional vector, $b_i$ is a scalar, and $M_1, M_2, M_3, N_1, N_2$ are sets of indices.

In detail, $M_1$ and $M_3$ are sets of indices for which the corresponding constraints are inequalities, $M_2$ is a set of indices for which the corresponding constraints are equalities, $N_1$ is a set of indices for which the corresponding variables are nonnegative, and $N_2$ is a set of indices for which the corresponding variables are nonpositive.

We also call $c$ the *cost vector*, and $x_1, x_2, \ldots, x_n$ the *decision variables*.

## Canonical Form Problems

The *canonical form* of a linear programming problem is given by:

$$
\begin{align*}
\text{minimize} & \quad \mathbf{c}' \mathbf{x} \\
\text{subject to} & \quad \mathbf{A} \mathbf{x} \geq \mathbf{b} \\
& \quad \mathbf{x} \geq \mathbf{0}
\end{align*}
$$

where $\mathbf{A}$ is an $m \times n$ matrix, $\mathbf{b}$ is an $m$-dimensional vector.

## Standard Form Problems

The *standard form* of a linear programming problem is given by:

$$
\begin{align*}
\text{minimize} & \quad \mathbf{c}' \mathbf{x} \\
\text{subject to} & \quad \mathbf{A} \mathbf{x} = \mathbf{b} \\
& \quad \mathbf{x} \geq \mathbf{0}
\end{align*}
$$

## Reducing General Form to Standard Form

To reduce a linear programming problem in general form to standard form, we need to (1) eliminate free variable, and (2)convert inequalities to equalities.

### Eliminating Free Variables

To eliminate free variables, we replace each free variable $x_j$ with the difference of two variables $x_j = x_j^+ - x_j^-$, where $x_j^+$ and $x_j^-$ are nonnegative.

### Converting Inequalities to Equalities

For an inequality $\mathbf{a}_i' \mathbf{x} \leq b_i$, we introduce a slack variable $s_i \geq 0$ such that $\mathbf{a}_i' \mathbf{x} + s_i = b_i$.

Similarly, for an inequality $\mathbf{a}_i' \mathbf{x} \geq b_i$, we introduce a surplus variable $s_i \geq 0$ such that $\mathbf{a}_i' \mathbf{x} - s_i = b_i$.

## Diet Problem

The *diet problem* is one of the first problems to be formulated as a linear programming problem. There are $n$ foods and $m$ nutrients that we need to consider.

### Notation

- $a_{i,j}$: amount of nutrient $i$ in food $j$.
- $b_i$: minimum amount of nutrient $i$.
- $c_j$: cost of food $j$.
- $x_j$: amount of food $j$ to buy.

### Formulation

Such problem can be formulated as a linear programming problem in the following way:

$$
\begin{align*}
\text{minimize} & \quad \mathbf{c}' \mathbf{x} \\
\text{subject to} & \quad \mathbf{A} \mathbf{x} \geq \mathbf{b} \\
& \quad \mathbf{x} \geq \mathbf{0}
\end{align*}
$$

The *objective function* is to minimize the cost of the food, which is given by $\mathbf{c}' \mathbf{x}$.

The *constraints* are that the amount of nutrients in the food should be greater than or equal to the minimum amount of nutrients required, which is given by $\mathbf{A} \mathbf{x} \geq \mathbf{b}$. Also, the amount of food should be nonnegative, which is given by $\mathbf{x} \geq \mathbf{0}$.

## Summary

- Linear programming problems can be formulated in general, canonical, or standard form.
- These forms can be converted into each other.
- The standard form is convenient for developing algorithms.


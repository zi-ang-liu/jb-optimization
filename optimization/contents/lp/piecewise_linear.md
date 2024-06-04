# Piecewise Linear Convex Functions

A *piecewise linear convex function* is a function that is defined by a set of linear functions, each defined over a different region of the domain. The piecewise linear convex function $f: \mathbb{R}^n \to \mathbb{R}$ can be written as:

$$
f(\mathbf{x}) = \max_{i=1, \dots, m} (\mathbf{a}_i' \mathbf{x} + d_i)
$$

A special case of piecewise linear convex functions is the absolute value function, which is defined by $f(x) = |x| = \max(x, -x)$.

## Piecewise linear convex constraints

Piecewise linear convex constraints are constraints that are defined by a set of piecewise linear convex functions. For example:

$$
\begin{align*}
\textrm{minimize} \quad & \mathbf{c}' \mathbf{x} \\
\textrm{subject to} \quad & \max_{i=1, \dots, m} (\mathbf{f}_i' \mathbf{x} + d_i) \leq b \\
& \mathbf{A} \mathbf{x} \geq \mathbf{b} \\
& \mathbf{x} \geq 0 
\end{align*}
$$

In this case, the constraint $\max_{i=1, \dots, m} (\mathbf{a}_i' \mathbf{x} + d_i) \leq b$ is a piecewise linear convex constraint. Such constraints are equivalent to a set of linear constraints as follows:

$$
\mathbf{f}_i' \mathbf{x} + d_i \leq b \quad \forall i = 1, \dots, m
$$

Hence, piecewise linear convex constraints can be reformulated as a set of linear constraints. 

## Piecewise linear convex objective functions

Piecewise linear convex objective functions are objective functions that are defined by a set of piecewise linear convex functions. Consider the following optimization problem:

$$
\begin{align*}
\textrm{minimize} \quad & \max_{i=1, \dots, m} (\mathbf{c}_i' \mathbf{x} + d_i) \\
\textrm{subject to} \quad & \mathbf{A} \mathbf{x} \geq \mathbf{b} \\
& \mathbf{x} \geq 0
\end{align*}
$$

The objective function of this optimization problem is a piecewise linear convex function. The objective function is equivalent to minimizing $z$ subject to $z \geq \mathbf{c}_i' \mathbf{x} + d_i$ for all $i = 1, \dots, m$. 

Therefore, this optimization problem can be reformulated as follows:

$$
\begin{align*}
\textrm{minimize} \quad & z \\
\textrm{subject to} \quad & z \geq \mathbf{c}_i' \mathbf{x} + d_i \quad \forall i = 1, \dots, m \\
& \mathbf{A} \mathbf{x} \geq \mathbf{b} \\
& \mathbf{x} \geq 0
\end{align*}
$$


# The Newsvendor Problem

## Problem Statement

A newsvendor buys newspapers from the publisher at a price $c$ and sells them to customers at a price $b$. The demand for newspapers is normally distributed with mean $\mu$ and standard deviation $\sigma$. If the newsvendor has unsold newspapers at the end of the day, he can sell them back to the publisher at a price $v$. 

The objective is to determine the optimal order quantity $q^*$ that maximizes the expected profit. 

## Formulation

To formulate the problem, we need to define the following variables:

- $h$: overage cost per unit, $c_o = c - v$
- $p$: underage cost per unit, $c_u = b - c$
- $D$: demand for newspapers
- $Q$: order quantity

The expected profit is given by:

$$ 
g(Q) = \mathbb{E} \left[ h (Q - D)^+ + p  (D - Q)^+ \right] 
$$

where $(x)^+ = \max \{ x, 0 \}$.



$$
\begin{align*}
        \mathbb{E} \left[ h \max \left\{ Q - D, 0 \right\} + p \max \left\{ D - Q, 0 \right\} \right] \\
        = h \int_{0}^{Q} (Q - d) f(d) dd + p \int_{Q}^{\infty} (d - Q) f(d) dd                        \\
    \end{align*}
$$
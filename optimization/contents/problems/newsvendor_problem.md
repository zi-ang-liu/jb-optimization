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

The first derivative of the expected profit is $h F(Q) - p (1 - F(Q))$. The second derivative is $h f(Q) + p f(Q)$. The optimal order quantity is:

$$
Q^* = F^{-1} \left( \frac{h}{h + p} \right)
$$

where $F^{-1}$ is the inverse of the cumulative distribution function of the demand.

## Python Implementation

We can implement the newsvendor problem in Python using the following code:

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.integrate import quad

def newsvendor(mu, sigma, h, p):
    # optimal order quantity
    critial_ratio = p/(p+h)
    Q = stats.norm.ppf(critial_ratio, mu, sigma)
    return Q

h = 0.18
p = 0.7
mu = 50
sigma = 8

Q = newsvendor(mu, sigma, h, p)
print("Q = {}".format(Q))
```



```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.integrate import quad

# parameters
h = 0.18
p = 0.7

# demand distribution
mu = 50
sigma = 8

# optimal order quantity
critial_ratio = p/(p+h)
Q = stats.norm.ppf(critial_ratio, mu, sigma)
print("Q = {}".format(Q))

# plot demand distribution
x = np.arange(0, 100, 0.1)
y = stats.norm.pdf(x, mu, sigma)
plt.figure(figsize=(10, 6))  # Increase the size of the figure
plt.plot(x, y, linewidth=2)  # Increase the line width
plt.xlabel("Demand", fontsize=14)  # Increase the font size of the x-label
plt.ylabel("Probability", fontsize=14)  # Increase the font size of the y-label
# plt.show()
plt.savefig('demand_distribution.pdf', format='pdf')

# plot Q and expected cost
def over_term(d, q):
    return (q-d)*stats.norm.pdf(d, mu, sigma)

def under_term(d, q):
    return (d-q)*stats.norm.pdf(d, mu, sigma)

q = np.arange(20, 100, 0.1)
cost = np.zeros(len(q))
for i in range(len(q)):
    over_term_result = quad(over_term, 0, q[i], args=(q[i]))[0]
    under_term_result = quad(under_term, q[i], np.inf, args=(q[i]))[0]
    cost[i] = h*over_term_result + p*under_term_result

plt.figure(figsize=(10, 6))  # Increase the size of the figure
plt.plot(q, cost, linewidth=2)  # Increase the line width
plt.xlabel("Order Quantity", fontsize=14)  # Increase the font size of the x-label
plt.ylabel("Expected Cost", fontsize=14)  # Increase the font size of the y-label
# plt.show()
plt.savefig('q_vs_cost.pdf', format='pdf')

# plot Q and expected cost for discrete demand
q = np.arange(54, 60, 1)
cost = np.zeros(len(q))
# transfer to discretize demand
demand = np.arange(0, 100, 1)
pmf = np.zeros(len(demand))
pmf[0] = stats.norm.cdf(0.5, mu, sigma)  # Probability of demand being 0
for i in range(1, len(demand)):
    pmf[i] = stats.norm.cdf(i + 0.5, mu, sigma) - stats.norm.cdf(i - 0.5, mu, sigma)

assert abs(sum(pmf) - 1) < 1e-6 # Check if the sum of pmf is 1

for i in range(len(q)):
    for j in range(len(demand)):
        if j <= q[i]:
            cost[i] += h*(q[i]-j)*pmf[j]
        else:
            cost[i] += p*(j-q[i])*pmf[j]

plt.figure(figsize=(10, 6))  # Increase the size of the figure
plt.plot(q, cost, linewidth=2)  # Increase the line width
plt.xlabel("Order Quantity", fontsize=14)  # Increase the font size of the x-label
plt.ylabel("Expected Cost", fontsize=14)  # Increase the font size of the y-label
plt.savefig('q_vs_cost_discrete.pdf', format='pdf')
```
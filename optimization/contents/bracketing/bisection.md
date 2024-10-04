# Bisection Method

```python
def bisection(f, a, b, tol=1e-6):
    """
    Bisection method: root finding algorithm

    Parameters
    ----------
    f : function
        The function to find the root of
    a : float
        Lower bound of the interval
    b : float
        Upper bound of the interval
    tol : float
        Tolerance

    Returns
    -------
    tuple
        A tuple of the form (a, b)
        where f(a) and f(b) have opposite signs
    """

    if f(a) * f(b) > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")

    if f(a) == 0:
        return a, a
    if f(b) == 0:
        return b, b

    while b - a > tol:
        m = (a + b) / 2
        if f(m) == 0:
            return m, m
        elif f(a) * f(m) < 0:
            b = m
        else:
            a = m
    return a, b


def f(x):
    return x**2 - 4


a, b = bisection(f, 0, 3)
print(f"a={a}, b={b}")
```
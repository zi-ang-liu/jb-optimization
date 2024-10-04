# Initial Bracket

```python
def bracket(f, x1=0, s=1e-2, k=2.0):
    """
    bracket function

    Parameters
    ----------
    f : function
        The function to minimize
    x1 : float
        Initial guess
    s : float
        Step size
    k : float
        Expansion factor

    Returns
    -------
    tuple
        A tuple of the form (x1, x2, x3, f1, f2, f3)
        where x1>x2>x3 and f1>f2<f3
    """
    f1 = f(x1)
    x2 = x1 + s
    f2 = f(x2)
    if f2 > f1:
        x1, x2 = x2, x1
        f1, f2 = f2, f1
        s = -s
    while True:
        x3 = x2 + s
        f3 = f(x3)
        if f3 > f2:
            return x1, x2, x3, f1, f2, f3
        x1, x2, x3 = x2, x3, x3 + s
        f1, f2, f3 = f2, f3, f(x3)
        s *= k


def f(x):
    return x**2


x1, x2, x3, f1, f2, f3 = bracket(f, -1)
# print with the format string
print(f"x1={x1}, x2={x2}, x3={x3}, f1={f1}, f2={f2}, f3={f3}")
```
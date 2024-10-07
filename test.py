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
    a : float
        Lower bound of the interval
    b : float
        Upper bound of the interval
    x : float
        The estimated root

    """

    if f(a) * f(b) > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")

    if f(a) == 0:
        return a, a, a
    if f(b) == 0:
        return b, b, b

    i = 0
    while b - a > tol:
        x = (a + b) / 2
        print(f"iter={i}, a={a}, b={b}, x={x}")
        if f(x) == 0:
            return x, x, x
        elif f(a) * f(x) < 0:
            b = x
        else:
            a = x
        i += 1
    return a, b, x


def f(x):
    return x**2 - 4


a, b, x = bisection(f, 0, 3)
print(f"Root={x}, Lower bound={a}, Upper bound={b}")


def newton(f, df, x0, tol=1e-6):
    """
    Newton's method: root finding algorithm

    Parameters
    ----------
    f : function
        The function to find the root of
    df : function
        The derivative of the function
    x0 : float
        Initial guess
    tol : float
        Tolerance

    Returns
    -------
    x : float
        The estimated root

    """

    x = x0
    while abs(f(x)) > tol:
        x = x - f(x) / df(x)
        print(f"x={x}")
    return x


def df(x):
    return 2 * x


def f(x):
    return x**2 - 4


x = newton(f, df, 3)

print(f"Root={x}")

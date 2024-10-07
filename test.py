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
    return x


def df(x):
    return 2 * x


def f(x):
    return x**2 - 4


x = newton(f, df, 3)

print(f"Root={x}")

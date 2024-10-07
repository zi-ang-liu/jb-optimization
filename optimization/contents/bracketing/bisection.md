# 二分法

二分法は数値解析において根を求めるためのアルゴリズムの一つです。

$f$が連続関数である区間$[a, b]$が与えられたとき、$f(a)$と$f(b)$が異符号であれば、$f$は$[a, b]$上で少なくとも1つの根を持ちます。この性質は**中間値の定理**により保証されます。二分法は区間$[a, b]$を狭めていくことで、根を求めるアルゴリズムです。

二分法では、$I^{(0)} = [a^{(0)}, b^{(0)}]$を与えられた区間とし、$f(a^{(0)})f(b^{(0)})<0$であるとします。$I^{(0)}$の中点を$x^{(0)} = (a^{(0)}+b^{(0)})/2$とします。$f(x^{(0)})$の値が以下の三つの場合に分けられます。

1. $f(x^{(0)}) = 0$
2. $f(x^{(0)})<0$
3. $f(x^{(0)})>0$

$f(x^{(0)})=0$の場合、$x^{(0)}$は根であるため、計算を打ち切ります。$f(x^{(0)})<0$、または$f(x^{(0)})>0$の場合、必ず$f(a^{(0)})f(x^{(0)})<0$、または$f(b^{(0)})f(x^{(0)})<0$のどちらかが成り立ちます。この性質を利用して、次の区間を選択します。

- もし$f(a^{(0)})f(x^{(0)})<0$であれば、$a^{(1)}=a^{(0)}, b^{(1)}=x^{(0)}$とします。
- もし$f(b^{(0)})f(x^{(0)})<0$であれば、$a^{(1)}=x^{(0)}, b^{(1)}=b^{(0)}$とします。

$I^{(1)} = [a^{(1)}, b^{(1)}]$を新しい区間として、$f(a^{(1)})f(b^{(1)})<0$であることが保証されます。このように反復計算を行うことで、$I^{(1)}, I^{(2)}, \ldots$と区間を狭めていき、根の近似値を求めることができます。

## アルゴリズム

```{prf:algorithm} Bisection method
:label: bisection-algorithm

**Inputs:** function $f$, interval $[a, b]$, tolerance $\text{tol}$
**Output:** interval $[a, b]$, estimate of the root $x$

1. Ensure $f(a)f(b) < 0$
2. While $b - a > \text{tol}$:
    1. $x \leftarrow (a + b) / 2$
    2. If $f(x) = 0$, return $m$
    3. Else if $f(a)f(x) < 0$, $b \leftarrow x$
    4. Else, $a \leftarrow x$
3. Return $a, b, x$

```

## Pythonによる実装

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
```
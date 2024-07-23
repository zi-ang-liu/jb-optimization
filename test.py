import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.integrate import quad


def newsvendor(mu, sigma, h, p):
    # optimal order quantity
    critial_ratio = p / (p + h)
    Q = stats.norm.ppf(critial_ratio, mu, sigma)
    return Q


h = 0.18
p = 0.7
mu = 50
sigma = 8

Q = newsvendor(mu, sigma, h, p)
print("Q = {}".format(Q))

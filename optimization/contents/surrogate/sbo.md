# Surrogate-based optimization


<!-- ```{prf:algorithm} Surrogate-based optimization
:label: sbo-algorithm

1. **while** stopping criterion not met **do**
 -->


```python
import gurobipy as gp
import numpy as np
from gurobipy import GRB
from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from gurobi_ml import add_predictor_constr


def peak2d(x1, x2):
    return (
        3 * (1 - x1) ** 2.0 * np.exp(-(x1**2) - (x2 + 1) ** 2)
        - 10 * (x1 / 5 - x1**3 - x2**5) * np.exp(-(x1**2) - x2**2)
        - 1 / 3 * np.exp(-((x1 + 1) ** 2) - x2**2)
    )


# prepare the data at random
x1 = np.random.random((1000, 1)) * 4 - 2
x2 = np.random.random((1000, 1)) * 4 - 2
y = peak2d(x1, x2)

# scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x1, x2, y)
plt.tight_layout()
plt.savefig("data.svg", bbox_inches="tight")

X = np.concatenate([x1.ravel().reshape(-1, 1), x2.ravel().reshape(-1, 1)], axis=1)
y = y.ravel()

# Run our regression
layers = [30] * 2
regression = MLPRegressor(hidden_layer_sizes=layers, activation="relu")
pipe = make_pipeline(PolynomialFeatures(), regression)
pipe.fit(X=X, y=y)

# test the model
X_test = np.random.random((100, 2)) * 4 - 2

r2_score = metrics.r2_score(peak2d(X_test[:, 0], X_test[:, 1]), pipe.predict(X_test))
max_error = metrics.max_error(peak2d(X_test[:, 0], X_test[:, 1]), pipe.predict(X_test))
print("R2 error {}, maximal error {}".format(r2_score, max_error))

# build and solve
m = gp.Model()

x = m.addVars(2, lb=-2, ub=2, name="x")
y_approx = m.addVar(lb=-GRB.INFINITY, name="y")

m.setObjective(y_approx, gp.GRB.MINIMIZE)

# add "surrogate constraint"
pred_constr = add_predictor_constr(m, pipe, x, y_approx)

pred_constr.print_stats()

m.Params.TimeLimit = 20
m.Params.MIPGap = 0.1
m.Params.NonConvex = 2

m.optimize()

print(
    "Maximum error in approximating the regression {:.6}".format(
        np.max(pred_constr.get_error())
    )
)

print(
    f"solution point of the approximated problem ({x[0].X:.4}, {x[1].X:.4}), "
    + f"objective value {m.ObjVal}."
)
print(
    f"Function value at the solution point {peak2d(x[0].X, x[1].X)} error {abs(peak2d(x[0].X, x[1].X) - m.ObjVal)}."
)


# plot the optimal solution on the function
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
x1, x2 = np.meshgrid(np.arange(-2, 2, 0.05), np.arange(-2, 2, 0.05))
y = peak2d(x1, x2)
surf = ax.plot_surface(x1, x2, y, cmap=cm.coolwarm, linewidth=0.01, antialiased=False)
# plot the optimal solution projected on the function
# ax.scatter(x[0].X, x[1].X, peak2d(x[0].X, x[1].X), color="red", s=50)
# Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
# tight
plt.tight_layout()
plt.savefig("optimal_solution.svg", bbox_inches="tight")

# plot the optimal solution on 2d with contour
fig, ax = plt.subplots()
CS = ax.contour(x1, x2, y, levels=15)
ax.scatter(x[0].X, x[1].X, color="red")
plt.tight_layout()
plt.savefig("optimal_solution_contour.svg", bbox_inches="tight")
```
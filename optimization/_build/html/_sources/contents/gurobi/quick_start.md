# Quick Start

Solving an optimization problem using Gurobi in Python usually involves the following steps:

1. Create an empty model
2. Add variables
3. Set the objective function
4. Add constraints
5. Optimize the model
6. Retrieve the solution

## Create an empty model

`gb.Model()` is the most used class in Gurobi. It represents a mathematical optimization model.

```python
import gurobipy as gb
from gurobipy import GRB

# Create an empty model
m = gb.Model("NewModel")
```

In the code above, `m` is an object of the class `Model`. The string `"NewModel"` is the name of the model.

## Add variables

Class `Model` has a method called `addVar()` that adds a variable to the model. The method has the following signature:

```python
# defuault arguments
x = m.addVar()

# arguments by name
y = m.addVar(lb=0, ub=1, obj=1.0, vtype=GRB.CONTINUOUS, name="y")
```

The arguments are:

* `lb`: lower bound of the variable
* `ub`: upper bound of the variable
* `obj`: coefficient of the variable in the objective function
* `vtype`: type of the variable. It can be `GRB.CONTINUOUS`, `GRB.BINARY`, or `GRB.INTEGER`
* `name`: name of the variable

By default, `addVar(lb=0.0, ub=float('inf'), obj=0.0, vtype=GRB.CONTINUOUS, name="", column=None)` is called.

In addition to the `addVar()` method, the `Model` class has the `addVars()` method and `addMVar()` method. The `addVars()` method adds multiple variables at once. The `addMVar()` method adds `NumPy` ndarray variables.

## Set the objective function

The objective function is set using the `setObjective()` method. This method has two arguments: the expression of the objective function and the sense of the optimization. The sense can be `GRB.MINIMIZE` or `GRB.MAXIMIZE`.

```python
# Set a objective function to maximize x + 2y
m.setObjective(x + 2 * y, GRB.MAXIMIZE)
# Set a objective function to minimize x^2 + y^2
m.setObjective(x**2 + y**2, GRB.MINIMIZE)
```

Note that `setObjectiveN()` can be used for multi-objective optimization.

## Add constraints

Constraints are added using the `addConstr()` method. The method has two arguments: the expression of the constraint and the name of the constraint.

```python
# Add a linear constraint x + y <= 5, named "c1"
m.addConstr(x + y <= 5, "c1")

# Add a quadratic constraint x^2 + y^2 <= 1, named "c2"
m.addConstr(x**2 + y**2 <= 1, "qc1")

# Add a ranged linear constraint x + y + z in [1, 2], named "rgc1"
m.addConstr(x + y + z == [1, 2], "rgc1")

# Add a matrix inequality constraint
m.addConstr(A @ x <= b, "matrix_inequality")

# Add a Absolute Value Constraint
m.addConstr(x == abs_(y), "abs1")

# Add a logical constraint
m.addConstr(z == and_(x, y, w), "logic1")

# Add a min or max constraint
m.addConstr(z == min_(x, y), "min1")

# Add an indicator constraint
m.addConstr((w == 1) >> (x + y <= 1), "ic1")
```

## Optimize the model

Once we have added the variables, the objective function, and the constraints, we can optimize the model using the `optimize()` method.

```python
m.optimize()
```

## Retrieve the solution

After the optimization, we can retrieve the solution using the `getVars()` method.

```python
for v in m.getVars():
    print(f"{v.VarName} {v.X:g}")

print(f"Obj: {m.ObjVal:g}")
```

## Example: Solving a mixed-integer linear programming problem

Let's solve the following mixed-integer linear programming problem:

$$
\begin{align*} \text{maximize} & \quad x + y + 2z \\ \text{subject to} & \quad x + 2y + 3z \leq 4 \\ & \quad x + y \geq 1 \\ & \quad x, y, z \in \{0, 1\} \end{align*}
$$

```python
import gurobipy as gb
from gurobipy import GRB

# Create an empty model
m = gb.Model("MILP")

# Add variables
x = m.addVar(vtype=GRB.BINARY, name="x")
y = m.addVar(vtype=GRB.BINARY, name="y")
z = m.addVar(vtype=GRB.BINARY, name="z")

# Set objective
m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)

# Add constraints
m.addConstr(x + 2 * y + 3 * z <= 4, "c1")
m.addConstr(x + y >= 1, "c2")

# Optimize
m.optimize()

# Print solution
for v in m.getVars():
    print(f"{v.VarName} {v.X:g}")

print(f"Obj: {m.ObjVal:g}")
```

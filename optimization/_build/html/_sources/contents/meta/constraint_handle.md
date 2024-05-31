# Constraint Handling

In metaheuristic algorithms, constraints are usually handled by the following methods:

- solution encoding
- penalty function
- feasibility rules
- repair

## Solution Encoding

In the solution encoding method, the solution is encoded in such a way that it satisfies the all or some of the constraints. For example, the traveling salesman problem can be encoded as a permutation of cities which inherently satisfies the constraint that each city is visited exactly once and without sub-tours.

## Penalty Function

In the penalty function method, the objective function is modified to include a penalty term for each constraint violation. The penalty term is usually a large value that is added to the objective function value when a constraint is violated. The penalty function method is simple to implement but it may require tuning of the penalty value.

## Feasibility Rules

The feasibility rules method compares the performance of the solutions based on a set of rules.

{cite:t}`Deb2000-zk` uses the following rules to compare two solutions:

1. If one solution is feasible and the other is infeasible, the feasible solution is better.
2. If both solutions are feasible, the one with the better objective function value is better.
3. If both solutions are infeasible, the one with the smaller constraint violation is better.

## Repair

The repair method modifies the infeasible solution to make it feasible. The repair method is usually problem-specific and requires domain knowledge.
import numpy as np
from scipy.optimize import minimize

# define the objective function
def objective(x):
  return (x[0] - 1)**2 + (x[1] - 2)**2

# define the gradient of the objective function
def gradient(x):
  return np.array([2*(x[0] - 1), 2*(x[1] - 2)])

# starting point
x0 = np.array([0, 0])

# minimize the objective function using gradient descent
result = minimize(objective, x0, method='CG', jac=gradient)

# print the minimum value of the objective function
print(result.fun)

# print the optimal solution
print(result.x)

# This code will find the minimum of the function f(x, y) = (x - 1)^2 + (y - 2)^2 starting from the initial point (0, 0),
# using the conjugate gradient method to determine the step size at each iteration.
# The minimum point found by the algorithm will be printed to the console.

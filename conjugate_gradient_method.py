# This code will find the minimum of the function f(x, y) = (x - 1)^2 + (y - 2)^2 starting from the initial point (0, 0),
# using the conjugate gradient method to determine the step size at each iteration.
# The minimum point found by the algorithm will be printed to the console.

#conjugate a gradient method
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

# minimize the objective function using conjugate gradient
result = minimize(objective, x0, method='CG', jac=gradient)

# print the minimum value of the objective function
print(result.fun)

# print the optimal solution
print(result.x)

# plot the objective function
import matplotlib.pyplot as plt
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = objective([X, Y])
plt.contour(X, Y, Z, levels=100)
plt.plot(result.x[0], result.x[1], 'ro')
plt.show()


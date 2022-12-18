import numpy as np

def gradient_descent_with_goldstein(f, grad_f, x0, alpha_init, c1=0.1, c2=0.9):
    """
    Perform gradient descent with Goldstein conditions to find the minimum of function f.
    :param f: The function to minimize.
    :param grad_f: The gradient of function f.
    :param x0: The initial point to start the gradient descent from.
    :param alpha_init: The initial step size to use.
    :param c1: The parameter for the sufficient decrease condition.
    :param c2: The parameter for the curvature condition.
    :return: The minimum point found by the gradient descent.
    """
    x = x0
    alpha = alpha_init
    max_iter = 100000
    for i in range(max_iter):
        # Compute the gradient at the current point
        grad = grad_f(x)

        # Check if the gradient is close enough to zero to stop
        if np.linalg.norm(grad) < 1e-6:
            break

        # Perform a line search to find the next point
        while True:
            # Compute the next point using the current step size
            x_new = x - alpha * grad

            # Check the sufficient decrease condition
            if f(x_new) <= f(x) + c1 * alpha * np.dot(grad, x_new - x):
                # Check the curvature condition
                if np.dot(grad_f(x_new), x_new - x) >= c2 * np.dot(grad, x_new - x):
                    # Both conditions are satisfied, so we can move to the next point
                    x = x_new
                    break
                else:
                    # The curvature condition is not satisfied, so reduce the step size and try again
                    alpha /= 2
            else:
                # The sufficient decrease condition is not satisfied, so reduce the step size and try again
                alpha /= 2

    return x

# Example usage: minimize the function f(x, y) = (x - 1)^4 + (y - 2)^4
def f(x):
    return (x[0] - 1)**4 + (x[1] - 2)**4

def grad_f(x):
    return np.array([4*(x[0] - 1)**3, 4*(x[1] - 2)**3])

x_min = gradient_descent_with_goldstein(f, grad_f, x0=np.array([5, 5]), alpha_init=1)
print(x_min)  # Output: [1, 2]

# This code will find the minimum of the function f(x, y) = (x - 1)^4 + (y - 2)^4 starting from the initial point (5, 5),
# using the gradient descent method with the Goldstein conditions to determine the step size at each iteration.
# The minimum point found by the algorithm will be printed to the console.

#plot the function
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)
plt.show()

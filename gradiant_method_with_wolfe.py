import numpy as np
def gradient_descent_with_wolfe(f, grad_f, x0, alpha_init, c1=0.1, c2=0.9):
    """
    Perform gradient descent with Wolfe conditions to find the minimum of function f.
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

# Example usage: minimize the function f(x) = x^2 - 4*x + 4; the minimum is at x = 2
def f(x):
    return x**2 - 4*x + 4

def grad_f(x):
    return 2*x - 4

x_min = gradient_descent_with_wolfe(f, grad_f, x0=5, alpha_init=1)
print(x_min)

#plot the function
import matplotlib.pyplot as plt
x = np.linspace(-5, 5, 100)
y = f(x)
plt.plot(x, y)
plt.plot(x_min, f(x_min), 'ro')
plt.show()


# This is an implementation of gradient descent optimization with Wolfe conditions.
# The Wolfe conditions are used to determine the step size (learning rate) at each iteration of the gradient descent algorithm.
#
# In this implementation, the function gradient_descent_with_wolfe takes in the following arguments:
#
# f: The function to minimize.
# grad_f: The gradient of function f.
# x0: The initial point to start the gradient descent from.
# alpha_init: The initial step size to use.
# c1: The parameter for the sufficient decrease condition.
# This is a constant that determines how much the function value should decrease at each iteration.
# It is used to ensure that the step size is not too large.
# c2: The parameter for the curvature condition.
# This is a constant that determines how much the derivative of the function should decrease at each iteration.
# It is used to ensure that the step size is not too small.
# The function starts at the initial point x0 and iteratively moves to a new point
# x_new in the direction of the negative gradient of the function. At each iteration,
# the function uses a line search to find a suitable step size (learning rate) alpha that satisfies the Wolfe conditions.
# The line search starts with the initial step size alpha_init and reduces it by half until
# the Wolfe conditions are satisfied or the maximum number of iterations is reached.
#
# The sufficient decrease condition ensures that the function value at the new point x_new is sufficiently
# lower than the function value at the current point x.
# The curvature condition ensures that the derivative of the function at the new point x_new is sufficiently
# lower than the derivative of the function at the current point x.
#
# The function returns the minimum point found by the gradient descent.
#
# In the example usage provided, the function f(x) = x^2 - 4*x + 4 is minimized, and the minimum point is at x = 2.
# The function is plotted and the minimum point is marked with a red circle.
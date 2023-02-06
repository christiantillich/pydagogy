

#%% [markdown]
"""
# Objective

Goal was to go over linear regression with introduction to three major methods 
of solving. 

# General Problem

We have some data, Dr. Ng uses housing prices as an example, with two variables:
`sqft` and `bedrooms`. The linear model takes the form
"""

#%% [python]
import math
import latexify

@latexify.function(use_math_symbols = True)
def h(B, x):
    return sum(B_j * x_j)

h

#%% [markdown]
"""
This is our "hypothesis". Ng acknowledges that's kindof a weird term here - and
I suspect that's because "hypothesis" tends to be reserved for fitness tests of
the model. `h` itself is the model, typically. 

Here, `x0 := 1`, and `B0` is just our constant term. 

Ultimately, the solution takes the form
"""

#%% [python]
i = {
    "min": "argmin_B"
}

@latexify.expression(use_math_symbols = True, identifiers = i)
def h_prime(x, y):
    return  min( 1/2 * sum(h(x_j) - y_j))

h_prime

#%% [markdown]
"""
for n params. Dr. Ng covers the three most common approaches to this problem. 

### "Batch" Gradient Descent

Visually, we're trying to get to the bottom of this guy. 
"""

#%% [python]
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')   # Create the axes

# Data
X = np.linspace(-8, 8, 100)
Y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(X, Y)
Z = X**2 + Y**2

# Plot the 3d surface
surface = ax.plot_surface(X, Y, Z,
                          cmap=cm.coolwarm,
                          rstride = 2,
                          cstride = 2)

# Set some labels
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')

plt.show()

#%% [markdown]
"""
We'll do it by taking small steps down the hill. Mathematically, we update `B` 
by the following formula. 
"""

#%% [python]
@latexify.function(use_math_symbols = True)
def B_k(B_j, alpha):
    return B_j - alpha * (delta/(delta*B_j)) * J(B)
B_k

#%% [markdown]
"""
Where `J` itself is defined as 
"""
#%% [python]
@latexify.function
def J(B, x, y):
    return 1/2 * (h(x**[i]) - (y**[i]))**2
J

#%% [markdown]
"""
And actually, this just ends up simplifying to
"""

#%% [python]
@latexify.function(use_math_symbols = True)
def B_k(B_j, alpha):
    return B_j - alpha * (h(x**[i]) - y**[i]) * x_j**[i]
B_k


#%% [markdown]
"""
### Stochastic Gradient Descent

So, all of that is actually pretty computationally expensive. We can actually 
make a small modification. We take a single observation, and update `B` one 
single house at a time. It's more error prone, and the algorithm needs higher 
error tolerances for convergence, because it is going to "bounce around" the 
minima more. But SGD is _much_ faster. 

And another practical modification is to decrease the learning rate over time. 
This makes the later oscillations much smaller. 

Finally, we can simply plot the error function `J` over iteration number. You 
should see `J` kindof bottom out at some point, and you can stop then. 
"""


#%% [markdown]
"""
### The Normal Equation

In the simplest form, Linear Regression actually has a single-shot solution. At
least, so far as we're comfortable working with linear algebra. We can reframe
our cost function `J` as 
"""

#%% [python]
@latexify.function(use_math_symbols = True)
def J(B):
    return (X*B - y)**T * (X*B - y)
J
#%% [markdown]
"""
And if we're trying to take the derivative of our error function, in linear 
algebra notation, it goes as
"""
#%% [python]
@latexify.expression
def line1():
    return grad(J) == 1/2 * grad_B * (B**T * X**T - y**T) * (X*B-y)
line1
#%% [python]
@latexify.expression
def line2():
    return 1/2 * grad_B * (B**T * X**T * X * B - B**T*X**T*y - y**T*X*B + y**T*y)
line2
#%% [python]
@latexify.expression
def line3():
    return 1/2 * (X**T*X*B + X**T*X*B - X**T*y - X**T*Y)
line3
#%% [python]
@latexify.expression
def line4():
    return X**T*X*B - X**T*y 
line4

#%% [markdown]
"""
So if we just set the derivative to zero, we get. 
"""
#%% [python]
@latexify.expression
def line5():
    return grad(J) == 0 == X**T*X*B == X**T*y
line5
#%% [python]
@latexify.expression
def line6():
    return B == (X**T*X)**-1*X**T*y
line6

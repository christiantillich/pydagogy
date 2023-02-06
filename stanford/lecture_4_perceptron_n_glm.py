#%% [markdown]
"""
# Objective

Review a number of algorithms - perceptron, generalized linear models, and 
softmax regression for multiclass. 

### Perceptron

Logistic regression uses the sigmoid function to map linear numbers to a 
continuous space between `y in [0,1]`. The perceptron model trades out this 
idea for a piecewise continuous function. 
"""
#%% [python]
import latexify
@latexify.function(use_math_symbols = True)
def g(z):
    if z > 0:
        return 1
    else:
        return 0
g
#%% [markdown]
"""
So that our predictor function now is 
"""
#%% [python]
@latexify.function(use_math_symbols = True)
def h(x):
    return g(theta**T*x)
h
#%% [markdown]
"""
And our gradient descent update rule is
"""
#%% [python]
@latexify.function(use_math_symbols = True)
def theta(theta):
    return theta[j] + alpha*(y**[i] - h[theta](x)**[i])*x[j]**[i]
theta

#%% [markdown]
"""
Here, `theta` defines a vector that's normal to this decision boundary line, 
above which `yhat` is 1 and below which `yhat` is 0. Gradient descent will try
to reposition this decision boundary to maximally separate the two classes. 

One person asked why this doesn't get used in practice, and the answer that the
TA gave was that it doesn't have a probabilistic interpretation. Also, the TA
gives an "Anscombe-like" example that the perceptron would immediately fail on. 
He also notes that the sigmoid used in logistic is itself "perceptron-like". 
He probably should _also_ note that this does end up serving as a starting 
point for neural architectures. 

Nevertheless, I feel like we're really rushing through this, might be good to 
go through the book and lecture notes on the topic. 
"""

#%% [markdown]
"""
### Exponential

An exponential family whose Probability Density Function takes the form. 
"""
#%% [python]
@latexify.function(use_math_symbols = True)
def P(y,eta):
    return b(y)*exp(eta**T*T(y) - a**eta)
P

#%% [markdown]
"""
Here  
* `y` is the data
* `T(y)` is a sufficient statistic of `y`
* `b(y)` is a base measure
* `a(eta)` is something called the log partition. 

The partition function is essentially a normalizing term, so the whole thing 
integrates to 1. 

This is all pretty abstract, but the general idea here is that if you can 
massage some PDF that you've got into this form, what you have done is proven
that the distribution belongs in the Exponential family. 

We look at the Bernoulli and Gaussian as an example.  

##### Bernoulli
"""

#%% [python]
@latexify.function(use_math_symbols = True)
def P(y, phi):
    return phi**y*(1-phi)**(1-y)
P

#%% [python]
from math import exp
@latexify.expression(use_math_symbols = True)
def line1(y, phi):
    return exp(log(phi**y*(1-phi)**(1-y)))
line1

#%% [python]
@latexify.expression(use_math_symbols = True)
def line2(y, phi):
    return exp(log(phi/(1-phi))*y + log(1-phi))
line2


#%% [markdown]
"""
So we haven't actually done anything by wrapping this in `exp(log())`. But
by doing so, we can show that  
* `b(y) = 1`
* `T(y) = y`
* `eta = log(phi/(1-phi))`
* `a(eta) = -log(1-phi)` where `phi` is itself a function of `eta`

And thus, the Bernoulli distribution is part of the Exponential Family. Neat. 

##### Gaussian

We'll cheat here just a bit and assume the variance is 1. 
"""
#%% [python]
from math import sqrt
@latexify.function(use_math_symbols = True)
def P(y, mu):
    return 1/sqrt(2*pi)*exp(-(y-mu)**2/2)
P
#%% [python]
@latexify.expression(use_math_symbols = True)
def line1(y, mu):
    return 1/sqrt(2*pi)*exp(-(y/2)) * exp(mu*y - 1/2*mu**2)
line1

#%% [markdown]
"""
So again  
* `b(y) = 1/2pi * exp(-y/2)`
* `T(y) = y`
* `eta = mu`
* `a(eta) = eta^2/2`

The whole point in covering this is that there are nice, exploitable properties
of the exponential family.  
* MLE with respect to eta is always concave. 
* The log-likelihood is convex. 
* The expectation of y is the first derivative of `a(eta)`
* The variance of y is the second derivative of `a(eta)`

You can prove them in the homework. The derivatives relationships are especially
important, because generally speaking finding the mean and variance of a 
distribution requires integration, which is much more complex. 
"""


#%% [markdown]
"""
### Generalized Linear Models

So the key to GMLs is that, by switching up our choice of exponential family 
distributions, we can actually unlock a number of really excellent model 
types. These types have a lot of real world applicability. 

* Real valued regression problems use Gaussian. 
* Binary data use the Bernoulli
* Count data can use Poisson
* Positive integers only can use the Gamma or Exponential (dist only, not family)
* Some other dists have Bayesian applications: e.g. Beta, Dirichlet
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def line1(y, theta):
    return y[x,theta] is Exponential(eta)
line1

#%% [python]
@latexify.expression(use_math_symbols = True)
def line2(eta, x):
    return eta == theta**T*x
line2

#%% [python]
@latexify.expression(use_math_symbols = True)
def line3(x, theta):
    return E[x,theta](y) == h[theta](x)
line3

#%% [markdown]
"""
And what we're really saying is, that for some inputs x, we can map to some 
value eta (by `theta**T*x`), which itself can map to some distribution parameters, 
which can be plugged into the prediction function `h` to yield the expected
value. 

So training again is just
"""

#%% [python]
from math import log
@latexify.expression(use_math_symbols = True)
def line4(y, mu):
    return argmin[theta]*log(P(y**[i], theta**T*x**[i]))
line4

#%% [markdown]
"""
but now for a whole family of distributions. _And_ the update rule is still the
same as ever. 
"""
#%% [python]
@latexify.function(use_math_symbols = True)
def theta(theta):
    return theta[j] + alpha*(sum(y**[i] - h[theta](x)**[i]))*x[j]**[i]
theta

#%% [markdown]
"""
Couple other notes here. 
* `eta` is called the "natural parameter". 
* `E[y|eta]` is called the "canonical response function" `g`
* `g^-1` is called the "link function". 

So in training, we learn the model parameters `theta`. The combo of `theta` and
`x` gives us the natural parameter for the distribution. And by plugging in 
the natural parameter, we can work out the shape of the distribution for those
x values as well as the expectation, i.e. the prediction. 

So ultimately, GLMs give us a really flexible way of modeling a lot of different
types of data all wrapped up nicely in a single framework. And ultimately the
trick is reframing these distributions as part of the exponential family, where
there is a logical mapping that goes 
learning params -> natural param -> prediction

Another way to think about this is that, in normal regression, we assume that
for every x, there is a normal distribution centered at that point where the
values can "typically" fall. In GLMs, that distribution surrounding x can be
coerced to be some other family. In logistic, that distribution is Bernoulli. 
In Poisson regression, that distribution is Poisson. And so on. 
"""


#%% [markdown]
"""
### Softmax

So softmax is one way of doing multilabel classification. There's a formulation
of this as a GLM. However, it's pretty messy, and the Cross-Entropy
interpretation is actually a lot cleaner. So while we could frame the problem as
GLM, and come up with all the same equations, we're going to take the easy path. 

Once again, we find ourselves wanting to draw these decision boundaries. The
example drawn is to put a new observation into one of three baskets, but it's
generalizeable. 

Our `y` is a matrix of mostly `0`, but a `1` for the particular column that 
denotes the class. Visually, for each class, we're trying to draw a decision
boundary and ultimately work out a probability of being above/below each 
boundary. 

We write our hypothesis function as 
"""

#%% [python]
@latexify.function(use_math_symbols = True)
def h(x):
    return exp(theta[i]**T*x) / sum(exp(theta[i]**T*x))
h

#%% [markdown]
"""
And this is essentially a canonical response to map our inputs to a probability
distribution over all of our classes. 

We still minimize errors, but we're going to do it in a different way now. By
minimizing the distance between `h(x)` and `y` according to the following 
formula
"""
#%% [python]
@latexify.function(use_math_symbols = True)
def CrossEnt(h, y):
    return -sum(p(y)*log(h(x)))
CrossEnt

#%% [markdown]
"""
we've gone from minimizing error to minimizing cross entropy. `p(y)` here is 
the actual probability of taking on each class, and the sum is over the available
classes. So for a given observation, `p(y)` collapses because each obs takes on
only one value. So the cross entropy of a single obs predicting class `c` over
all classes `C` takes the form. 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def line2(y):
    return -log(exp(theta[c]**T*x) / sum(exp(theta[C]**T*x)))
line2

#%% [markdown]
"""
It's worth noting too here that there's now different values for `theta` based 
on the different classification labels. The visualization here again is decision
boundaries, and there's no "leave one out" going on here, since each boundary
gives us a prob of being above/below. 
"""

#%% [markdown]
"""
# Objective

To review the common modifications to regression - locally weighted and logistic
regression.

### Locally Weighted Regression

So the problem we try to solve here is "what happens when the data you're trying
to fit isn't a straight line"? Dr. Ng gives a small nod that some problems are
simply feature engineering problems, but locally weighted regression is another
approach when the simple solutions aren't going to be enough. 

This is a non-parametric algo, and specifically that means that the number of
parameters required grows (in this case, linearly), with the amount of data. 

The idea, in general, is that we have a moving window (kernel). We fit a line
just for that small window, and we advance the window at each point. 
"""
#%% [python]
import latexify
@latexify.expression(use_math_symbols = True)
def fit_lwr(beta):
    return sum(w**[i]*(y**[i]-beta**T*x**[i])**2)
fit_lwr

#%% [markdown]
"""
Here `w**[i]` is a weight function, and a common choice might include
"""
#%% [python]
from math import exp
@latexify.function(use_math_symbols = True)
def w(x, tau):
    return exp(-(x**[i] - x)**2 / (2*tau**2))
w

#%% [markdown]
"""
And in general

$$
\begin{cases} 
    w^{[i]} \approx 1,         & \text{if } |x^{[i]} - x| \to 0\\
    w^{[i]} \approx 0,         & \text{if } |x^{[i]} - x| \text{ otherwise}   
\end{cases}
$$

Dr. Ng added a `tau` symbol, to control the bandwidth of our kernel. The whole
function makes our kernel normal-ish, and tau controls the spread. 

"""

#%% [markdown]
"""
### Probabilistic Interpretation

Dr. Ng provides pretty much the same perspective as Dr. Shalizi on this topic - 
up until this point, all we're doing is pure math. Fitting lines through points, 
that's it. If we make the assumption, however, that our model includes an error
term, and the error is normally distributed, it is precisely this assumption
that allows us to start making probabilisitic statements from our model. 

In other words, combining the following two expressions:
"""

#%% [python]
@latexify.expression(use_math_symbols = True)
def linear(x):
    return y**[i] == beta**T*x[i] + epsilon**[i]
linear

#%% [markdown]
"""
$$
\epsilon^{[i]} \sim \mathcal{N}(\mu, \, \sigma^{2})
$$

Allows us to write a statement about the probability of a given value of `y` 
from the prediction of the model. 

$$
\mathcal{P}(y^{[i]} | x^{[i]}; \beta^{[i]}) = \dfrac{1}{\sqrt{2\pi\sigma}} e^{-\dfrac{(y^{[i]}-\beta^{T}x^{[i]})^{2}}{2\sigma^{2}}}
$$

Remember too, the I.I.D. assumption is pretty critical for this thing to work, 
and the example Dr. Ng gives of house prices - he fully admits - do not follow 
this assumption. Yknow, especially in market crashes and non-stable market 
behavior. If we're not assuming distributions, we don't have to worry about this, 
but as soon as the distributions start, so do the assumptions. 

Now that we've got a probabilistic formulation, however, we can actually 
get another solution to the problem as well. Enter the Maximum Likelihood
formulation. 

$$
\mathcal{L}(\beta) = \Pi_{i} \mathcal{P}(y^{[i]} | x^{[i]}; \beta)
= \Pi_{i} \dfrac{1}{\sqrt{2\pi\sigma}} e^{-\dfrac{(y^{[i]}-\beta^{T}x^{[i]})^{2}}{2\sigma^{2}}}
$$

and by taking the log of the likelihood, we can coerce this thing into something
that is more-easily calculable _and_ based on a summation. So when we do MLE, 
we effectively are _doing least squares anyway_, which is pretty neat. 

$$
argmin_{\beta} \ ln(\mathcal{L}(\beta)) \\
argmin_{\beta} \ ln(\dfrac{1}{\sqrt{2\pi\sigma}}) + \Sigma_{i}\dfrac{-(y^{[i]}-\beta^{T}x^{[i]})^{2}}{2\sigma^{2}} \\ 
argmin_{\beta} \ \Sigma_{i}\dfrac{-(y^{[i]}-\beta^{T}x^{[i]})^{2}}{2\sigma^{2}}
$$
"""

#%% [markdown]
"""
### Logistic Regression

So for the simplest classification problem, we want to start working towards
a form of regression that allows for `y in [0,1]`. We _could_ just do regression, 
yknow, we _can_ do anything we want. However, there's a lot of reasons not to, 
and Dr. Ng focuses on how one outlier will change the classification space
for a wide number of observations. For anyone who isn't a clown, logistic 
regression is the natural choice. 

What we want is essentially a mapping of our cost function `h(x)` to `[0,1]`. 
And all we really do is "wrap" our linear equation with the logistic function. 
"""
#%% [python]
@latexify.function(use_math_symbols = True)
def g(beta,x):
    return 1 / (1+e**(beta**T*x))
g
#%% [markdown]
"""
But it's not enough to fit lines, we need to make probability statements here. 
We can model the probability of `y=1` and `y=0` as 
"""
#%% [python]
@latexify.function(use_math_symbols = True)
def P(y, x, beta):
    return h(x|beta)**y * (1-h(x|beta))**(1-y)
P
#%% [markdown]
"""
And so we spent all that time covering maximum likelihood above, because that 
gives us a really easy way to frame the problem we're trying to solve. We can 
simply swap out the new probability function we've created and see where that 
leads us 

$$
argmin_{\beta} \ ln(\mathcal{L}(\beta)) \\
argmin_{\beta} \ ln(\Pi_{i} \mathcal{P}(y^{[i]} | x^{[i]}; \beta^{[i]})) \\ 
argmin_{\beta} \ ln(\Pi_{i} h_{\beta}(x^{[i]})^{y^{[i]}}(1-h_{\beta}(x^{[i]})^{1-y^{[i]}}) \\ 
argmin_{\beta} \ \Sigma_{i} y^{[i]} \ ln \ h_{\beta}(x^{[i]}) + (1-y^{[i]}) ln (1-h_{\beta}(x^{[i]}))
$$
"""

#%% [markdown]
"""
So that's the game, only now we're trying to _maximize_ the log likelihood. So 
we switch up our updating function so that 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def update(beta):
    return B_j == B_j + delta / (delta * B_j) * alpha * sum(y**[i] - h(x**[i])) * x_j**[i]
update

#%% [markdown]
"""
### Newton's Method

Ultimately, the advantage here is that we can take some pretty big jumps, while
still having something that converges. The tradeoff here is that each given 
step is more costly. 

The overall flow of Newton's method is kindof like this:
1. Take a starting value `x`. 
2. Draw a tangent line at `f(x)`.
3. Follow the tangent line to where it crosses y=0. Call it `x'`
4. `f(x')` is your new starting guess

The general form for parameters B looks very similar to gradient descent. 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def newton(beta):
    return beta[j+1] == beta[j] - f(beta[j]) / (delta*f(beta[j]))
newton

#%% [markdown]
"""
And in full vector notation we may see it written as 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def nvec(beta):
    return beta[j+1] == beta[j] + H**(-1)*grad[beta](LL)
nvec

#%% [markdown]
"""
Where H is the Hessian matrix
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def H(i,j):
    return H[i,j] == delta**2 * LL / (delta*beta[i] * delta*beta[j])
H
#%% [markdown]
"""
In English, to update parameter `Beta`, we choose some starting value, 
compute the gradient of the log loss at that point, multiply _that vector_ by the
inverse of the Hessian (which is a matrix of partial derivatives), and then 
adding the whole thing to the starting guess. It's a lot, and it breaks down
in high dimensional cases. 

This whole section is kindof rushed, and it might be worth going back to the old
440 Linear Algebra book to refresh on gradients and Hessians. 
"""
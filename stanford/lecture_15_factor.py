#%% [markdown]
"""
# Objective

Continue our discussion on EM convergence, also talk about monitoring for EM. 
Then we review Factor Analysis, and the EM method for solving that. 

### EM Convergence

What we proved last time that the E-step chooses Q - some point that is both
on the log-likelihood curve. The choice of Q dictates a smaller-form optimization
(from Jensen's Equality) that gives us a pseudo-likelihood equation to optimize
instead. Optimizing that function necessarily gets us closer to the optimal 
point on the likelihood curve, because of Jensen's Inequality, so optimizing that
replacement is the M-step. This defines a new value for Q that we work out in a
new E-step, choose max theta in a new M-step, and so on. 

He writes out the equations again, I'm content to just look back at lecture 14. He
finishes up a couple thoughts but I just put it there so everything's in one 
spot. 
"""
#%% [markdown]
"""
### Coordinate Ascent

We're going to generalize lecture 14's results a bit here, though. If we define

"""
#%% [python]
import latexify 

@latexify.expression(use_math_symbols = True)
def f(x):
    return J(theta, Q) == Sigma[i]*Sigma[j]*Q(z[i]) * log(P(x[i],z[i]|theta)/Q(z[i]))
f
#%% [markdown]
"""
Then what we showed in lecture 14 is `L(theta) >= J(theta,Q)` for any `[theta,Q]`.

And even more generally, EM can be thought of as 
* E-step - maximize 'J' wrt 'Q'
* M-step - maximize 'J' wrt 'theta'

This is generically referred to as Coordinant Ascent, and is probably going to
be a useful way of thinking about the form of EM generally as we move into
different algorithms
"""

#%% [markdown]
"""
### Factor Analysis

So Guassian Mixture models are going to work well when we have a small number of
features and a large number of training examples. Dr. Ng frames factor analysis
as an excellent use-case when you've got something like the reverse. 

I think that's kindof a limited way of thinking about FA. I would probably frame
it more like "you use GM when you're pretty confident that you've got certain
classes of observations, and you use FA when you're certain you have classes of
variables". FA is going to group variables as belonging to a latent scale, like
30 SAT questions mapping to a "mathematical competency" measure that you can't
directly observe. But we'll go with this for now. 

We've got a lot of variables and about as many observations, so what do we do?
We could model it as a GMM with one Gaussian, but that's actually not possible -
the covariance matrix is going to be singular. Imagine trying to fit a single
Gaussian in 2d variable space with two obs - you get a distribution that's
infinitely skinny. 

One option is to ensure `sigma` is diagonal. Then we can get around the tricky
mathematics of having a singular covariance matrix. But the problem is the 
assumption - we're now assuming all our variables are uncorrelated. So unless
you have a very particular type of data, this isn't going to work. 

Another option is that all features have the same scalar variance, so 
`Sigma = sigma**2 * I`. That's actually way more restrictive, and requires 
even more particular data. 

If only there was some way to capture the feature correlations in such a way that
our constructs that capture those correlations were independent of each other. 
Enter Factor Analysis. 

Alright, I see now. Dr. Ng is really presenting FA as a very useful tool in 
small-data problems _because_ of its ability to group types of variables. I still
don't think you should pigeon-hole FA as just a small data problem, but I 
get why he would hold up small data problems as a place where FA excels. 

Anyway, onto the math. Assume we're looking for a joint distribution
"""

#%% [python]
@latexify.function(use_math_symbols = True)
def f(x):
    return P(x,z) == P(x|z)*P(z)
f

#%% [markdown]
"""
where `z` is hidden. we assume `z ~ N(mu, I)` and `z in R**d`, where `d` is 
less than `n`. And then assume
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return x == mu + Lambda * z + epsilon
f

#%% [markdown]
"""
So, explicitly, we're saying any observation is the sum of some variable-specific
average, some participation in a latent variable that we can't observe, and some
error. That's our model now. 

Imagining 100 temperature sensors all placed throughout the room, we could
imagine that many of them give similar readings. And we could imagine the real
main _causes_ of why all the sensors might have similar readings - different
ambient temperature outside, different intensities of the lights on the left vs
right, and perhaps some sensors placed closer to AC vent, etc. Intuitively, this
type of model makes a lot of sense - often our observations are just an imperfect
reflection of what it is we're trying to measure. If we are comfortable making
that assumption, FA gives us a great tool for quantifying its structure. 

### EM Steps

So this is perhaps the trickiest derivation we look at in the class. Before we
get into it, we need to describe. Dr. Ng shows a notation for illustrating that
for a vector `x`, some fraction belongs to space `r` and others `s`. There's no
way to do this with `latexify` I don't think. 

Quick review, marginal densities let us take a join distribution and figure out
the distribution for a single variable. 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return P(x) == Sigma[y] * P(x,y)*dy
f
#%% [markdown]
"""
here I don't have a good integral sign, so we're gonna hang with `Sigma` for now
and see how it goes. We also need to remember that with conditionals, we can
show that 
"""
#%% [python]
@latexify.function(use_math_symbols = True)
def f(x):
    return P(x[1]|x[2]) == N(mu[1|2], sigma[1|2])
f

#%% [markdown]
"""
with these properties, let's derive the EM algorithm for Factor Analysis. 

First, we derive `P(x,z)`
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return [z, x] == N(mu[z,x], sigma)
f
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return z == N(0,I)
f
#%% [markdown]
"""
so
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return E(x) == E(mu + Lambda*z + epsilon) == mu
f
#%% [markdown]
"""
and 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return E(sigma) == E(x-E(x))*E(x-E(x))**T
f
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return E(sigma) == E(Lambda * z * z**T * Lambda**T) + E(epsilon*epsilon**T)
f
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return E(sigma) == Lambda * Lambda**T + Psi
f
#%% [markdown]
"""
Now, we can therefore say that 

$$
\begin{bmatrix} x \\ z \end{bmatrix} \sim \mathcal{N}( \begin{bmatrix} 0 \\ \mu \end{bmatrix}, \begin{bmatrix} I & \lambda^T \\ \lambda & \lambda\lambda^T + \Psi \end{bmatrix})
$$

We _could_ try to take `P(x[i])`, set derivatives of the log likelihood equal to
zero, and then try to solve. However, Dr. Ng just straight up says there's no
closed form solution here, so that's a wash. But we still want to do it, so 
let's look to EM. 

The E-step is about computing
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x): 
    return Q(z[i]) == P(z[i]|x[i], theta)
f
#%% [markdown]
"""
we can actually express
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return z[i] | x[i] == N(mu[z[i]|x[i]],sigma[z[i]|x[i]])
f
#%% [markdown]
"""
and so, computationally, we can express `Q` as 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return mu[z[i]|x[i]] == 0 + Lambda**T * (Lambda*Lambda**T + Psi)**(-1)*(x[i]*mu)
f
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return sigma[z[i]|x[i]] == I - Lambda**T * (Lambda*Lambda**T + Psi)**(-1)*Lambda
f
#%% [markdown]
"""
So in the E-step, we compute mu and compute sigma and `Q=N(mu, sigma)` from above. 

And so the M-step now is really long and complicated, but it all turns on the 
following trick. There are a couple parts in the derivation where you get an
integral of `Q(z)*z*dz`. But it's just the expected value of `z` when it's drawn
from distribution `Q`.

The M-step, ultimately, is 

$$
\underset{\theta}{\arg\max} \sum_{i} \int_{z_{i}} Q(z_{i}) log(\dfrac{P(x_{i},z_{i})}{Q(z_{i})}) dz_{i}
$$

and can be written as an expectation 

"""

#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return argmax[theta] * Sigma[i] * E[z[i] in Q]*[P(x[i],z[i]) / Q(z[i])]
f

#%% [markdown]
"""
and if we plug in the Guassian density for `P` and `Q`, the whole thing becomes
a quadratic equation, and you can take the derivative with respect to the mean, 
set it to zero, and get a closed form solution. Dr. Ng leaves it to the students
to go through the lecture notes. 
"""

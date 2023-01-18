#%% [markdown]
"""
# Objective

We start on unsupervised learning methods. We go over k-means, guassian mixtures, 
and then use those to understand Expectation Maximization. 

### K-means

You know the drill - pick random points, bisect the space between them, classify,
take the average in the param space, use this as your new center, repeat. Run to
convergence. 

Let's see it in math now. We set `mu` as some initialization vector of centroids, 
and we update
"""
#%% [python]
import latexify
@latexify.expression(use_math_symbols = True)
def f(x):
    return c[i] == argmin[j]*abs(abs(x[i] - mu[j]))**2
f
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return mu[j] == Sigma[i] * int(c[i] == j)*x[i] / (Sigma[i]*c[i] == j)
f

#%% [markdown]
"""
This alternation step is important, and in fact I believe we can show it's a 
rudimentary example of EM later. The first step sets class `c` assigned to each
observation, based on who's closer. The second step redefines the new means. 

We can show that this will always converge. 

This leaves us with a hyperparameter to tune - k. And oftentimes it's extremely
ambiguous. Dr. Ng kindof handwaves away a bunch of the decision aids, but in 
practice I've found skree plots and PCA plots _extremely_ helpful. His advice, 
though, is that usually the application gives you a hint as to how many segments
you're looking to actually utilize. That's fair, but in case that's ambiguous or
ill-defined, skree and PCA give you a place to start. 
"""

#%% [markdown]
"""
### Mixture Models

Sometimes, however, we want more. We can imagine a plot with two densely populated
clusters, and one observation that doesn't fit neatly in either. This is more 
of an "anomaly detection" class of problems, and so what we really want is to 
define rigid boundaries where you can be of Class 1, Class 2, or none of the 
above. 

Enter mixture models, where we try to draw n-dimensional Guassian distributions
around the clusters of points. This gives us a way of assigning probabilities
of belonging in each cluster, and then we can better assess anomalies. 

We assume that there are latent Gaussian distributions responsible for the 
points as we see distributed.
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return P(x[i], z[i]) == P(x[i]|z[i])*P(z[i])
f
#%% [markdown]
"""
where `z ~ MN(phi)` and `x[i]|z[i]==j ~ N(mu, sigma)`. This is actually pretty
similar to GDA, except we no longer get to observe z. So we assume that it 
has some distribution, and write out the likelihood anyway, take the derivative, 
and so on . 
"""
#%% [python]
@latexify.function(use_math_symbols = True)
def L(phi, mu, sigma):
    return Sigma[i]*log(p(x[i], z[i], phi, mu, sigma))
f
#%% [markdown]
"""
We've already done this kindof thing before, so we'll just write out the MLE
estimates of mu, phi
"""

#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return phi[j] == 1/m * Sigma[i]*int(z[i] == j)
f
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return mu[j] == Sigma[i]*int(z[i] == j)*x[i] / (Sigma[i]*int(z[i] == j))
f

#%% [markdown]
"""
None of this has changed from GDA, except swapping y for z. However, because
z is now no longer observable, these equations can't be solved directly. Fortunately, 
that's the whole point of E-M. We're going to bounce back and forth between
the two equations and kindof work our way into the solution through the back
door. 
"""

#%% [markdown]
"""
### E-M derivation

EM has two steps
1. Expectation
2. Maximize

Expectation is about "finding the centers", much like from k-means. The analogy
for our Gaussian distributions is 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return w[j, i] == P(z[i] == j| x[i], phi, mu, sigma)
f
@latexify.expression(use_math_symbols = True)
def f(x):
    return w[j, i] == P(x[i]|z[i] == j)*P(z[i] == j) / (Sigma[l]*P(x[i]|z[i] == l)*P(z[i]==l))
f

#%% [markdown]
"""
furthermore, we know the expressions for `P(x|z)` and `P(z)`. So we plug those
in and use Bayes rule and get a really nasty expression for `w ~ f(x[i], phi, mu, sigma)`.
But given that we start with initialized parameters for `[phi, mu, sigma]`, this 
is all computable. And so we do for each observation. 

The maximization step, then, takes our MLE estimates and replaces the indicator
expression with the expectation. 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return phi[j] == 1/m * Sigma[i] * w[j,i]
f
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return mu[j] == Sigma[i]*w[j,i]*x[i] / (Sigma[i]*w[j])
f

#%% [markdown]
"""
The analogy from k-means again is that, rather than creating an arbitrary class
assignment, each point gets assigned `w`, which is the expectation `E[int(z[i]
== j)]`. So for any observation, there is a vector `w` that encodes a
"likelihood strength" of being from cluster 1, cluster 2, etc. And we use that
soft assignment to work back into `[mu, phi]`.

The expectation step finds the formula for `w` given `phi, mu`, the maximization
step uses that formula to find a new value of `phi, mu` given the formula for
`w`. We bounce back and forth between these two until we converge on `phi, mu`. 

The end result is the full joint density `P(x,z)`, and when we actually collapse
it to just `P(x)` by computing the marginals, we typically get a rich, complex
distribution over x that isn't neatly captured by a single parametric model. It
can be multimodal, or incorporate a wide variety of skews, kurtoses, etc. It's 
something analogous to Taylor Series approximation, or Fourier Analysis, but 
for distributions. We have a way of figuring out the "spectrum" of guassians that
would most likely explain the pattern we're seeing. Neat. 

### EM For Realz

This is still kindof hand-wavy. All we've done so far is show the "shape" of the
EM algorithm, that's not the same as deriving it or proving that it will
necessarily converge. We begin the proof now. 

In order to do the proof, we need something called Jensen's Inequality. Suppose
we have a convex function `f` (i.e. `f''(x) > 0`). Jensen's Inequality states 

> Let X be a random variable, `f(E[x]) <= E[f(x)]`. Furthermore, if `E[f(x)] =
> f(E[x])`, then x is a constant. 

It's kindof really just a fancy way of saying that the basin of a convex function
is necessarily lower than the midpoint between any line drawn between two points
on that function. And if the two are equal, you're at the bottom. 

So here's the density estimation problem 

> We have a model for `P(x,z; theta)` but only observe `x`

The log likelihood is given by 
"""
#%% [python]
@latexify.function(use_math_symbols = True)
def L(theta):
    return Sigma[i] * log * P(x[i]|theta)
L

#%% [markdown]
"""
And we're going to phrase this likelihood in the following way, for convenience.
"""
#%% [python]
@latexify.function(use_math_symbols = True)
def L(theta):
    return Sigma[i] * log(Sigma[z[i]] * P(x[i], z[i]|theta))
L

#%% [markdown]
"""
The optimization goal is still 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return argmax[theta] * L(theta)
f
#%% [markdown]
"""
and we can add a further convenience edit here that goes like
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(theta):
    return argmax[theta] * Sigma[i] * log(Sigma[z[i]] * Q[i](z[i]) * (P(x[i], z[i]|theta) / (Q[i]*z[i])))
f
#%% [markdown]
"""
We haven't actually changed anything - that's just multiplying by 1. But this
move allows us to rephrase the whole problem as an expectation
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(theta):
    return argmax[theta] * Sigma[i] * log(E[z[i] in Q[i]]*((P(x[i], z[i]|theta) / (Q[i]*z[i]))))
f
#%% [markdown]
"""
and from Jensen's inequality, we know that the following rewrite must always be 
less than the value above:
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(theta):
    return argmax[theta] * Sigma[i] * E[z[i] in Q[i]](log((P(x[i], z[i]|theta) / (Q[i]*z[i]))))
f
#%% [markdown]
"""
unpacking that, we get
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(theta):
    return argmax[theta] * Sigma[i] * Sigma[z[i]] * Q[i](z[i]) * (log((P(x[i], z[i]|theta) / (Q[i]*z[i]))))
f
#%% [markdown]
"""
This whole function is just a function of theta. x's are fixed, z is just the 
possible values of x  that we're summing over, and Q(z) is the probability of 
those values. Theta is the only thing that we actually have to input. 

We're trying to find the optimal values L(theta). But we've turned the problem
into argmaxing this other, nastier function of theta instead. And by Jensen's 
inequality, we can be sure that the value of theta that we get from this uglier
function will necessarily move us in the direction we want. Dr. Ng uses the visualization
of a smaller convex function necessarily contained by a larger convex function. 
Argmaxing the ugly one gives us a value of theta closer to what we're looking
for. 

In fact, if we take the inequality further, what we actually want is 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(theta):
    return Sigma[i] * E[z[i] in Q[i]](log((P(x[i], z[i]|theta) / (Q[i]*z[i])))) == Sigma[i] * log(E[z[i] in Q[i]]*((P(x[i], z[i]|theta) / (Q[i]*z[i]))))
f

#%% [markdown]
"""
In order for that to happen, 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(theta):
    return (P(x[i], z[i]|theta) / (Q[i]*z[i])) == C
f

#%% [markdown]
"""
We haven't actually specified the _form_ of the distribution Q here, so we're
pretty much free to choose whatever distribution we want. And the next trick is 
to set it so that 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return (Q[i]*z[i]) == P(x[i], z[i]|theta) / (Sigma[z[i]] * P(x[i], z[i]|theta))
f

#%% [markdown]
"""
and now, there's a couple steps skipped here, but we can show that 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return (Q[i]*z[i]) == P(z[i] | x[i], theta)
f

#%% [markdown]
"""
So, to summarise, in the E-step, we set
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return (Q[i]*z[i]) == P(z[i] | x[i], theta)
f
#%% [markdown]
"""
and then in the M-step, we construct our ugly function 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(theta):
    return theta == argmax[theta] * Sigma[i] * Sigma[z[i]] * Q[i](z[i]) * (log((P(x[i], z[i]|theta) / (Q[i]*z[i]))))
f

#%% [markdown]
"""
and update theta. 
"""

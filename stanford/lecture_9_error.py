#%% [markdown]
"""
# Objective

It looks like we're going deep into the theory of machine learning today. 
"""

#%% [markdown]
"""
### Setup & Assumptions

There are two assumptions we rely on generally. 

1. There exists a data distribution `D` where `(x,y)` are sampled. i.e. `(x,y) ~
   D`    
2. All samples are created independent from each other by the same data
   generating process `D`  

The data is born of a data generating process. We try to find a learning algo
that can learn the process, and this is how we get our "hypothesis" function. 
The data is generated via a random variable, this feeds into a deterministic 
function to find `h(theta)`, and that implies that `h` itself is a random 
variable. And therefore `h` itself follows some distribution, and we call that
distribution the "Sampling Distribution". 

If we're going deep in the statistics, we assume that there is an `h*`, the 
_true_ parameterization of the data generation process. But we never really 
see it, or know it, with any kind of certainty. It's a parameter that we 
assume to help us work out the math. 
"""

#%% [markdown]
"""
### Bias-Variance Tradeoff

The bias-variance tradeoff is best illustrated by viewing the parameter space.
Assuming there is a true parameter `theta*`, there's a clear analogy between
`accuracy:precision :: bias:variance`. Bias is error from the true parameter,
variance is the spread of all of our attempts to estimate `theta`. In fact, bias
and variance are just properties of the 1st and 2nd moments of the sampling
distribution. 

Bigger data only means that variance comes down. Any bias inherent in the
learning algorithm will stay. And we get some new terminology. 

*Efficiency*: As the number of obs goes to infinity, the rate at which the 
variance drops to zero. 
*Consistency*: As the number of obs goes to infinity, the rate at which the 
bias goes to zero. 

If your algorithm has high bias, you're never going to get to `theta*`. And 
similarly, if your algorithm has high variance, it's easily swayed by noise 
in your data. 

So when fighting variance, we have a couple options. 

1. Increase the amount of data.  
2. Regularization - injecting a small amount of bias to get a larger reduction
   in variance. 

Assuming there is some true hypothesis `g`, it may not actually be best
represented by the class of algorithms you're using. The goal, of course, is to
pick a class of learning algorithms that can get us to `g`, but we'll settle for
being able to be as close as possible. This absolute best-class estimate given
we're operating in some finite space given by the chosen algorithm class is 
called the "approximation error". And what we actually get from a finite sample
is called the "estimation error". 

The way regularization works is it shrinks the space. Not just hypothesis works
now, only certain hypotheses. If the regularization is formulated correctly, 
this will "recenter" our space of eligible hypotheses, but also "shrink" it. And
variance is measured in area, while bias from `g` is a linear distance, so this
tradeoff is often worth it. 

I won't lie, I've seen this material presented better elsewhere. There's a good
mental model here, but it doesn't "show" why regularization is often beneficial
in the same way that I've seen it presented before. 
"""


#%% [markdown]
"""
### Empirical Risk Minimizer

We try to find an estimate in the class of hypotheses that minimize the training
error. 
"""

#%% [markdown]
"""
### Uniform Convergence

There are two core questions here. 

1. If we just do empricial risk minimization, what does that give us in
   generalization error?
2. How does the generalization error of the learned hypothesis compare to the
   error of the best possible hypothesis?

To address this, we're going to use a couple of tools. 

1. The Union Bound - For events `A1, A2, ... Ak`
"""

#%% [python]
import latexify

@latexify.expression(use_math_symbols = True)
def f(x):
    return P(A[1], A[2],...,A[k]) <= Sigma[i] * P(A[i])
f

#%% [markdown]
"""
This is true whether A is independent or not, and is proven in most intro to 
probability courses. We also have

2. The Hoeffding Inequality - Let `Z1, Z2, ..., Zm ~ Bern(phi)`
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return P(abs(phi[est] - phi) > gamma) <= 2*e**(-2*gamma**2*m)
f
#%% [markdown]
"""
where
"""
#%% [python]
@latexify.function(use_math_symbols = True)
def f(x):
    return phi == (1/m)*Sigma[i]*Z[i]
f
#%% [markdown]
"""
That's a lot of math, let's translate. For a number of events that can be
classified `[0, 1]`, if we can show that they follow a Bernoulli distribution
(where the param is simply the average of events), then we can show that the
error between the estimate and the true value is bounded. And that bound is a 
function of the size of the data. 

Honestly, I'm giving up on the TA here. This is fairly complex material, and
he's not connecting the math concepts to intuitions. And you can tell he's lost
the whole class, too. I get that we can mathematically show something similar to
the Law of Large Numbers here, where we can show that for a set of events
labeled `[0,1]` we can show generally that our estimate will always remain in
some bound. But I feel like he's presenting it all in a very non-intuitive way.
And not even in a way where I can follow the logic of the math either. This
lecture is just... bad. 

This feels like the kind of thing that's either never even relevant in practice, 
or if it is, you'd never really know it. 

Somehow, we get to a rule where "with probability at least 99%, the margin of
error between the empirical risk and the generalization risk is going to be 
less than [something] as long m is greater than some formula." I'm being told
this is "something actionable". I don't see it. 

"""


#%% [markdown]
"""
### VC Dimensions

Supposedly, this is some kind of generalization of the above but to a space of
infinite hypotheses. It's super rushed. 
"""
#%% [markdown]
"""
# Objective

The goal here is to first introduce the difference between Generative and 
Discriminative approaches to modeling. We do this primarily through the 
framework of Gaussian Discriminant Analysis. Looking at this generative model
opens up two avenues for discussion

1. The comparison between GDA and Logistic serves to underscore several themes
about assumptions and computation time that will recur later. 
2. GDA serves as a useful platform for understanding Naive Bayes, since the 
likelihood functions are so similar. 

### Gaussian Discriminant Analysis

One way to think about logistic regression is using gradient descent to search
for a line that determines your classification boundaries. Generative models
work a little bit differently - we build one model for the 1s and another for
the 0s. 

**Discriminative**: Learn `P(y|x)`, `map x -> y directly`  
**Generative**: Learn `P(x|y)` and `P(y)`, and uses Bayes rule to deduce 
"""

#%% [python]
import latexify

@latexify.function(use_math_symbols = True)
def P(x):
    return P(x|y==1)*P(y==1) + P(x|y==0)*P(y==0)
P

#%% [markdown]
"""
In GDA, we assume that `P(x|y)` is M.V. Gaussian. 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def z(x):
    return z is N(mu, Sigma)
z

#%% [python]
@latexify.expression(use_math_symbols = True)
def z(x):
    return E(z) == mu
z

#%% [python]
@latexify.expression(use_math_symbols = True)
def z(x):
    return cov(Sigma)
z

#%% [python]
from math import sqrt, exp
@latexify.function(use_math_symbols = True)
def P(z):
    return 1/((2*pi)**(n/2)*abs(Sigma)**(1/2)) * e**(-1/2*(x-mu)**T*Sigma**(-1)*(x-mu))
P

#%% [markdown]
"""
So in GDA, we assume that `P(x|y=0)` and `P(x|y=1)` are both Guassian. And 
furthermore, we assume the same covariance structure in both. You don't _have_ 
to, but it's very common. And then we assume
"""

#%% [python]
@latexify.function(use_math_symbols = True)
def P(y):
    return phi**y * (1-phi)**(1-y)
P

#%% [markdown]
"""
So the optimization we do here is to maximize the joint likelihood, where 
likelihood is defined as 
"""

#%% [python]
@latexify.expression(use_math_symbols = True)
def L(phi, mu, Sigma):
    return L(phi, mu[0], mu[1], Sigma) == Pi[i==1] * P(x[i], y[i], phi, mu[0], mu[1], Sigma)
L

#%% [markdown]
"""
And contrast this with the normal discriminative likelihood function for a sec,
Generative approaches concern themselves with the joint distribution,
Discriminative with a conditional distribution. I.e. `P(y,x)` vs `P(y|x)`. 
Spoiler alert, doing this yields MLE estimates of 
"""

#%% [python]
@latexify.expression(use_math_symbols = True)
def phi(y):
    return (
        phi == Sigma[i]*int(y[i] == 1) / m
        ,mu[0] == Sigma[i]*(y[i] == 0)*x[i] / (Sigma[i]*(y[i] == 0))
        ,mu[1] == Sigma[i]*(y[i] == 1)*x[i] / (Sigma[i]*(y[i] == 1))
        ,Sigma == 1/m * Sigma[i]*(x[i] - mu[y[i]])*(x[i]-mu[y[i]])**T
    )
phi

#%% [markdown]
"""
And now predicting a given class label for a new patient is just determining
"""

#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return argmax[y] * P(y|x) == argmax[y] * (P(x|y) * P(y) / P(x))
f

#%% [markdown]
"""
Here, too, `P(x)` is only necessary when you actually need the probabilities. It
can be omitted if you're only looking for a class prediction, because it's just
a normalizing constant. 

One neat thing that Dr. Ng illustrates too, is that GDA _implies_ a decision 
boundary that is very very close to the one logistic regression gives you, at 
least in a toy 2-d example. But the method to get there is very different. 

In answering a question, he also addresses the fact that you can parameterize it
with separate `Sigmas` for each distribution. When you do, the decision boundary
is no longer necessarily linear. And you double parameters, increase computation
time, etc. 
"""

#%% [markdown]
"""

### Generative and Discriminative Comparison

First, Dr. Ng illustrates that both logistic and GDA use sigmoid functions in
mapping ` x -> P(y=1)`, and alludes to a homework problem where you prove that.
And, in fact, you can illustrate that if you have a GDA, there is an equivalent
logistic regression. However, this does not work in reverse. Ultimately, GDA has
stronger assumptions than logistic regression. 

This general relationship holds even as we vary the distribution. If we assume
GDA but where `x|y ~ Poisson`, this still implies a logistic boundary. This works
for any distribution that belongs in the exponential family, where `P(y=0)` and 
`P(y=1)` differ only by the distribution parameters. 

One of the neat consequences of this is that logistic regression holds
independent of the distribution we're assuming. Hence, we don't really need to
worry about the distribution of our data. This is probably why logistic
regression is such a work-horse. 

Dr. Ng admits he uses GDA where there are significant concerns about efficiency,
or where we're expected to build a large number of models. The algo for GDA is
much more efficient, and he notes that we see this as a recurrent theme:
stronger assumptions lead to more efficient computation but less precise
estimates, and weaker ones take longer but are more precise. 
"""

#%% [markdown]
"""
### Naive Bayes

Naive Bayes is the entry level problem for text classification. And so we'll 
start by looking at spam classification as a motivating problem. To get there, 
though, we have to start talking about data conventions for text problems. 

In theory, our feature vector `x` would be every single word in the English
language. In practice, we usually choose about the top 10,000 words of the
corpus. `x` for a given doc is expressed as a binary vector illustrating whether
a word appears in a document or not. 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return x[i] == int(i in email)
f

#%% [markdown]
"""
Now, we assume `x[i]` is conditionally indepdendent given y. This is...  a 
really strong assumption, but in some text problems, it kindof works. In english, 
what this means is that given an email is spam, no word is more probable given
any other word in the document. In math, what it means is that we can simplify
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def P(x):
    return P(x[1], x[2],...,x[10000]) == Sigma[i==1] * P(x[i]|y)
P

#%% [markdown]
"""
So the parameters of the model are 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return phi[j|y==1] == P(x[j] == 1 | y == 1)
f
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return phi[j|y==0] == P(x[j] == 1 | y == 0)
f
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return phi[y] == P(y==1)
f

#%% [markdown]
"""
And the joint likelihood is 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return L(phi[y], phi[j|y]) == Sigma[i]*P(x[i],y[i],phi[y],phi[j|y])
f
#%% [markdown]
"""
And your maximum likelihood estimates for these parameters are
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return {
        phi[y] == Sigma[i]*int(y[i] == 1) / m
        ,phi[j|y==1] == Sigma[i]*int(x[i,j] == 1, y[i] == 1) / (Sigma[i]*int(y[i] == 1))
    }
f
#%% [markdown]
"""
Is logistic regression better at classifying? Yes, almost always. But Naive Bayes
is good enough. And not just good enough, the maximum likelihood estimates are
based just on counts. And _that means_ it's quick to train, and you can update 
it incrementally as you get new emails. And those are some powerful advantages. 
"""
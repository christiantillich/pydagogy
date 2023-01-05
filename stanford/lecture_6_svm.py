#%% [markdown]
"""
# Objective
"""

#%% [markdown]
"""
### Naive Bayes review

`x` is a vector that denotes participation in a dictionary of `j` items. We're
tasked with building our generative model by finding an expression for `P(x|y)`
and `P(y)`. In GDA, we assume that both take on the form of a normal distirbution.
In NB, we assume instead that 
"""
#%% [python]
import latexify
@latexify.expression(use_math_symbols = True)
def f(x):
    return P(x|y) == Pi[j]*P(x[j]|y)
f
#%% [markdown]
"""
In English, the probability of seeing a group of words given a specific document
type is just the product of the probability of seeing any individual word given
the document type. And by finding `phi_j|0` and `phi_j|1` given MLE and by using
Bayes' Rule, we have everything we need to solve for `P(y=1|x)`
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return P(y==1|x) == P(x|y==1)*P(y==1) / (P(x|y==1) * P(y==1) + P(x|y==0)*P(y==0))
f
#%% [markdown]
"""
Dr. Ng points out, though, that if you've never seen an email with certain words
before, though, `P(x|y==1) = 0` as well as `P(x|y==0) = 0`. The whole thing 
ends up being `0 / (0+0)`, which is _not great_, and we have a real problem 
philosophically by saying something has a 0% probability just because we haven't
seen it yet. 

### Laplace Smoothing

This is really just a small trick we can pull to correct for this problem. Let's
say we've got a ML estimator that just takes the form `#x==0 / (#x==0 + #x==1)`
and a data set where `#x==0` or `#x==1` evaluates to zero. If we add a marginal
amount to every term - say 1 - we can take something that was originally `0 / (0
+ 4)` and turn it into `1 / (1 + 5)`. It's good prophylaxis against divide by
zero errors. Laplace used this to correct for the probability of seeing the sun
rise, as an example. Dr. Ng uses the example of a bad football team (well, bad
year anyways). 

If we apply this to the ML estimates for Naive Bayes, we get
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return phi[i|y==0] == Sigma[i]*(int(x[j,i] == 1 & y[i] == 0) + 1) / (Sigma[i]*int(y[i] == 0) + 2)
f

#%% [markdown]
"""
Also small note, we can take any continuous variable and bucket, and if we want
to apply NB to continuous variables, just bucket them and run it. 

### Variations on Naive Bayes

We lose information when we encode `x in [0,1]`. This is called the Multivariate
Bernoulli Event Model. But we have other ways of encoding `x` to help preserve
that information. 

The Multinomial Event Model encodes `x` as a much smaller and variable length
vector, where the value of `x[i]` is the position in the dictionary, and `len(x)`
is the length of the document. And if we set up the Naive Bayes model again
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return P(x,y) == Pi[i] * P(x[j]|y) * P(y)
f
#%% [markdown]
"""
The math isn't fundamentally changing, but the intepretation is. `P(x_j|y)` is
now a multinomial probability, hence the name. When `x` is encoded `[0,1]`, 
`P(x_j|y)` is the probability of seeing word `j` in a doc classified as `y`. 
When `x` is encoded as document position, `P(x_j|y)` is the probability of seeing
word `j` in position `i` in the doc. This encoding helps us retain some sense of
positional relevance, although there's probably better ways still. 

Similarly, our MLE estimates are similar, but 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return phi[k|y==0] == (Sigma[i]*int(y[i]==0) * Sigma[j]*int(x[j]==k) + 1) / (10000 + Sigma[i] * int(y[i] == 0) * n[i])
f
#%% [markdown]
"""
now the expression says, roughly, 

```
Look at all the words in all non-spam emails (y=0). What fraction is a specific word? That is the MLE estimator for that word in the non-spam emails. 
```

Note we added the laplace smoothing here, but now the denominator term is
10,000. This is the number of possible values for x. Honestly, I'm kindof
confused by this. I thought the value was determined by the different possible
values of `y`, not `x`. Need to dig into this more. 

Note that there are other ways of framing `x`. Term frequencies is another
possible approach, as are word embeddings. It might be useful to revisit Naive
Bayes in some of these other framings, but that's left to us. 

### Career Advice Tangent

If you're doing applications, really focus on getting a simple algorithm up 
and running in a quick and dirty way. Don't spend weeks designing what you want 
to do, start by getting a data set and doing NB or Logistic and see what you 
can learn. Move fast and break things. 

### Support Vector Machines

So let's suppose our goal now is to find _non-linear_ decision boundaries. This
is...   harder. We could do it by framing logistic regression, but with 
quadratic terms and interactions. But doing so requires a lot of data, is 
very computationally expressive, and is probably going to start leading into 
the realm of regularization on a Logistic framework. 

SVM can do this, though, and without a lot of parameterization. But we need to 
start with some definitions: 

**Optimal Margin Classifier (separable)**: The decision boundary that sits the
maximum possible distance away from the nearest point on both sides of the 
boundary. 

**Kernel**: A mapping between some base set of features, and some higher 
dimensional feature space. This can even be a set of infinite features. 

**Functional Margin**: Think about this the band of space that fills out normal
to the decision boundary. If the decision boundary is the line, the functional
margin is the width of that line, and this is the thing we will be trying to 
optimize. 

**Geometric Margin**: The actual Euclidian distance to an observation as defined
by the Functional Margin between that observation and the decision boundary. 

Ultimately, the goal of SVMs are about finding the Optimal Margin classifier. It 
does so by trying to find the Geometric Margin that sits as far away from the 
closest point on either side of the line as possible. 
"""

#%% [markdown]
"""
We're going to frame our prediction function as 
"""
#%% [python]
@latexify.function(use_math_symbols = True)
def h(x):
    return g(w**T*x[i] + b)
h
#%% [python]
@latexify.function(use_math_symbols = True)
def g(z):
    if z >= 0:
        return 1
    else:
        return -1
g
#%% [markdown]
"""
This means that unlike logistic regression, we're not trying to find a
probability. We're just outputting a classification directly. When 
`w**T*x[i] + b > 0`, it gets labeled +1, otherwise -1. 

Note that the way we frame the inner term isn't actually any different then
`Theta^T*X`, but I suspect it'll be more convenient later. We define the
functional margin of a hyperplane as
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return gamma == y[i]*(w**T*x[i] + b)
f
#%% [markdown]
"""
Right now, we're assuming that the training set is completely linearly
separable. We're going to relax this in lecture 7, but it's easier to see what's
happening by starting with the easier case. 

w,b are usually normalized, and if you don't do it, you can actually just cheat
and scale w & b linearly to get larger/smaller numbers for `gamma` without
actually moving the boundary. 

So we add the normalization to the functional margin to keep us from cheating in
this way. 
"""
#%% [markdown]
"""
Now, all of this functional margin talk deals with the relationship between
the decision boundary and a single point, in theory. We make a small modification
as we talk about the relationship between the decision boundary and a training
set. Specifically, the functional margin with respect to a whole set of data is
defined as
"""

#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return Gamma == min(gamma[i])
f

#%% [markdown]
"""
So, and I think Dr. Ng kindof glosses over this a bit, but what I'm gathering
is that the Geometric Margin is actually the Euclidian distance to some observable
value, and the Functional Margin is...  something else. Some arbitrary vector 
pointing normal to the decision boundary in the direction of the closest obs?
I'm not really sure. But he's pretty explicit that unlike with the Functional 
Margin, the Geometric Margin gives us an actual distance. And the calculation 
for that is proved in the lecture notes, but stated as
"""

#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return Gamma[i] == y[i]*(w**T*x[i] + b) / abs(abs(w))
f

#%% [markdown]
"""
And, much like with the Functional Margin, w.r.t. the training set, we look
for the Geometric Margin to the closest point on either side of the decision 
boundary. 

Dr. Ng uses hats to denote the difference, where the hatted `gamma` is the 
functional margin. Here I will use a capital Gamma, because I don't have hats. 
But we now define our Optimal Margin Classifier as
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return argmax[w,b]*Gamma
f

#%% [markdown]
"""
Choose `(w,b)` to maximize the geometric margin. 

We could do this two ways. The first is a constrained optimization where we
try to pick a `Gamma` that's maximized so long as the Euclidean distance from 
every single training observation is at least as big as `Gamma`
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return max[[ {gamma, w, b} for gamma,w,b in (y[i]*(w**T*x[i] + b) / abs(abs(w)) > Gamma)]]*Gamma
f
#%% [markdown]
"""
In other words, find the Geometric Margin so that each point in the training set
lies at a distance at least the length of the Geometric Margin you chose. 

So this is one way to maximize our worst-case geometric margin. But this is not
convex, and it's really unweildy, so there's actually an equivalent expression 
that's a bit better. 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return min[[ {w,b} for w,b in (y[i](w**T*x[i] + b) >= 1)]]*abs(abs(w))**2
f
#%% [markdown]
"""
Choose `(w,b)` to minimize the squared norm of `w` but where the functional
margin for each training observation is at least 1. The proof of this is
non-trivial, but this alternative expression _is_ convex and thus is much easier
to solve. 
"""


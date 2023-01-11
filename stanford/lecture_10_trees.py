#%% [markdown]
"""
# Objective

Review decision trees and ensemble methods, and discuss why they tend to be 
utilized together. 
"""

#%% [markdown]
"""
### Decision Trees

Greedy, top down, recursive partitioning is the name of the game. We're ultimately
looking for a split function `s[p]` (or region `R[p]`) where
"""

#%% [python]
import latexify 
@latexify.expression(use_math_symbols = True)
def f(x):
    return s[p](j,t) == { X | X[j] < t, X | X[j] >= t }
f

#%% [markdown]
"""
In other words, a threshold `t` for a variable `j` is going to define two 
different values of X, one above the threshold and one below. Our tree is going
to be a list of these splits working recursively through the parameter space until
neat separations can no longer be made. So how do we find them?

Well, we can start by defining a loss function
"""

#%% [python]
@latexify.function(use_math_symbols = True)
def L(R):
    return 1 - max[c]*p[c]
L

#%% [markdown]
"""
Where we have `C` classes and `p[c]` is the proportion of examples in R that 
are in class `c`. We can actually break out the loss of a given branch as 
"""

#%% [python]
@latexify.function(use_math_symbols = True)
def L(R):
    return max[j,t] * (L(R[p]) - (L(R[1]) + L(R[2])))
L

#%% [markdown]
"""
And we can even ignore the loss of the parent, since it is already defined. So
at any step of the algorithm, we're really just interested in the loss of the 
children we're going to be defining (1 and 2). 

Oops, psych out. Misclassification Loss is actually a bad choice.
Misclassification Loss has no incentive to put more of the same class into a
bucket when it finds some decision boundary that already puts 100% of a single
class into that bucket. It's not "sensitive enough", and the example we go over
shows that with 900+ and 100- cases, there's no difference in `L` between having
a split with 200+/0- and 500+/0-. The loss in each case is exactly 100. So
instead we look to cross-entropy loss. 
"""
#%% [python]
@latexify.function(use_math_symbols = True)
def L(x):
    return Sigma[C] * p[c] * log[2] * p[c]
L

#%% [markdown]
"""
We don't really define why this loss-function is applicable, but we can kindof
see geometrically why it works. Loss over `p` is concave, so if child 1 defines
a region of lower `p`, and if child 2 defines a region of higher `p`,
necessarily the loss of the average of the two is lower than the loss of the
parent it was split off from. 

Misclassification error isn't concave, it's pyramid-shaped. And because of that, 
the loss of the parent is often equal to the average loss of the two childrend.

There are other losses too, most all of them are concave. The Gini Loss is 
`p[c] * (1 - p[c])`. There are others. 

We can do regression too. The tree model predicts the average of `y`, and we
use the squared loss function
"""
#%% [python]
@latexify.function(use_math_symbols = True)
def L(x):
    return Sigma[i in R[m]] * (y[i] - y[m])**2 / abs(R[m])
L

#%% [markdown]
"""
Decision trees are one of the few learning algorithms where multiclass categorization
is actually a real natural choice. However, split categories go as `2**q`, though
so the calculation increases exponentially as we add more categories. So be 
careful with really high-ordered classification problems. 

### Regularizing Trees

We have five options:

1. Have a minimum leaf size.  
2. Have a maximum depth.  
3. Have a maximum number of nodes.  
4. Establish some threshold decrease in loss.  
5. Prune using a misclassification rule with some validation set.  

It's interesting that none of this is really a penalization on the loss function, 
it's all parameters or some kind of configuration of the algorithm. 

### Runtime

Let's say we have `n` training examples, `f` features, and the 
current tree depth is `d`. At test time, runtime is `O(d)`. And `d < log(n)`, so
all this goes pretty quick. 

Train time is the bulk of the cost. Each point is part of `O(d)` nodes, and 
the cost of each point at each node is `O(f)`. So the total cost is `O(n*f*d)`, 
for n points ordered linearly across `f` features and looked at `d` times.

### Downsides

First, there's no additive structure. Meaning that approximating a behavior
that's linearly separable by a diagonal like is...   convoluted at best. So 
there's no substitute for knowing the shape of your data. 

They're also high variance, typically. So it's easy to overfit. 
"""

#%% [markdown]
"""
### Ensemble Methods

Let's say we have a bunch of random variables `x[i]` that are IID. Let's say
`V(X[i]) = sigma**2`. It's possible to show that 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return V(E(X)) == V((1/n)*Sigma[i]*X[i]) == sigma**2/n
f
#%% [markdown]
"""
But even if we were to drop the independence assumption, we can still show that
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return V(E(X)) == rho * sigma**2 + (1-p)/n*sigma**2
f

#%% [markdown]
"""
Now, there are a number of different ways to ensemble. 

1. Group a bunch of different algorithms together.  
2. Incorporate different training sets.  
3. Bagging (Random Forest)  
4. Boosting (Adaboost, xgboost)  

Bagging is actually also known as "Bootstrap Aggregation". We use bootstrapping
typically when trying to calculate the uncertainty in an estimate when it doesn't
have a convenient closed-form. We have some true population `P` and a sample
training set `S ~ P`. Bootstrapping assumes that the population _is_ the training
sample `P = S`, and we take multiple samples `Z ~ S`. And so by carving up 
our training set, and fitting the parameters in question for each, we can talk
about the mean and standard dev. of our parameters, and thus talk about the 
uncertainty in measuring them. 

### Bagging

Now, when bagging, we're not interested in the uncertainty so much as the average
estimate of the output. So we take segments of the data `Z1, Z2, ..., Zn`. We
train model `G[m] on Z[m]`. And our "meta-model"
"""

#%% [python]
@latexify.function(use_math_symbols = True)
def G(m):
    return Sigma[m]*G[m](x)/M
G

#%% [markdown]
"""
By bootstrapping, we drive `V(E(X))` closer to just `rho*sigma**2`. Bootstrapping
is just going to drive the variance down to its theoretical minimum. So by adding
more models M, we can asymptote out the variance. But this isn't cost-free. We
do increase the bias. Because `Z`s are all fairly highly correlated with each
other, because they're all still sampled from the same population. 

### Random Forests

So Random Forest is decision trees + bagging. Any given DT model is high variance, 
low bias. Thus, they make a great candidate to bag; since most of our model error
comes from the variance end, it's worth it to trade a small amount of bias for
a reduction in variance by adding more trees. 

And actually, we could bag trees and still not be at Random Forest. Random Forest
makes the additional decision that we only consider a fraction of the _variables_
as well as the training sample. This decision gets around a theoretical maximum
variance reduction driven by just more models, and actually decreases `rho` towards
a theoretical minimum as well. So the game is decorrelating models, and RF does
this by sampling features _and_ observations. 
"""


#%% [markdown]
"""
### Boosting

Boosting is more about decreasing model bias, and we do this by having each 
new tree predict on the errors of the last, so the final prediction is like a 
big long Taylor approximation of `y`. I've gone over XGB before in detail, in 
practice this concept is mathed out to optimize a fairly counter-intuitive 
objective function, but it's all in service of updating on the errors. 
"""
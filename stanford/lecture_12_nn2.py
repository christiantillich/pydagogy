#%% [markdown]
"""
# Objective

In the previous lecture, we kindof set up some of the mathematical formalism
of neural nets, and began establishing the Loss/Cost functions. We'll pick up
where we left off, and go through the gory math of backpropogation in more 
detail. 

### Backprop - Batch

First off, we start off with the cost function
"""

#%% [python]
import latexify

@latexify.expression(use_math_symbols = True)
def f(x):
    return J(y[pred], y) == (1/m) * Sigma[i] * L**{i}*(y[pred], y)
f

#%% [markdown]
"""
and we have the loss defined for a given level `{i}` as 
"""

#%% [python]
@latexify.expression(use_math_symbols = True)
def L(x):
    return L**{i} == -[y**{i}*log(y[pred]**{i}) - (1-y**{i})*log(1-y[pred]**{i})]
L

#%% [markdown]
"""
We were focusing specifically on a 3 layer network, 1 hidden layer. And we decided
to move forward with gradient descent, and starting from the parameters closest
to y. 

The good news is that nothing's really changed. The formulas are still 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return w**{l} == w**{l} - alpha * (delta*J / (delta * w**{l}))
f
@latexify.expression(use_math_symbols = True)
def f(x):
    return b**{l} == b**{l} - alpha * (delta*J / (delta * b**{l}))
f
#%% [markdown]
"""
for `l in [1,2,3]`. If we start with w3, this is actually the simplest approach. 
We're really just asking "how much should we move `w3` in order to get maximally
closer to y", and `w3` has fewer moving parts. 

The bad news is that things are about to get really ugly. We can (and are forced
to) write out the change in the loss relative to `w**{3}`.
"""

#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return delta*L / (delta*w**{3}) == -[y[i] * (delta/(delta*w**{3}))*(log(sigma(w**{3}*a**{2} + b**[3]))) + (1-y[i]) * (delta/(delta*w**{3}))*(log(1 - sigma(w**{3}*a**{2} + b**[3]))) ]
f

#%% [markdown]
"""
Now, we're gonna simplify things up by subbing in `a**{3}` here, and we also 
need to know that the derivative of the sigmoid function is
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return delta*sigma(x)/(delta*x) == sigma(x) * (1-sigma(x))
f

#%% [markdown]
"""
Combining these two, we rephrase the derivative above as
"""

#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return -[y**(i) * (1/a**{3}) * a**{3} * (1-a**{3}) * a**({2}*T) + (1-y[i]) * (1/(1-a**{3})) * -a**{3} * (1-a**{3}) * a**({2}*T)]
f
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return -[y**(i) * (1-a**{3}) * a**({2}*T) - (1-y[i]) * a**{3} * a**({2}*T)]
f
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return -[y[i]*a**({2}*T) - a**{3}*a**({2}*T)]
f
@latexify.expression(use_math_symbols = True)
def f(x):
    return delta*L / (delta*w**{3}) == -[(y[i] - a**{3}) * a**({2}*T)]
f

#%% [markdown]
"""
So even though it starts off pretty ugly, that actually gets pretty manageable. 
And the total cost then is just the sum
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return delta*J / (delta*w**{3}) == (-1/m) * Sigma[i] * -[(y[i] - a**{3}) * a**({2}*T)]
f

#%% [markdown]
"""
And then, we just take this, and plug it into the gradient descent rule to work
out new values of `w**[3]`. 

Now, let's do the loss for `w*[2]`. But let's do it _lazily_. From the chain 
rule of calculus, we can write the change in loss from `w**{2}` as 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return delta*L/(delta * w**{2}) == (delta*L/(delta * a**{3})) * (delta*a**{3}/(delta * z**{3})) * (delta*z**{3}/(delta*a**{2})) * (delta*a**{2}/(delta*z**{2})) * (delta*z**{2}/(delta*w**{2}))
f

#%% [markdown]
"""
and the trick to know here is that the first two terms of this expression is 
actually `dL/dw**{3}`. And the rest are actually fairly simple to evaluate on
their own. So 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return delta*L/(delta * w**{2}) == (a**{3}-y[i]) * a**({2}*T) * (delta*z**{3}/(delta*a**{2})) * (delta*a**{2}/(delta*z**{2})) * (delta*z**{2}/(delta*w**{2}))
f
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return delta*L/(delta * w**{2}) == (a**{3}-y[i]) * a**({2}*T) * (w**({3}*T)) * (delta*a**{2}/(delta*z**{2}))  * (delta*z**{2}/(delta*w**{2}))
f
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return delta*L/(delta * w**{2}) == (a**{3}-y[i]) * a**({2}*T) * (w**({3}*T)) * (a**{2} * (1-a**{2})) * (delta*z**{2}/(delta*w**{2}))
f
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return delta*L/(delta * w**{2}) == (a**{3}-y[i]) * a**({2}*T) * (w**({3}*T)) * (a**{2} * (1-a**{2})) * (a**({1}*T))
f

#%% [markdown]
"""
This actually isn't entirely correct. The shape is wrong, and there's a more
rigorous method in the notes that will capture the shapes correctly. We could
also use shape analysis to notice that one part of the multiplication should be
element-wise, and kindof work backwards into the correct answer. He confesses
though that there's no real need to actually do this anymore, NN frameworks handle
this very well. 

But, ultimately, this is the gambit. On any feed forward network, we can write
the change in the loss from some parameter as a chain of derivatives, each piece
easily calculable given that you've already done everything up to that point. 
Hence, backprop. He leaves `w**{1}` as an at-home exercise. 

The TA also notes that if we were to calculate everything sequentially, we often
find ourselves needing the values that were easily calculated in the prediction
part of the calculation - e.g. `a**{1}`, `a**{2}`, and so on. So a real NN
framework is going to cache all these values when forward predicting, so they're
accessible when backpropping. 

### Improvements

Most of the time, these simple feed forward networks are garbage. We're going to 
need to do the following if we want any hope at solving real problems

1. Activation Functions - we have a lot to play with here, some examples:
    * sigmoid - good for probabilities, can interpret as a percent. However, high
        z values tend not to update, because gradient vanishes. 
    * ReLu - Gradient does not saturate, so this works real well with outlier obs.
    * tanh - same gradient saturation as sigmoid, but the range of output is [-1,1]. 
        So if penalization is really important for us, this may be better than
        sigmoid. 
    * Identity - If you do this, you aren't actually doing anything different than
        regression. Your `w` and `b` ends up being linear combinations of everything
        that came before. Complexity in the model _comes from activation functions_. 
2. Initialization Methods
    * Normalization - just center and express as std. devs from the mean. Great
        for our activation functions that saturate at high-z. Also speeds up
        computation, since the path of greatest descent always points to the 
        center. But remember productionalizing means saving a snapshot of mu
        and sigma during training. 
    * Weight Initialization - for sigmoids, `R * sqrt(1/n**(L-1))`, where `R` is
        a vector of random values of the length of `w`. There's also Xavier and
        He initialization, which are similiar formulas but slightly modified. 
        All of this is in the service of starting in a place where our gradients
        are not saturated. It must be random, though - if you choose the same
        number for every weight, you run into what's called the "symmetry problem". 
3. Regularization

### Exploding Gradients

Assume a 2-feature NN with 10 hidden layers and an identity activation function. 
Your prediction is the multiplication of 10 weights in succession. Assuming 
weights significantly different from 1, your prediction always tends towards
infinity or 0. 

### Regularization

Previously on Gradient Descent, we talked about batch and stochastic gradient 
descent. There's actually a middle road, called "mini-batch". The idea is that
in a million obs data set, we could do our gradient descent on 1000 obs, and that
would probably be "good enough" to get a general direction of "down", in a much
better way than a single obs would. 

The algo goes:
1. Select a batch, some subset of the data. 
2. Forward prop for the batch. 
3. Backprop the batch to find the weights. 

It's important to have this, because there's another algorithm called The 
Momentum Algorithm, which is a regularized take on SGD that exploits ideas from 
physics about momentum and friction. 

So we modify our update equations for w to include a velocity term. 
"""

#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return v == beta * v + (1-beta)*(delta*L/(delta*w))
f
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return w == w - alpha * v
f

#%% [markdown]
"""
We don't really go over the specifics here, but the general idea is that if
updates all point in the same direction historically, they'll tend to maintain
that size and be less likely to change from it. But if they point in different
directions historically, they'll tend to cancel each other out and future
updates in that direction will tend to be very small. 


"""


#%% [markdown]
"""
# Objective

To discuss neural networks, over the course of a 2-lecture series. We'll start
from logistic regression and move up from there. 

### Logistic Regression

Let's say we're building an app, cat-or-no-cat style. Images are 3-D matrices - 
64x64x3, where the 3 corresponds to RBG values. We flatten this matrices to a 
vector. Then we pass it to all to a single layer of the form
"""

#%% [python]
import latexify
@latexify.function(use_math_symbols = True)
def y(x):
    return sigma(w**T*x + b)
y

#%% [markdown]
"""
x has length of 12,288. The single layer predicts our single output, and that's
that. Our solving algorithm is maximum likelihood:
"""

#%% [python]
@latexify.function(use_math_symbols = True)
def L(x):
    return argmin[w,b] * -[y*log(y[pred]) + (1-y)*log(1-y[pred])]
L

#%% [markdown]
"""
And maximizing the likelihood gives our solution
"""

#%% [python]
@latexify.expression(use_math_symbols = True)
def w():
    return w == w - alpha*(delta*L / (delta*w))
w

#%% [python]
@latexify.expression(use_math_symbols = True)
def b():
    return b == b - alpha*(delta*L / (delta*b))
b

#%% [markdown]
"""
### Neural Nets

A neuron is a linear mapping of inputs and an activation. Here, we have one 
neuron. The model is defined by some architecture, and the parameters. Here, the
architecture is a single layer feed-forward net, and the parameters are `w` and
`b`. In fact, the parameters will always be `[w,b]`, but the architecture is 
going to vary dramatically. 

If we changed our goal to predict one of three things - cat/lion/iguana - we
could modify our architecture and map the inputs to three neurons, one for 
each label. We would still have one _layer_, but y is now a vector of size 3. 
We would write the formula for `y_pred` as follows:
"""

#%% [python]
@latexify.expression(use_math_symbols = True)
def y(x):
    return y[i] == a[i]**[1] == sigma(w[i]**[1]*x + b[i])
y

#%% [markdown]
"""
The brackets here now denote layers, and we have different sets of weights and
biases for `i in [1,2,3]`, where `i[1]` is our housecat, `i[2]` is our lion, and
`i[3]` is our iguana. And, in fact, we could even have images with housecats and
lions, and our model would probably still be robust to multi-instance labeling. 

We'd have to change our likelihood to accomodate this though. 
"""

#%% [python]
@latexify.function(use_math_symbols = True)
def L(x):
    return Sigma[i]*[y[i]*log(y[i, pred]) - (1-y[i])*log(1-y[i,pred])]
L

#%% [markdown]
"""
Where we just sum over the three neurons.
"""

#%% [markdown]
"""
### Exploring Softmax

Let's run with the same
"""
#%% [python]
y

#%% [markdown]
"""
But now we're going modify our architecture to incorporate softmax. Let's first
start by rewriting the our formula so that 
"""

#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return y[i] == a[i]**[1] == sigma(z[i]**[1])
f

#%% [markdown]
"""
Here, `z` is now the linear component of the calculation, and we're going to 
change our activation function `sigma` to 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return e**z[i]**[1] / (Sigma[i] * e**(z[i]**[1]*x))
f

#%% [markdown]
"""
This architecture gives a way to exclusively look at photos with _only_ a cat, 
lion, or iguana, and cannot accomodate multiple animals in the same picture. 

The likelihood function changes slightly again to be the cross entropy loss
"""

#%% [python]
@latexify.function(use_math_symbols = True)
def L(x):
    return -Sigma[i]*y[i]*log(y[i, pred])
L

#%% [markdown]
"""
And while the partials of the likelihood in the inclusive case go to 0 in all
nodes not being evaluated, here the partials get ugly. But otherwise, the 
formulas for `w` and `b` don't change. We still use gradient descent where the
values are updated by the partial derivatives. 

### Layers

Let's say we do a 3-layer feed forward net now. 

![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR6a_PnBXhfv3T_GCeV3DeOZM_EUAHEgk7XZu_CXK7_Rp4DAigwhD8mGZNhUFKZiXujp74&usqp=CAU)

Now we have 3 layers. One input layer, one output layer, one hidden layer. The 
theory here is that our input layer detects very fundamental properties of the
image, such as edges or boundaries. The hidden layer then composes those features, 
like edges, and put them together to compose things like "ears". And then the 
output composes those features like "ears" into "cat-or-no-cat". 

But it rarely comes out that clean. You have no guarantee that the network finds
ears, only that with enough sample you're gonna get "cat" as an output. 

In practice, we could either use fully connected models (every input to every
output), or manually connect certain features to the layers. The latter option
gives us a model where we can start to interpret the hidden layers. The former
typically gives us much better performance. 

### Propogation Equations

Let's start off with the equation for the linear component of our first layer. 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return z**[1] == w**[1]*x + b**[1]
f
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return a**[1] == sigma(z**[i])
f
#%% [markdown]
"""
z here is a 3-valued vector. w a 3xn matrice, x an n-length vector, and b a 
3-valued vector too. 

a is also length 3 (there are 3 features output to the hidden layer). 

If we go to the next level
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return z**[2] == w**[2]*a**[1] + b**[1]
f
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return a**[2] == sigma(z**[i])
f

#%% [markdown]
"""
z is length 2. w is now 2x3 (3 outputs mapped to 2 hidden nodes, remember it's
the _transposed_ features). Instead of x now, we have `a[1]`, which we know was
3x1. And `a[2]` is 2x1. 

We can go into the output layer, but you get the idea. We're pretty much
recursively going through the NN structure and applying the same equations. 

### Again But With Data

This is all just formalism for a single image. So let's do matrices now where
"""

#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return X == [x**{1}, x**{2}, ..., x**{m}]
f

#%% [markdown]
"""
The notation here has changed now. Brackets on superscripts are used to denote
layer, braces will now denote observation numbers. So if we were to examine 
the size of `Z` now, we'd see
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return Z**[1] == w**[1]*X + b**[1]
f
#%% [markdown]
"""
where
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return Z[1] == [Z**([1]*{1}), ..., Z**([1]*{m}) ]
f

#%% [markdown]
"""
`Z[1]` in total would be a matrice of size 3xm, where m is the number of obs. `X`
is nxm, but `w` does not change in size. And in order to get the algebra to work, 
we coerce b (which is 3x1) to 3xn by repetition (this is called broadcasting). 
Taking this formalism allows us to repeat the propagation equations above, but
with multiple observations, but he leaves this as an exercise for home. 

### NN Loss Functions - Batch

How the hell do we optimize this thing? Well, we need to choose our model parameters. 
What are those? They are
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def P(x):
    return P == {w**[1], w**[2], w**[3], b**[1], b**[2], b**[3]}
P
#%% [markdown]
"""
And so we're going to define our loss function
"""

#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return J(y[pred], y) == (1/m) * Sigma[i] * L**{i}
f
#%% [markdown]
"""
Okay, so `L**{i}` is the loss for a single observation, and we're just adding it
up across all observations. Great, nothing new here. The loss for a single 
observation is just given by 
"""

#%% [python]
@latexify.expression(use_math_symbols = True)
def L(x):
    return L**{i} == -[y**{i}*log(y[pred]**{i}) - (1-y**{i})*log(1-y[pred]**{i})]
L
#%% [markdown]
"""
Now, the neat thing is that the derivative for `J` and the derivative for `L`
with respect to a single parameter is the exact same, since derivation is a linear
operator. 

We talk a bit about backprop at the end here, but I'm going to move all of that
to lecture 12 since that's the focus there. 

"""





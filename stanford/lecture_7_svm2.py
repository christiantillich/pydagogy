#%% [markdown]
"""
# Objective

Last lecture was really more like "foundations of SVM". These are the building
blocks, but it's not the whole thing. To get all the way, we need to think about
abstracting the feature terms. Because there's a trick we're going to do that 
relies on reframing the feature terms as an inner product, and by doing this 
trick, we can bundle in arbitrary interactions into the algo with no extra 
computation time. 

Once we have the "real" SVM in front of us, we can start to pick apart the 
assumption that the data has to be truly linearly separable. 
"""

#%% [markdown]
"""
### SVM Abstracted

So far, we've derived everything as if our problem had a reasonable number of
dimensions to it. However, suppose that assume that the weights `w` can be 
defined as a linear combination of the training features 
"""

#%% [python]
import latexify
@latexify.function(use_math_symbols = True)
def w(x):
    return Sigma[i]*alpha[i]*x[i]
w
#%% [markdown]
"""
This is a theorem called the Representative Theorem that shows that you can
assume this without any loss of generalization. The proof is pretty complex, 
though, so we gloss over it. 

We can even add `y[i]` into the summation here, if we wanted, because we're 
dealing with a classifier that just outputs +/-1. And if we do, it makes the 
math a bit easier downstream. 

We're not going to prove the Representative Theorem, but we _can_ provide some
motivating intuitions. Suppose you run a logistic regression. You start with 
`Theta == 0` and do your gradient descent and update. The formula for 
gradient descent is just
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return theta == theta - alpha*h[theta](x[i] - y[i])*x[i]
f

#%% [markdown]
"""
The trick is that if theta starts out as 0, then the first iteration is simply
a linear combination of the training values. And then the second iteration is
again a linear combination of the training values. And so on. So by induction, 
theta is always a linear combination of your training examples. This is S.G.D., 
but Batch is no different either. 

Our second intuitional argument comes from decision boundaries. If we define our
decision boundary as 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return g(w**T*x + b)
f
#%% [markdown]
"""
like we did in SVM, `w` is always normal to the boundary. `w` defines the 
direction of the decision boundary (by always facing normal to it), and b moves
the boundary without changing the angle. `w` and `b` can always be expressed 
as a linear combination of the features `x1` and `x2`, and let's say you had an
`x3` that was always 0. Even if you did, the decision boundary would work out
so that `w3` was zero, which is just another way of saying that the vector `w`
just spans the feature space of `(x1, x2)`. 

These aren't super concrete proofs, but I can kindof see it. So for now we assume
"""
#%% [python]
f

#%% [markdown]
"""
So for SVM, plug `w` in and solve
"""

#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return min[[{w,b} for w,b in y[i]*(w**T*x[i] + b)]]*abs(abs(w))**2
f
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return min[['...']]*(1/2)*(Sigma[i]*y[i]*x[i])**T*(Sigma[i]*y[i]*x[i])
f
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return min[[{w,b} for w,b in y[i]*Sigma[j]*alpha[j]*y[j]*x[i] + b >= 1]]*(1/2)*(Sigma[i]*Sigma[j]*alpha[i]*alpha[j]*y[i]*y[j]*x[i]**T*x[j])
f

#%% [markdown]
"""
And Dr. Ng makes a very heavy point to note that `x[i]**T*x[j]` is the inner 
product, and that it's very very important to the derivation of kernels. This
inner product appears both in objective as well as the constraint. And this is 
the only place the feature vectors x appear now, within this inner product term. 
_That's_ the trick - if you have a million or even infinite dimensional set
of feature vectors, but the inner product of the feature vectors is still 
capable of being calculated efficiently, then we can just plug in the inner 
product directly without ever having to touch the original feature vectors. 

Now, once again, we find ourselves with some pretty ugly derivations. This one
goes by the name of the Dual Optimization Problem. The minimization above can 
be expressed by the following instead:
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return max[[alpha for alpha in {alpha[i] > 0, Sigma[i]*y[i]*alpha[i] == 0}]] * (Sigma[i]*alpha[i] - (1/2)*Sigma[i]*Sigma[j]*y[i]*y[j]*alpha[i]*alpha[j]*InnerProd(x[i],x[j]))
f

#%% [markdown]
"""
And subbing in the definition of w now to the prediction function, we also get
"""
#%% [python]
@latexify.function(use_math_symbols = True)
def h(x):
    return g(Sigma[i]*alpha[i]*y[i]*x[i])**T*x + b
h
@latexify.function(use_math_symbols = True)
def h(x):
    return g(Sigma[i]*alpha[i]*y[i]*InnerProd(x[i],x)) + b
h

#%% [markdown]
"""
### The Kernel Trick

The Kernel Trick works as follow. 

1. Write the algorithm in terms of inner products. e.g. `<x_i, x_j>` or `<x,z>`. 
2. Let there be some mapping from `x -> phi(x)`, where phi is some higher
   dimensional space or set of features. E.g. `[x1, x2]` to `[x1, x2, x1*x2,
   x1**2]`, etc.
3. Find a way to compute kernel `K(x,z) = phi(x)**T * phi(z)` efficiently
4. Replace `<x,z>` with `K(x,z)`. 

As an example, let's suppose we have `x=[x1,x2,x3]`, and suppose we define `phi`
as all the pairwise mappings, i.e. `[x1*x1, x1*x2, ...]`

Since there are `n**2` elements, you need `O(n**2)` time to compute all the 
inner products, and then the cross product of them. But we can write the 
Kernal as 
"""
#%% [python]
@latexify.function(use_math_symbols = True)
def K(x, z):
    return phi(x)**T * phi(z)
K
#%% [python]
@latexify.function(use_math_symbols = True)
def K(x, z):
    return Sigma[i]*x[i]*z[i] * Sigma[j]*x[j]*z[j]
K
#%% [python]
@latexify.function(use_math_symbols = True)
def K(x, z):
    return Sigma[i]*Sigma[j]*x[i]*x[j]*z[i]*z[j]
K
#%% [python]
@latexify.function(use_math_symbols = True)
def K(x, z):
    return (x**T*z)**2
K
#%% [markdown]
"""
and `(x**T*z)` is only `O(n)`. That's quite an improvement. And even if we 
were to scale up to `K(x,z) = (x**T*z + c)**d`, this is still `O(n)` time. This 
means that you can have order `d` interaction terms with no increase in 
computation time.   

Personal note here, the promise of "infinite features" is a bit of a sleight of
hand, a click-bait headline. It's infinite interaction terms from a finite set
of features. You still can't get more signal than is contained in your starting
set of features, but SVM is going to wring out every last drop. 

But the SVM is the Optimal Margin Classifier combined with the Kernel trick, 
to give you decision boundaries in some arbitrary n-dimensional space but with
no increase in computation time. Neat. 

Dr. Ng actually also shows a visualization of what happens. We've got some y=1
points surrounded by y=0 points in a 2-d space. SVM projects up to a 3-d space (
where all the points sit on a cone), finds the hyperplane that linearly
separates y, and the intersection of the plane and the cone is projected back
down into 2D space to define the boundary, where it is very non-linear. It's a
fantastic way of visualizing what's going on. 

### Making Kernels

If x,z are similar, `K(x,z) = phi(x)*phi(z)` is large. If they're dissimilar, 
`K(x,z)` is small. We can think of Kernels as "similarity" functions. One similar
and very familiar similarity function is just our Guassian shape. So we might
consider using 
"""
#%% [python]
@latexify.function(use_math_symbols = True)
def K(x,z):
    return e**(-abs(abs(x-z))**2/(2*sigma**2))
K

#%% [markdown]
"""
And, it turns out, you _can_ use this kernel function. In fact, there's a theorem
that dictates if a kernel function is valid. It's called Mercer's theorem. It
roughly goes 

```
K is a valid kernal function iff for any d points, the corresponding kernel matrix K >= 0 (is positive semi-definite)
```

Proving it feels a little...   in the weeds, but Dr. Ng proves one half of it. 
But if we did the proof for the proposed Kernel function above, we could show
that it is valid. It's a pretty commonly used Kernel, called the "Gaussian
Kernel", and it gives you infinite combinations of your input features. 

Obviously, the Kernel Trick isn't exclusive to the Support Vector Machine, we
could take it and apply it to other learning algorithms, and it sounds like in 
the homework we do. Linear, Logistic, even Neural Nets can take advantage of it. 
Even PCA has a kernel-ized version. But in SVM it is considered absolutely 
foundational, whereas in others it's a nice-to-have. 
"""


#%% [markdown]
"""
### Soft-Margin SVM

In theory, we might always be able to find a high-enough-dimensional representation
of the data to always find a purely linearly separable boundary. But in practice
we may be comfortable having a bit of mis-classification. So soft-margin SVM
is really about drawing more natural and obvious boundaries that allow for some 
error. To do this we actually modify the constraint. 
"""

#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return min[[ {w,b,zeta} for w,b,zeta in (y[i](w**T*x[i] + b) >= 1 - zeta)]](abs(abs(w))**2 + C*Sigma[i]*zeta[i])
f

#%% [markdown]
"""
Another way to view it is discounting the effects of small numbers of
misclassified observations. And if we go through the work of posing the modification
as the dual-formed expression, we really only add one additional condition.
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return max[[alpha for alpha in {0 <= alpha[i] <= C, Sigma[i]*y[i]*alpha[i] == 0}]] * (Sigma[i]*alpha[i] - (1/2)*Sigma[i]*Sigma[j]*y[i]*y[j]*alpha[i]*alpha[j]*K(x[i],x[j]))
f
#%% [markdown]
"""
C is a hyperparameter here, and it controls that tradeoff between simple boundaries
and error. 

### Examples of Kernels

##### Linear

Gives you no polynomial stuff. Effectively not using a kernel at all. 
"""
#%% [python]
@latexify.function(use_math_symbols = True)
def K(x,z):
    return x**T*z
K

#%% [markdown]
"""
##### Polynomial

All interaction terms
"""
#%% [python]
@latexify.function(use_math_symbols = True)
def K(x,z):
    return (x**T*z)**d 
K

#%% [markdown]
"""
##### Gaussian

All interaction terms, but with a normal distribution flavor now. 
"""
#%% [python]
@latexify.function(use_math_symbols = True)
def K(x,z):
    return e**(-abs(abs(x-z))**2/(2*sigma**2))
K

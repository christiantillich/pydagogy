#%% [markdown]
"""
# Objective

Today we're covering the Bias/Variance tradeoff, regularization, and data 
splitting (traditional and CV)
"""

#%% [markdown]
"""
### Bias/Variance Tradeoff

Let's say we've got a housing price prediction problem, `sale price ~ size`. It's
not strictly linear, maybe logarithmic or something. We could fit it with a
line, we could fit it with a quadratic, we could fit it with a 5th order
polynomial. What we're describing here is a kindof Goldilocks scenario, linear
is too rigid, the fifth order polynomial is too flexible, and the quadratic
comes out "about right". The linear case is what we call "high bias", the
overfit case is what we call "high variance". 

The "variance" we're referring to in this case is the variance in different
model estimates you might get if you fit slightly different subsets multiple
times and compared model predictions. 

We also call this "high-variance" case "overfitting" - usually the estimates
or the decision boundary tries to absorb error into the model prediction as well.
"""

#%% [markdown]
"""
### Regularization

For linear regression, we can add an additional term to our objective function. 
"""

#%% [python]
import latexify
@latexify.expression(use_math_symbols = True)
def f(x):
    return min[theta] * (1/2) * Sigma[i] * abs(abs(y[i] - theta**T*x[i])) + (Lambda/2)*abs(abs(theta))**2
f

#%% [markdown]
"""
`Lambda` here adds a bias. We're no longer strictly minimizing the errors. But
it's a bias that penalizes model complexity. So we minimize the combination of
error and complexity, and adding this little extra bit pulls us towards simpler
models. This is often the preferred solution. 

A note on SVM, by minimizing the norm of the weights squared, we actually are
kindof doing a regularization just "baked in". Dr. Ng mentions the proof of this 
is pretty complex. 

Also, for spam filters, logistic regression by it's own will overfit pretty 
badly. And Naive Bayes is superior, but is still pretty bad. But LaSSO, the 
logistic regression with regularization, does really well. 

Why not regularize per-parameter? I.e. add a different lambda for each theta.
Answer: way more computationally expensive. Imagine doing it for 10,000
parameters. This is one way of accomodating different parameters at different
scales, but yknow what else works? Normalizing your variables. Much simpler. 

Given a training set `S`, we want theta to max out the likelihood given that set. 
"""

#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return P(theta | S) == P(S|theta) * P(theta) / P(S)
f

#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return argmax[theta] * P(theta | S) == argmax[theta] * (P(S|theta) * P(theta) / P(S))
f

#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return argmax[theta] * (Pi[i] * P(y[i]|x[i], theta)) * P(theta)
f

#%% [markdown]
"""
And if you assume `P(theta)` is Guassian, you can actually derive the 
regularization formula from above. So one way to think about regularization is
by saying that we want to maximize the likelihood of some parameters given some
data set, and then setting the prior of theta to be Guassian. Regularization
falls out of a Bayesian framing of the regression problem. 

### Train/Dev/Test Splits

So we're starting to conceptualize model training as not just fitting a single 
data set, but generalizing to the population behavior so that any new data
drawn from that population also is well-described by the rules. 

For any single training set, as we increase model complexity, error asymptotes
to 0 exponentially. But we're not concerned with optimizing a single set. We 
want a model that could work on new draws from the same population. That's
generalization error, and what you find is that error decreases up to a certain
point of model complexity, then starts increasing as we get more complex from 
there. Ultimately, we want to find this lowest point of error. There's a couple
procedures for this. 

Let's say we have a model problem, and we could fit any order polynomial that we
desired. What's the best one? One way to approach the problem is to fit several, 
and compare their out-of-sample estimates. Whichever provides the best 
out-of-sample accuracy is the winner. We split our data `S` into `S[train]`, 
`S[dev]` and `S[test]`. We provide the following method:

1. Train each candidate model on `S[train]`. 
2. Measure the error in `S[dev]`. Choose the model with the lowest `S[dev]` error.
3. Report actual error for the candidate model from `S[test]`. 

This works because candidates can only overfit on `S[train]`. They don't see
`S[dev]` till it's too late. But we also want `S[test]`, because the choice of
model is still an optimization problem that was fitted on `S[dev]`. To truly 
give an unbiased estimate of the final model error, you need a third test set
where no modeling decision has been made. Hence `S[test]`. 

Note that I've worked at places where the whole DS function swapped `dev` and 
`test` functionally. It's not really relevant, just makes for confusing 
conversations if you weren't expecting it. Always double check you're 
on the same page. 

There's an actual research paper on the topic of "CIFAR" where the researchers
were accused of unintentionally over-fitting to a training set. It definitely 
happens. 
"""

#%% [markdown]
"""
### Model Selection & CV

Train/test/split kindof lends itself to arbitrary rules, though. Most people do
70/30, or 60/20/20, or something. And if you have a data set that's say 10
million rows, you can sometimes find yourself asking "do I really need 2 million
obs in my training set?" Sometimes you do (advertising sets deal in small nudges
so sometimes you need millions). But on the whole, we might want to consider
something less arbitrary. 

Cross Validation is an interesting generalization of this idea. The above TTS
procedure we described is a special case, commonly titled Hold-Out Cross
Validation. But we'll describe the generalization below. 

We have a training set `S`. Divide it into `k` pieces (5 is typical for
illustration, 10 is typical in practice). The algo goes roughly like...

    For i in (1,k):
        m = model_train(S[i]) 
        scores[i] = m.test(S[-i])

    final_result = avg(scores)

This is all for the goal of choosing hyperparameters, evaluating different model
formulations, etc. As a final step, we fit the final model on the full data set.

This lets us come up with `k` dev sets while also utilizing the entire set. It's
a "no-waste" approach, and is even more exceptional on small data sets. `k in
[5,10,20]` is still kindof arbitrary, but is probably a kind of loosely optimal
balance between computation time (high `k`) and confidence in generalizeability
(low `k`). 

We can think about feature selection similarly as a kindof "each new feature
describes a new model" and approach feature selection in fundamentally the same
way. Step-wise regression, a non-regularized approach, functionally looks to 
greedily optimize the `S[dev]` performance for each added variable trained on 
`S[train]`. 
"""
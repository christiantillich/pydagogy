#%% [markdown]
"""
# Objective

We've discussed the mechanics of a lot of different learning algorithms so far. 
But now we need to pivot to advice on building things that work. This is perhaps
the hardest advice to actually apply. 
"""


#%% [markdown]
"""
### Bias v. Variance

It's very rare that the model works the first time we run it. Everything from
here on out is pertinent to what happens after this point. 

Let's start by assuming we're building an anti-spam classifier. we do 
LogReg with Regularization in a Bayesian context. When it doesn't work, people
typically dive into the following approaches. 

1. More training sample. 
1. Smaller set of features. 
1. Larger set of features. 
1. Invent new features. 
1. Run our algorithm longer. 
1. Change the fitting algorithm. 
1. Hyperparameter tune. 
1. Change the model paradigm entirely. 

And it's hard, from the armchair, to know which is the best option. 

What we should be doing is more like
1. Run diagnostics to figure out what the problem is. 
2. Apply the right fix. 

One of the most-common diagnostics that Dr. Ng uses is what he calls a 
Bias vs. Variance diagnostic. He says he uses this in pretty much every 
application. "If you can systematically apply it, it makes you much more 
efficient". 

So let's say our LogReg test error is 20%, and we are sure this is unacceptably
high. We can first explore the problem by asking if the poor performance is 
more due to high variance or high bias. 

If the problem is variance, plotting training size (m) vs. error rates will 
be helpful. As m increases, we should see test error decrease and training error
increase. But the key to focus on is how big the gap is between training error
and test error. The larger the gap, the more the problems with the model are 
actually related to high variance. The second thing to watch out for here is
where both of these curves asymptote - we want this thing to eventually converge
to some desired performance, so there should be some level of `m` that will 
take us there. 

If the problem is bias, by contrast, the gap between test and training eventually
gets small, but the difference between the asymptotic point of those lines and
the actual desired performance is large. In other words, even in the training
set we're not even approaching desired performance. Seeing this will easily show
that collecting data is not your problem. Moar data will never get you to where
you want to be. 

We can actually go through our feature list and classify these by what they fix.

1. More training sample [V]
1. Smaller set of features. [V] 
1. Larger set of features. [B]
1. Build new features. [B]
1. Run our fitting for longer. 
1. Change the fitting algorithm.
1. Hyperparameter tune. 
1. Change the model paradigm entirely.

Or, we can at least for the first four. Later in the discussion we'll start to 
incorporate some tools to address the last four. 

In general, the advice Dr. Ng has is to implement something quick and dirty first, 
then iterate heavily using this diagnostic plot as a guide. 

### Algorithm Convergence

We _should_ be able to plot the cost function value over the iteration number. 
We _should_ see it approach some asymptote. This is how we address whether 
running the fitting algorithm for longer will help. If this is your issue, you
might have the following questions:

1. Are you optimizing the _right_ cost function?
1. Do you have the correct hyperparameter values?
1. Do we get better convergence running different model paradigms? What can that
    teach us about hyperparam selection? About variable selection (e.g. does 
    svm find important interaction terms that helps a LogReg model?)

Dr. Ng goes over an example where we build the same model with a LogReg and 
SVM. And we take the SVM parameters and train the LogReg with that same set
of parameters. In theory, we should get the same cost functions and accuracy, but
if we don't, there's some diagnostic value in the two cases. 

In the first case, the cost function of the SVM is superior, and the accuracy of
the SVM is higher. This means that there is something funky with the optimization
algorithm itself, and there's not much to do here except try a new package (or
get to coding if you own it). 

But it's possible that the cost fuction is similar to or even better than the S
SVM, but the accuracy is _noticeably_ worse than the SVM. In other words, in 
spite of succeeding at optimizing the cost function, you get a worse set of 
parameters. This means that you're _optimizing the wrong cost_. You might have
to change the cost function you're using, or at least start exploring the
differences between the two cost functions. 

So we can revisit our fixes and classify them as fixing the algorithm or fixing
the objective. 

1. Run our fitting for longer. [A]
1. Change the fitting algorithm. [A]
1. Hyperparameter tune. [O]
1. Change the model paradigm entirely. [O]

This classification is true for the LogReg example anyways. I'm trying to abstract
the conclusions to be more general here, but the classification for some of these
items might change depending on algorithm specifics. Like, adding the hyper-param
`early_stopping` on an XGB is actually probably an algorithm fix, not an objective
function fix. But the important part here is really the questions we're asking. 

### The Helicoptor Example

Dr. Ng goes over a self-piloting helicopter. He said he did the following things

1. Build a simulation of the copter
1. Choose a cost function. 
1. Use reinforcement learning to minimize the cost in the simulation. 
1. Translate the learned parameters to flying on the real thing. 

Now, let's say we did all this and we don't have performance near a human, we
want to know what to do next:

1. Improve the simulator
1. Change the cost
1. Modify the RL algorithm. 

Many of these are very costly projects. Which one do you work on? We want some
diagnostic that helps us isolate the problem to one of these three areas. We have
the following diagnostics available

1. Does the helicopter fly well in the simulation?
    * If so, the issue is probably the simulation itself. It doesn't map to reality
        well, so we need to add stuff. Often, even adding noise into the simulation
        is helpful. 
1. Compute the cost function on the human. Is it lower than the cost of the learned params?
    * If not, your cost is actually probably incorrect. We want a cost function where
        the human's cost is lower, so our learner can approach that. 
1. Otherwise, the problem is the algorithm. The cost of the learned params should
    be moving towards the human, but for some reason isn't. 

The insertion of a simulated approach here is really insightful. It opens up
a number of discrete checkpoints where we can evaluate ourselves before doing
anything in the real world. It'd be really helpful to abstract this idea further, 
but unfortunately we have to make do with interpolating from this example. 
"""

#%% [markdown]
"""
### Error Analysis

Most learning algorithms, in practice, are actually complicated pipelines of 
machine learning models. The example of facial recognition includes a 
pipeline that has many small components like background removal, face detection, 
segmentation into eyes, nose and mouth components, and then regression on those
E/N/M components to identify if the features are a match. 

Being explicit about these components gives us, again, discrete checkpoints for
error analysis. For example, for the background component, we could use photoshop
to perfectly label the background in our training set. We could then run our model
pipeline using the two data sets - the automatically and manually removed background - 
and measure the difference in test accuracy using the two sets as input. What we
want here is the expected improvement in final accuracy given a change to the
background removal step. 

I take his point, but this seems like infeasible advice. At least, manually labeling
enough backgrounds that you could get meaningful end-of-pipeline accuracy seems
like an insurmountable task. Or, perhaps, he means you have to farm that work 
out. 

The general idea though - isolate a component of the pipeline, create the "ideal"
input to that pipeline, and measure end-of-pipe accuracy. Good idea, tricky in 
practice, I think. If you can get it, though, you have a really good map of 
priorities. 

Going the other way, though is much easier. This is called Ablative Analysis - 
remove components one at a time and monitor the degradation in accuracy. This isn't
so much useful as prioritizing new areas to improve, so much so as quantifying
the improvement from changes you've _already made_. 
"""
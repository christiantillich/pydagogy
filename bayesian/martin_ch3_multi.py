#%% [markdown]
"""
# Objective

The goal of this document is to build out the examples in Chapter 3 of Bayesian
Analysis with Python. Specifically we're starting to build simple multivariate
and even hierarchical models. 

# Analysis

In the first example, we start with a data set where we try to model a
normally-distributed variable, assigning estimates to both the mean and stdev of
that distribution. 
"""

#%% [python]
import numpy as np
import plotly.express as px
import pymc as pm
import arviz as az
import arviz_plots as azp

#%% [python]
data = np.array([51.06, 55.12, 53.73, 50.24, 52.05, 56.40, 48.45, 52.34,
55.65, 51.49, 51.86, 63.43, 53.00, 56.09, 51.93, 52.31, 52.33, 57.48, 57.44,
55.14, 53.93, 54.62, 56.09, 68.58, 51.36, 55.47, 50.73, 51.94, 54.95, 50.39,
52.91, 51.5, 52.68, 47.72, 49.73, 51.82, 54.99, 52.84, 53.19, 54.52, 51.46,
53.73, 51.61, 49.81, 52.42, 54.3, 53.84, 53.16])
px.histogram(data)


#%% [markdown]
"""
You can see we've got something approximately normal, with some heavy outliers.
So the book has us start by modeling with a normal dist. 
"""

#%% [python]
with pm.Model() as norm:
    # Priors
    mu = pm.Uniform('mu', 40, 75)
    sigma = pm.HalfNormal('sigma', sigma = 10)

    # Inferences
    y = pm.Normal('y', mu = mu, sigma = sigma, observed = data)
    trace_norm = pm.sample(1100)

#%% [markdown]
"""
Now, the book's approach to plotting is outdated. They use functions that have
been deprecated to extract the raw predictive posterior, then use `seaborn` to
plot the raw data. The new hotness is to lean on `az` as a data extraction
layer, then `azp` for the plots. But this adds some layers of abstraction now
that we have to work back through. 

Below, we show the `traceplot` replacement. `azp` does it natively, but we use
the `.viz['figure'].item()` call to traverse the wrapper and extract the
original plotly. Then we can do our normal plotly magic. 
"""
    
#%% [python]
(azp
    .plot_trace_dist(trace_norm.copy(), backend = "plotly")
    .viz['figure'].item() #Retrieves the actual plotly object from its wrappers
    .update_layout(width = 1000, height = 1000)
    .show()
)

#%% [python]
az.summary(trace_norm)

#%% [markdown]
"""
Now, what the book does next is generate 100 different synthetic data sets to
put into a plot to compare against the actual. It does this by sampling from the
predictive posterior for y. But the function it uses to do that sampling doesn't
exist anymore either - it's extraction, that's `az`. I show how you can do that
extraction below. We do have to tell `norm` to explicitly sample the posterior
predictive - it won't do it otherwise. But once we do, we can tell `az.extract`
to look for `y` in the posterior predictive. 
"""

#%% [python]
with norm: 
    pm.sample_posterior_predictive(trace_norm, extend_inferencedata=True)
y_pred = az.extract(trace_norm, num_samples = 100, group = "posterior_predictive")['y']

#%% [markdown]
"""
But we don't actually have to do this extraction either. `azp` will just give us
the plot we want. However, there's caveats. It gets the KDE wrong if we don't
explicitly tell it the limits for `y` can go as low as 30 (which we only get
from referencing against the book, nobody would tell you this). And the default
is to not even show the density axis, which is a little alarming if you see a
KDE with a smooshed tail and no y-axis and all of this abstraction makes the
whole process very opaque. 

The lesson here is that `azp` makes things kindof abstract, so check that
there's not some sort of meaningful parameter buried in the docs. 
"""

#%% [python]
(azp
    .plot_ppc_dist(
        trace_norm
        , num_samples=100
        , backend="plotly"
        , visuals={"remove_axis": False}
        , stats={
            "predictive_dist": {"bound_correction": False, "custom_lims": (30, 75)},
            "observed_dist":   {"bound_correction": False, "custom_lims": (30, 75)}
        },
    )
    .viz['figure'].item() #Retrieves the actual plotly object from its wrappers
    .update_layout(width = 1000, height = 1000)
    .show()
)

#%% [markdown]
"""
Anyway, back to the lesson. Before the plotting detour, what this plot tries to
show us is that the model might be poorly specified. The Gaussian assumption
produces a predictive distribution that's too narrow. A Student-T distribution,
on the other hand, would give us a lever to control tail weight, and costs us
one extra parameter. It's a good trade, and they illustrate how the switch
brings the diagram in line. 
"""

#%% [python]
with pm.Model() as model_t:
    # priors
    mu = pm.Uniform('mu', 40, 75)
    sigma = pm.HalfNormal('sigma', sigma = 10)
    nu = pm.Exponential('nu', 1/30)

    # inferences
    y = pm.StudentT('y', mu=mu, sigma = sigma, nu = nu, observed = data)
    trace_t = pm.sample(1100)

    # And sample the posterior predictive as well
    pm.sample_posterior_predictive(trace_t, extend_inferencedata=True)

az.summary(trace_t)

#%% [python]
(azp
    .plot_ppc_dist(
        trace_t
        , num_samples = 100
        , backend = "plotly"
        , visuals = {"remove_axis": False}
        , stats={
            #"predictive_dist": {"bound_correction": False, "custom_lims": (30, 75)},
            "observed_dist":   {"bound_correction": False, "custom_lims": (30, 75)}
        },
    )
    .viz['figure'].item() #Retrieves the actual plotly object from its wrappers
    .update_layout(width = 1000, height = 1000)
    .update_xaxes(range=[40, 75])  
    .show()
)

#%% [markdown]
"""
The key to getting this plot to work was to put bound correction on the observed
distribution, but not the predictive one. 

So, if I understand this correctly:
  1. We pass the fitted trace to plot ppc dist, telling it to use a plotly
     backend. 
  1. Plot ppc dist uses KDE under the hood. That KDE does estimate density
     everywhere, but something in `azp` under the hood sees the original input
     range, and chooses to only use the KDE estimations in the range of the
     original observed data as a default.   
  1. Internal bound correction logic takes this further. When `True`, it
     normalizes the distribution at the observed boundary and reflects the
     remaining probability mass back inward, artificially lifting the density
     further.  And it defaults to `True`. 
  1. So we need to turn the boundary correction off, to let `azp` know that we
     expect values lower than 47 even though that's the lowest observed values.
  1. We must only pass this correction to `observed_dist`. If we pass it to
     `predictive_dist`, it hard errors, because it sees all these potential
     distribution fits that contain values outside the named range. And because
     this is Student T's, fat tails are kindof the whole game, and we don't want
     to preclude that. We just need to tell plot ppc to be smarter about the
     range of our inputs. 
"""